# -*- coding: utf-8 -*-
import os
import shutil
import paramiko
import pandas as pd
from datetime import datetime, timedelta

from pathlib import Path
from config import load_config
cfg = load_config()


def fetch_remote_csv() -> None:
    """
    Copie le CSV distant vers le chemin local défini dans config.yaml.
    Sauvegarde horodatée du fichier local existant dans `local_backup_dir`.
    """
    local_path   = Path(cfg["paths"]["local_csv"])
    remote_path  = cfg["paths"]["remote_csv"]
    backup_dir   = Path(cfg["paths"]["local_backup_dir"])
    host         = cfg["ssh"]["host"]
    port         = cfg["ssh"]["port"]
    user         = cfg["ssh"]["user"]
    password     = cfg["ssh"]["password"]

    backup_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Sauvegarde locale (si le fichier existe)
    if local_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{local_path.stem}_{stamp}{local_path.suffix}"
        shutil.copy2(local_path, backup_file)
        print(f"Sauvegarde effectuée → {backup_file}")
    else:
        print("Aucun fichier local à sauvegarder (premier import).")

    # ---------- Transfert SFTP
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=user, password=password)
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.get(remote_path, str(local_path))
        transport.close()
        print(f"Transfert réussi : {remote_path} → {local_path}")
    except Exception as exc:
        print(f"[ERREUR] transfert SFTP : {exc}")


def load_data(start_day_str: str, end_day_str: str) -> pd.DataFrame:
    """
    Charge le CSV `local_path` dans un DataFrame, applique les filtres temporels
    et renvoie le DataFrame filtré et formaté (colonnes additionnelles, etc.).
    """
    start_day = pd.to_datetime(start_day_str)
    end_day = pd.to_datetime(end_day_str)

    local_path = cfg["paths"]["local_csv"]  # depuis le YAML

    df = pd.read_csv(local_path, header=None)
    df.columns = ['timestamp', 'inch']
    df['inch'] = pd.to_numeric(df['inch'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Filtre temporel (du début à la fin, fin inclus jusqu'à 23:59 du jour end_day)
    mask_time = (df['timestamp'] >= start_day) & (df['timestamp'] < (end_day + pd.Timedelta(days=1)))
    df = df[mask_time]

    # Filtre sur la fréquence (conservation des points espacés d'au moins 10s ou premier point)
    mask_freq = (df['timestamp'].diff() >= pd.Timedelta(seconds=10)) | (df['timestamp'].diff().isna())
    df = df[mask_freq]

    # Ajout de colonnes temporelles et unité métrique
    df['day'] = df['timestamp'].dt.date
    df['mm'] = df['inch'] * 25.4
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60 + df['timestamp'].dt.second / 3600
    df['hour_bin'] = df['timestamp'].dt.hour
    df['half_hour'] = df['timestamp'].dt.hour + (df['timestamp'].dt.minute // 30) * 0.5

    return df
