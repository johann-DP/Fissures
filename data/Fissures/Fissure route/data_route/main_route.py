# -*- coding: utf-8 -*-
"""
Point d’entrée Dash – inclut les onglets “Distributions & Analyses” (heures et valeurs),
l’onglet de corrélation Heure/Valeur et l’onglet de prévisions Prophet.
"""
import os
import warnings
from datetime import datetime, date
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import paramiko

from data_loader import load_data, fetch_remote_csv
from aggregation import aggregate_by_half_hour, extract_extremes, calculate_daily_stats, calculate_confidence_intervals
from stats_calculator import (
    compute_confidence_intervals_hours,
    compute_extremes_stats,
    compute_parametric_ci,
    get_primary_peak_hours,
    fit_distributions_hours,
    compute_value_extremes_stats,
    compute_value_clusters,
    compute_value_cluster_stats,
)
from figures import (
    create_fig_main,
    create_fig_values,
    create_fig_hours,
    create_fig_jour,
    create_fig_hours_law_comparison,
    create_fig_qq_dual_laws,
    create_fig_value_step,
    create_fig_values_timeseries,
    create_fig_diff_min_max_timeseries,
    create_fig_values_law_comparison,
    create_fig_qq_values,
    create_fig_forecast
)
from time_calculator import get_extreme_half_hours, calculate_central_times
from time_calculator import compute_daily_extrema_timestamps
from prophet import Prophet  # import du modèle Prophet pour les prévisions

from pathlib import Path
from config import load_config
from logger import get_logger
import argparse
import sys

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- Paramètres
cfg = load_config()

LOCAL_CSV      = cfg["paths"]["local_csv"]
REMOTE_CSV     = cfg["paths"]["remote_csv"]

BACKUP_DIR     = cfg["paths"]["local_backup_dir"]
MIN_DT_SECONDS = cfg["analysis"]["min_interval_seconds"]
WINDOW_HALF    = cfg["analysis"]["window_half_width"]

hostname = cfg["ssh"]["host"]
port     = cfg["ssh"]["port"]
username = cfg["ssh"]["user"]
password = cfg["ssh"]["password"]


def parse_args(cfg):
    # -------- valeur par défaut (str ou date)
    default_start = cfg["analysis"]["start_day_default"]
    if isinstance(default_start, (datetime, date)):
        default_start = default_start.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Dash Fissure Route – analyse temporelle"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        metavar="YYYY-MM-DD",
        default=default_start,
        help="Premier jour inclus dans l’analyse (format ISO)",
    )
    args = parser.parse_args()

    # -------- normaliser ce qui vient d’argparse
    start_raw = args.start_date
    start_str = (
        start_raw.strftime("%Y-%m-%d")
        if isinstance(start_raw, (datetime, date))
        else start_raw
    )

    # -------- validations
    try:
        datetime.strptime(start_str, "%Y-%m-%d")
    except ValueError:
        sys.exit("⛔  --start-date doit être au format YYYY-MM-DD")

    if datetime.strptime(start_str, "%Y-%m-%d").date() >= datetime.now().date():
        sys.exit("⛔  --start-date ne peut pas être dans le futur")

    args.start_date = start_str          # on renvoie toujours une str
    return args


def print_system_status():
    console = Console()
    # Connexion SSH au Raspberry Pi
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port=port, username=username, password=password, timeout=10)

    # Disk usage sur /home
    stdin, stdout, stderr = client.exec_command(f"df -h /home/{username}")
    out = stdout.read().decode().splitlines()
    if len(out) > 1:
        parts = out[1].split()
        total, used, avail, percent = parts[1], parts[2], parts[3], parts[4]
    else:
        total = used = avail = percent = "N/A"

    # Memory usage
    stdin, stdout, stderr = client.exec_command("free -m | awk 'NR==2{print $2, $3, $3*100/$2}'")
    mem_vals = stdout.read().decode().split()
    if len(mem_vals) == 3:
        mem_total_m, mem_used_m, mem_percent = mem_vals
        mem_total_gb = float(mem_total_m) / 1024
        mem_used_gb = float(mem_used_m) / 1024
    else:
        mem_total_gb = mem_used_gb = mem_percent = "N/A"

    # Service status
    stdin, stdout, stderr = client.exec_command("systemctl is-active comparator.service")
    svc_status = stdout.read().decode().strip() or "unknown"

    # Last CSV timestamp
    stdin, stdout, stderr = client.exec_command(f"stat -c %y {REMOTE_CSV}")
    ts_line = stdout.read().decode().strip()
    last_ts = ts_line.split('.')[0] if ts_line else "unknown"

    client.close()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Resource", style="dim")
    table.add_column("Status")
    table.add_row("Disk Usage", f"{used}/{total} ({percent}) on Pi")
    table.add_row("Memory Usage", f"{mem_used_gb:.1f}/{mem_total_gb:.1f} GB ({float(mem_percent):.1f}%) on Pi")
    table.add_row("ComparatorSvc", svc_status)
    table.add_row("Last CSV update", last_ts)

    console.print(Panel(table, title="Raspberry Pi Status", expand=False))

    # résumé texte dans le log
    logger.info(
        f"Pi disk {used}/{total} ({percent}) | RAM {mem_used_gb:.1f}/{mem_total_gb:.1f} GB "
        f"({float(mem_percent):.1f} %) | Service={svc_status} | Last CSV={last_ts}"
        )


# ---------------------------------------------------------- 0. CLI --start-date
args = parse_args(cfg)

START_DAY_STR = args.start_date
END_DAY_STR   = datetime.now().strftime('%Y-%m-%d')

logger.info("Date de début retenue : %s", START_DAY_STR)

# ---------------------------------------------------------- 1. FETCH / LOAD
print_system_status()
fetch_remote_csv()
df = load_data(START_DAY_STR, END_DAY_STR)

# ─── Supprime les deux mesures erronées ────────────────────────────────────────
err_ts = [
    "2025-05-25T13:18:06.391823",
    "2025-05-25T13:18:06.687862"
]
df = df[~df['timestamp'].isin(pd.to_datetime(err_ts))]
# ───────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------- 2. Statistiques quotidiennes
daily_stats, gmin, gmax = calculate_daily_stats(df)
daily_stats = calculate_confidence_intervals(df, daily_stats)

# ---------------------------------------------------------- 3. Heures centrales des extrêmes
daily_extrema_df = compute_daily_extrema_timestamps(df)
min_times, max_times = calculate_central_times(df, daily_stats)
min_times = np.array(min_times, dtype=float)
max_times = np.array(max_times, dtype=float)

# ---------------------------------------------------------- 4. Agrégation par demi-heure
df_half_mean, df_half_med, jour_moyen, jour_median = aggregate_by_half_hour(df)
ext_hr = get_extreme_half_hours(df_half_mean, df_half_med)
extremes = extract_extremes(df_half_mean, df_half_med)


# ---------------------------------------------------------- 5. Bootstrap IC (½-h)
def _ic(series):
    res = compute_confidence_intervals_hours(series.dropna().values)
    lo, hi = res['ic_boot_mean_lo'], res['ic_boot_mean_hi']
    return (lo + hi) / 2, (hi - lo) / 2, f"IC bootstrap [{lo:.2f} h – {hi:.2f} h]"


centers = {}
for key, ser in (
        ('jour_moyen_max', ext_hr['max_half_times_mean']),
        ('jour_moyen_min', ext_hr['min_half_times_mean']),
        ('jour_median_max', ext_hr['max_half_times_median']),
        ('jour_median_min', ext_hr['min_half_times_median'])
):
    centers[key] = _ic(ser)

# ---------------------------------------------------------- 6. Statistiques d'extrêmes journaliers
ext_stats = compute_extremes_stats(df)

# 1. Heures brutes d'extrêmes
h_min = min_times
h_max = max_times

# 2. Isolation du cluster principal (heures)
h_max_cluster = get_primary_peak_hours(h_max, window_half_width=WINDOW_HALF)
h_min_cluster = get_primary_peak_hours(h_min, window_half_width=WINDOW_HALF)

# 3. Ajustement de lois et IC95 paramétriques sur le cluster (pour Jour MOYEN)
law_max_cl, params_max_cl = fit_distributions_hours(h_max_cluster)
ci_max_cl = compute_parametric_ci(law_max_cl, params_max_cl)
jm_center_max = ci_max_cl['mode']
jm_margin_max = (ci_max_cl['hi'] - ci_max_cl['lo']) / 2
jm_annot_max = (
        f"Cluster MAX → {law_max_cl}  params=["
        + ", ".join(f"{p:.2f}" for p in params_max_cl)
        + f"],  IC95=[{ci_max_cl['lo']:.2f},{ci_max_cl['hi']:.2f}]  mode={ci_max_cl['mode']:.2f}"
)

law_min_cl, params_min_cl = fit_distributions_hours(h_min_cluster)
ci_min_cl = compute_parametric_ci(law_min_cl, params_min_cl)
jm_center_min = ci_min_cl['mode']
jm_margin_min = (ci_min_cl['hi'] - ci_min_cl['lo']) / 2
jm_annot_min = (
        f"Cluster MIN → {law_min_cl}  params=["
        + ", ".join(f"{p:.2f}" for p in params_min_cl)
        + f"],  IC95=[{ci_min_cl['lo']:.2f},{ci_min_cl['hi']:.2f}]  mode={ci_min_cl['mode']:.2f}"
)

# Pour le jour médian, on réutilise les mêmes paramètres que jour moyen
md_center_max, md_margin_max, md_annot_max = jm_center_max, jm_margin_max, jm_annot_max
md_center_min, md_margin_min, md_annot_min = jm_center_min, jm_margin_min, jm_annot_min

# Debug : affichage console des résultats bootstrap sur extrêmes quotidiens
print("\n──────── Bootstrap : Heures extrêmes quotidiennes ────────")
for lab in ("max", "min"):
    s = ext_stats[lab]
    ci = s['ci']
    print(f"{lab.upper()} → {s['law']}  params={s['params']}  "
          f"µ CI95 [{ci['mean_lo']:.2f},{ci['mean_hi']:.2f}]  "
          f"médiane CI95 [{ci['median_lo']:.2f},{ci['median_hi']:.2f}]  "
          f"σ CI95 [{ci['sigma_lo']:.2f},{ci['sigma_hi']:.2f}]")

# ---------------------------------------------------------- 7. Préparation pour analyses des valeurs
# Bins de 0.01 inch et agrégation pour valeur vs heure
df['inch_bin'] = (df['inch'] // 0.01) * 0.01
df_bin_mean = df.groupby('inch_bin')['hour'].mean().reset_index()
df_bin_med = df.groupby('inch_bin')['hour'].median().reset_index()

# Couleurs utilisées dans les figures
colors = {
    'min': 'rgba(0,0,255,0.8)',  # bleu pour min
    'max': 'rgba(255,0,0,0.8)',  # rouge pour max
    'mean': 'rgba(0,128,0,0.8)',  # vert pour moyenne
    'median': 'rgba(255,165,0,0.8)'  # orange pour médiane
}

# ---------------------------------------------------------- 8. Création des figures
# Figure principale (série temporelle complète avec stats quotidiennes)
fig_main = create_fig_main(df, daily_stats, gmin, gmax, colors)

# Distributions des valeurs extrêmes (MAX/MIN) avec KDE et clusters
nbinsy_values = 25
fig_values = create_fig_values(daily_extrema_df, nbinsy_values, colors)

# Distributions des heures extrêmes (MIN/MAX) avec KDE et clusters
nbinsx_hours = 24
fig_hours = create_fig_hours(min_times, max_times, nbinsx_hours)

# Série temporelle des valeurs quotidiennes (min, max, moyenne, médiane)
fig_values_ts = create_fig_values_timeseries(daily_stats, colors)
# Série temporelle de la différence Max–Min chaque jour
fig_diff_ts = create_fig_diff_min_max_timeseries(daily_stats, gray_color='rgba(128,128,128,0.6)')

# Δ densités (écart empiriques vs théoriques) pour les heures des extrêmes (lois ajustées)
fig_diff_hours = create_fig_hours_law_comparison(
    max_times, min_times,
    # lois ajustées sur cluster (jours types)
    law_max_cl, params_max_cl,
    law_min_cl, params_min_cl,
    nbinsx_hours,
    title="Δ Densités – Heures des extrêmes"
)

# Calcul des lois optimales pour les valeurs extrêmes journalières (max et min)
val_ext_stats = compute_value_extremes_stats(daily_stats)
# Δ densités + PDF pour valeurs extrêmes (empirique vs lois ajustées)
fig_diff_values = create_fig_values_law_comparison(
    daily_stats['max'].values,
    daily_stats['min'].values,
    val_ext_stats['max']['law'], val_ext_stats['max']['params'],
    val_ext_stats['min']['law'], val_ext_stats['min']['params'],
    nbins=nbinsy_values,
    title="Δ Densités – Valeurs des extrêmes"
)

# QQ-plots des heures des extrêmes (global vs cluster)
fig_qq_hours = create_fig_qq_dual_laws(
    max_times, min_times,
    # lois globales ajustées (console)
    ext_stats['max']['law'], ext_stats['max']['params'],
    ext_stats['min']['law'], ext_stats['min']['params'],
    # lois ajustées sur clusters (jours types)
    law_max_cl, params_max_cl,
    law_min_cl, params_min_cl,
    title="QQ plots – Heures des extrêmes"
)

# Clusters de valeurs (autour du pic de densité principal) et lois ajustées sur ces clusters
val_clusters = compute_value_clusters(daily_stats, window_half_width=0.1)
val_cluster_stats = compute_value_cluster_stats(daily_stats, window_half_width=0.1)

# QQ-plots des valeurs extrêmes (global vs cluster, avec annotations sur lois clusters)
fig_qq_values = create_fig_qq_values(
    daily_stats['max'].values,
    daily_stats['min'].values,
    val_clusters['cluster_max'],
    val_clusters['cluster_min'],
    # lois globales ajustées
    val_ext_stats['max']['law'], val_ext_stats['max']['params'],
    val_ext_stats['min']['law'], val_ext_stats['min']['params'],
    # lois ajustées sur clusters
    val_cluster_stats['max']['law'], val_cluster_stats['max']['params'],
    val_cluster_stats['min']['law'], val_cluster_stats['min']['params'],
    title="QQ plots – Valeurs des extrêmes"
)

# Figures "Jour moyen" et "Jour médian" (évolution sur 24h)
fig_jour_moyen = create_fig_jour(
    jour_moyen, "Jour moyen",
    extremes['max_half_mean'], extremes['min_half_mean'],
    jm_annot_max, jm_annot_min,
    jm_center_max, jm_margin_max,
    jm_center_min, jm_margin_min,
    colors,
    format_half_hour_func=lambda hh: None,  # la fonction format_half_hour n'est pas utilisée dans la figure
)
fig_jour_median = create_fig_jour(
    jour_median, "Jour médian",
    extremes['max_half_median'], extremes['min_half_median'],
    md_annot_max, md_annot_min,
    md_center_max, md_margin_max,
    md_center_min, md_margin_min,
    colors,
    format_half_hour_func=lambda hh: None,
    base_data=jour_moyen
)

# Figures de corrélation Heure vs Valeur (heure moyenne et médiane auxquelles chaque valeur est atteinte)
# fig_val_step_mean = create_fig_value_step(df_bin_mean, "Heure moyenne vs Valeur", color=colors['mean'])
# fig_val_step_med = create_fig_value_step(df_bin_med, "Heure médiane vs Valeur", color=colors['median'])

# ---------------------------------------------------------- 9. Prévisions basées sur Prophet
# # Préparation des données pour Prophet
# df_prophet = df[['timestamp', 'inch']].rename(columns={'timestamp': 'ds', 'inch': 'y'})
# model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
# model.fit(df_prophet)
# # Génération des dates futures (7 jours à fréquence horaire)
# horizon_days = 7
# future = model.make_future_dataframe(periods=horizon_days * 24, freq='H', include_history=False)
# forecast_df = model.predict(future)
# # Figure de prévision (historique + projection avec IC95)
# fig_forecast = create_fig_forecast(df, forecast_df, horizon_days=horizon_days)

# ---------------------------------------------------------- 10. Configuration Dash et onglets
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Tabs(children=[
        dcc.Tab(label="Figure principale", children=[
            dcc.Graph(figure=fig_main, style={'height': '90vh'})
        ]),
        dcc.Tab(label="Distributions & Analyses : heures", children=[
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_values, style={'height': '45vh'}),
                    dcc.Graph(figure=fig_hours, style={'height': '45vh'}),
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=fig_jour_moyen, style={'height': '45vh'}),
                    dcc.Graph(figure=fig_jour_median, style={'height': '45vh'})
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=fig_diff_hours, style={'height': '45vh'}),
                    dcc.Graph(figure=fig_qq_hours, style={'height': '45vh'})
                ], style={'width': '33%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label="Distributions & Analyses : valeurs", children=[
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_values, style={'height': '45vh'}),
                    dcc.Graph(figure=fig_hours, style={'height': '45vh'}),
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=fig_values_ts, style={'height': '45vh'}),
                    dcc.Graph(figure=fig_diff_ts, style={'height': '45vh'})
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=fig_diff_values, style={'height': '45vh'}),
                    dcc.Graph(figure=fig_qq_values, style={'height': '45vh'})
                ], style={'width': '33%', 'display': 'inline-block'})
            ])
        ]),
        # dcc.Tab(label="Heure vs Valeur", children=[
        #     html.Div([
        #         dcc.Graph(figure=fig_val_step_mean, style={'height': '50vh'}),
        #         dcc.Graph(figure=fig_val_step_med, style={'height': '50vh'})
        #     ])
        # ]),
        # dcc.Tab(label="Prévisions", children=[
        #     dcc.Graph(figure=fig_forecast, style={'height': '90vh'})
        # ])
    ])
])

if __name__ == "__main__":
    app.run_server(debug=False, port=8051)
