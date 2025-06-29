# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import t
from scipy import stats

from logger import get_logger
logger = get_logger(__name__)

log = get_logger(__name__)


# ---------------------------------------------------------------- daily-stats
def calculate_daily_stats(df: pd.DataFrame):
    """
    Calcule pour chaque jour :
      - min et max en excluant :
          • les mesures EXACTEMENT à 00:00 ou 24:00,
          • tout palier traversant ces bordures (paliers dont l’un des extrêmes est à 00:00 ou 24:00),
      - mean, median (toujours sur toutes les mesures),
      - day_start, day_end, noon,
      - diff_mm = (max - min)*25.4,
      - diff_global_max_mm et diff_global_min_mm.
    """
    from time_calculator import compute_daily_extrema_timestamps

    # Préparer timestamps et date
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['timestamp'].dt.date

    # 1) Moyenne et médiane sur toutes les mesures
    daily = (
        df.groupby('day')['inch']
          .agg(['mean', 'median'])
          .reset_index()
    )

    # 2) Calcul des bornes temporelles
    daily['day_start'] = pd.to_datetime(daily['day'])
    daily['day_end']   = daily['day_start'] + pd.Timedelta(days=1)
    daily['noon']      = daily['day_start'] + pd.Timedelta(hours=12)

    # 3) Récupérer min/max absolus corrects via compute_daily_extrema_timestamps
    ext = compute_daily_extrema_timestamps(df)
    ext = ext.set_index('day')

    # 4) Assigner val_min et val_max au DataFrame daily
    daily['min'] = np.nan
    daily['max'] = np.nan
    for idx, row in daily.iterrows():
        d = row['day']
        if d in ext.index:
            daily.at[idx, 'min'] = ext.at[d, 'val_min']
            daily.at[idx, 'max'] = ext.at[d, 'val_max']
        # sinon laisser NaN
        else:
            logger.info(f"aggregation.py/calculate_daily_stats:{d}, day not in index")

    # 5) diff_mm
    daily['diff_mm'] = (daily['max'] - daily['min']) * 25.4

    # 6) Global min/max sur toutes les données
    gmin = df['inch'].min()
    gmax = df['inch'].max()
    daily['diff_global_max_mm'] = (gmax - daily['max']) * 25.4
    daily['diff_global_min_mm'] = (daily['min'] - gmin) * 25.4

    return daily, gmin, gmax


# ------------------------------------------------------- Student + bootstrap
def _bootstrap_ci(sample: np.ndarray,
                  alpha: float = .05,
                  B: int = 1_500) -> Tuple[float, float]:
    """
    IC non‑paramétrique (percentile) pour la moyenne d'un échantillon 1‑D.
    Retourne (ci_lower, ci_upper).  Renvoie (nan, nan) si len(sample) < 3.
    """
    n = sample.size
    if n < 3:
        return np.nan, np.nan

    rng    = np.random.default_rng()
    draws  = rng.integers(0, n, size=(B, n))     # matrice (B, n)
    boots  = sample[draws].mean(axis=1)          # moyenne sur l'axe n
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1 - alpha/2)])
    return float(lo), float(hi)


def calculate_confidence_intervals(df: pd.DataFrame,
                                   daily_stats: pd.DataFrame,
                                   *,
                                   force_normal_test: bool = False,
                                   alpha: float = .05,
                                   B: int = 1_500) -> pd.DataFrame:
    """
    Ajoute :
        ci_lower,  ci_upper      (Student t, inch → normal présumé)
        ci_lower_np, ci_upper_np (bootstrap np pour même échantillon)
        normal (bool)            True ↔ utiliser IC Student ; False ↔ IC bootstrap
    Les colonnes historiques restent donc intactes.
    """
    ci_lo, ci_hi, ci_lo_np, ci_hi_np, normal = [], [], [], [], []

    for day, g in df.groupby("day")['inch']:
        arr = g.to_numpy()
        n   = arr.size

        # ----------- cas n < 3  ------------------------------------------
        if n < 3:
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            ci_lo_np.append(np.nan)
            ci_hi_np.append(np.nan)
            normal.append(False)
            continue

        # ----------- IC Student‑t (héritage) -----------------------------
        m, s  = arr.mean(), arr.std(ddof=1)
        margin = t.ppf(1 - alpha/2, n - 1) * s / np.sqrt(n)
        ci_lo.append(m - margin)
        ci_hi.append(m + margin)
        normal.append(force_normal_test)  # force_normal_test = True ? → rectangles verts

        # ----------- IC bootstrap ---------------------------------------
        lo_np, hi_np = _bootstrap_ci(arr, alpha=alpha, B=B)
        ci_lo_np.append(lo_np)
        ci_hi_np.append(hi_np)

    out = daily_stats.copy()
    out['ci_lower']     = ci_lo
    out['ci_upper']     = ci_hi
    out['ci_lower_np']  = ci_lo_np
    out['ci_upper_np']  = ci_hi_np
    out['normal']       = normal

    return out


# ---------------------------------------------------- agrégation ½-heure etc.
def aggregate_by_half_hour(df: pd.DataFrame):
    df_mean = df.groupby(['day', 'half_hour'])['inch'].mean().reset_index()
    df_median = df.groupby(['day', 'half_hour'])['inch'].median().reset_index()

    grid = pd.DataFrame({'half_hour': np.arange(0, 24, 0.5)})
    jour_moyen = grid.merge(df_mean.groupby('half_hour')['inch'].mean().reset_index(),
                            on='half_hour', how='left')
    jour_median = grid.merge(df_median.groupby('half_hour')['inch'].median().reset_index(),
                             on='half_hour', how='left')
    return df_mean, df_median, jour_moyen, jour_median


def extract_extremes(df_mean: pd.DataFrame, df_median: pd.DataFrame) -> Dict[str, object]:
    jm = df_mean.groupby('half_hour')['inch'].mean().reset_index()
    jmed = df_median.groupby('half_hour')['inch'].median().reset_index()

    max_hh_mean = jm.loc[jm['inch'].idxmax(), 'half_hour']
    min_hh_mean = jm.loc[jm['inch'].idxmin(), 'half_hour']
    max_hh_med = jmed.loc[jmed['inch'].idxmax(), 'half_hour']
    min_hh_med = jmed.loc[jmed['inch'].idxmin(), 'half_hour']

    return {
        'max_half_mean': max_hh_mean,
        'min_half_mean': min_hh_mean,
        'data_max_mean': df_mean.loc[df_mean['half_hour'] == max_hh_mean, 'inch'],
        'data_min_mean': df_mean.loc[df_mean['half_hour'] == min_hh_mean, 'inch'],
        'max_half_median': max_hh_med,
        'min_half_median': min_hh_med,
        'data_max_median': df_median.loc[df_median['half_hour'] == max_hh_med, 'inch'],
        'data_min_median': df_median.loc[df_median['half_hour'] == min_hh_med, 'inch'],
    }
