# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import t

from logger import get_logger
logger = get_logger(__name__)


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
def calculate_confidence_intervals(df: pd.DataFrame,
                                   daily_stats: pd.DataFrame,
                                   force_normal_test: bool = True):
    ci_lo, ci_hi, normal = [], [], []
    for day, g in df.groupby("day")['inch']:
        n = len(g)
        if n < 3:
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            normal.append(False)
            continue
        m, s = g.mean(), g.std(ddof=1)
        margin = t.ppf(0.975, n - 1) * s / np.sqrt(n)
        ci_lo.append(m - margin)
        ci_hi.append(m + margin)
        normal.append(force_normal_test)
    daily_stats = daily_stats.copy()
    daily_stats['ci_lower'] = ci_lo
    daily_stats['ci_upper'] = ci_hi
    daily_stats['normal'] = normal
    return daily_stats


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
