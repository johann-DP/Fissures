# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import shapiro, jarque_bera, kstest, norm, t
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
    # lo, hi = np.percentile(boots, [100*alpha/2, 100*(1 - alpha/2)])
    return np.percentile(boots, [100*alpha/2, 100*(1 - alpha/2)]) # float(lo), float(hi)


def _normality_pvals(x: np.ndarray) -> Dict[str, float]:
    """Renvoie les p‑values des trois tests ; NaN si effectif < 3."""
    if x.size < 3:
        return dict(sw=np.nan, jb=np.nan, ks=np.nan)
    sw_p = shapiro(x).pvalue
    jb_p = jarque_bera(x).pvalue
    ks_p = kstest(x, 'norm',
                  args=(x.mean(), x.std(ddof=1))).pvalue
    return dict(sw=sw_p, jb=jb_p, ks=ks_p)


def calculate_confidence_intervals(df: pd.DataFrame,
                                   daily_stats: pd.DataFrame,
                                   *,
                                   ALPHA_NORM = .05,
                                   alpha: float = .05,
                                   B: int = 1_500) -> pd.DataFrame:
    """
    Ajoute :
        ci_lower,  ci_upper      (Student t, inch → normal présumé)
        ci_lower_np, ci_upper_np (bootstrap np pour même échantillon)
        normal (bool)            True ↔ utiliser IC Student ; False ↔ IC bootstrap
        p‑values normalité.
    """
    ci_lo, ci_hi, ci_lo_np, ci_hi_np = [], [], [], []
    normal, p_sw, p_jb, p_ks = [], [], [], []

    for day, g in df.groupby("day")['inch']:
        arr = g.to_numpy()
        n = arr.size

        # ---------- normality tests ------------------------------------
        pvals = _normality_pvals(arr)
        is_normal = (
                pvals['sw'] > ALPHA_NORM and
                pvals['jb'] > ALPHA_NORM and
                pvals['ks'] > ALPHA_NORM
        )
        normal.append(is_normal)
        p_sw.append(pvals['sw'])
        p_jb.append(pvals['jb'])
        p_ks.append(pvals['ks'])

        # ---------- IC Student (possible même si pas normal) ----------
        if n >= 3:
            mean, sd = arr.mean(), arr.std(ddof=1)
            margin = t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
            ci_lo.append(mean - margin)
            ci_hi.append(mean + margin)
        else:
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)

        # ---------- IC bootstrap --------------------------------------
        lo_np, hi_np = _bootstrap_ci(arr, alpha, B)
        ci_lo_np.append(lo_np)
        ci_hi_np.append(hi_np)

    out = daily_stats.copy()
    out['ci_lower'] = ci_lo
    out['ci_upper'] = ci_hi
    out['ci_lower_np'] = ci_lo_np
    out['ci_upper_np'] = ci_hi_np
    out['normal'] = normal
    out['p_sw'] = p_sw
    out['p_jb'] = p_jb
    out['p_ks'] = p_ks
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
