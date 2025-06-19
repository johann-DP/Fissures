# time_calculator.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def calculate_central_times(df, daily_stats=None):
    """
    Renvoie deux arrays (min_times, max_times) en heures décimales, basés sur
    compute_daily_extrema_timestamps(df). daily_stats est ignoré.
    """

    daily_ext = compute_daily_extrema_timestamps(df)
    # Trier par jour pour l’ordre
    daily_ext = daily_ext.sort_values("day")

    # Convertir timestamps en heures décimales
    to_hours = lambda ts: ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600

    max_times = to_hours(daily_ext["time_max"])
    min_times = to_hours(daily_ext["time_min"])

    return np.array(min_times, dtype=float), np.array(max_times, dtype=float)


def compute_daily_extrema_timestamps(
        df: pd.DataFrame,
        neighbor_layers: int = 4
) -> pd.DataFrame:
    """
    Calcule pour chaque jour (sauf premier et dernier) les extrema (min et max)
    en utilisant LOWESS pour lisser et un raffinage local autour de chaque extrême.

    Paramètres
    ----------
    df : pd.DataFrame
        Colonnes ['timestamp','inch'].
    neighbor_layers : int, optionnel (défaut=4)
        Utilisé comme `margin` pour exclure les points de bord, équivalent au
        nombre de points à ignorer au début et à la fin de chaque journée.

    Retour
    ------
    pd.DataFrame
        Colonnes ['day','time_max','val_max','time_min','val_min'].
    """
    # -- Paramètres internes pour LOWESS et raffinage --
    frac = 0.05
    tol_cand = 1e-3
    tol_plateau = 1e-6
    margin = neighbor_layers
    window_min = 20

    def refine_window_extremum(res, vals, times):
        """
        Raffine les extrêmes initiaux en cherchant, dans une fenêtre de ±window_min
        minutes, un candidat plus extrême (tol_cand) sur un plateau (tol_plateau),
        ou en recentrant sur le run global.
        """
        def center_of_run(s, e):
            t0, t1 = times.iloc[s], times.iloc[e]
            return t0 + (t1 - t0) / 2

        def find_runs(idxs):
            runs = []
            if not idxs:
                return runs
            start = prev = idxs[0]
            for i in idxs[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    runs.append((start, prev))
                    start = prev = i
            runs.append((start, prev))
            return runs

        for key_t, key_v, is_max in [("t_max","v_max",True), ("t_min","v_min",False)]:
            diffs = (times - res[key_t]).abs().values
            raw_i = int(np.argmin(diffs))
            v0 = res[key_v]

            t_center = res[key_t]
            lower = t_center - pd.Timedelta(minutes=window_min)
            upper = t_center + pd.Timedelta(minutes=window_min)
            mask = (times >= lower) & (times <= upper)
            win_idxs = np.nonzero(mask)[0]

            v_cand = None
            if win_idxs.size:
                seg = vals[win_idxs]
                extremum = seg.max() if is_max else seg.min()
                cond = (is_max and extremum > v0 + tol_cand) or (not is_max and extremum < v0 - tol_cand)
                if cond:
                    v_cand = extremum

            if v_cand is not None:
                cand_mask = mask & np.isclose(vals, v_cand, atol=tol_plateau)
                idxs_ext = np.nonzero(cand_mask)[0].tolist()
                runs = find_runs(idxs_ext)
                run = next((r for r in runs if r[0] <= raw_i <= r[1]), None) \
                      or (max(runs, key=lambda r: r[1]-r[0]) if runs else None)
                if run:
                    s, e = run
                    res[key_v] = v_cand
                    res[key_t] = center_of_run(s, e)
            else:
                plateaus = np.nonzero(np.isclose(vals, v0, atol=tol_plateau))[0].tolist()
                runs0 = find_runs(plateaus)
                run = next((r for r in runs0 if r[0] <= raw_i <= r[1]), None) \
                      or (max(runs0, key=lambda r: r[1]-r[0]) if runs0 else None)
                if run:
                    s, e = run
                    res[key_v] = v0
                    res[key_t] = center_of_run(s, e)
        return res

    def method_lowess(vals, times):
        trend = lowess(vals, np.arange(len(vals)), frac=frac, return_sorted=False)
        d = np.diff(trend)
        idx_max = [i+1 for i in range(len(d)-1) if d[i]>0 and d[i+1]<0]
        idx_min = [i+1 for i in range(len(d)-1) if d[i]<0 and d[i+1]>0]
        interior = range(margin, len(vals)-margin)
        peaks   = [i for i in idx_max if i in interior]
        valleys = [i for i in idx_min if i in interior]
        if not peaks or not valleys:
            return None
        max_i = max(peaks, key=lambda i: vals[i])
        min_i = min(valleys, key=lambda i: vals[i])
        res = {'t_max': times.iloc[max_i], 'v_max': vals[max_i],
               't_min': times.iloc[min_i], 'v_min': vals[min_i]}
        return refine_window_extremum(res, vals, times)

    # Préparation
    df2 = df.copy()
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df2 = df2.sort_values('timestamp')
    df2['day'] = df2['timestamp'].dt.date

    days = sorted(df2['day'].unique())
    if len(days) <= 2:
        return pd.DataFrame(columns=['day','time_max','val_max','time_min','val_min'])

    first_day, last_day = days[0], days[-1]
    records = []

    for day, grp in df2.groupby('day'):
        if day in (first_day, last_day):
            continue
        sub = grp.sort_values('timestamp').reset_index(drop=True)
        if len(sub) < margin*2 + 3:
            continue
        vals = sub['inch'].to_numpy()
        times = sub['timestamp']
        out = method_lowess(vals, times)
        if out is None:
            continue
        records.append({
            'day': day,
            'time_max': out['t_max'],
            'val_max': out['v_max'],
            'time_min': out['t_min'],
            'val_min': out['v_min']
        })

    return pd.DataFrame(records)


def get_extreme_half_hours(df_day_half_mean, df_day_half_median):
    """
    Récupère, pour chaque jour, l'heure (en demi-heure décimale) où se produit
    le max/min dans df_day_half_mean et df_day_half_median.
    """
    max_half_times_mean = df_day_half_mean.groupby('day').apply(
        lambda grp: grp.loc[grp['inch'].idxmax(), 'half_hour']
    )
    min_half_times_mean = df_day_half_mean.groupby('day').apply(
        lambda grp: grp.loc[grp['inch'].idxmin(), 'half_hour']
    )
    max_half_times_median = df_day_half_median.groupby('day').apply(
        lambda grp: grp.loc[grp['inch'].idxmax(), 'half_hour']
    )
    min_half_times_median = df_day_half_median.groupby('day').apply(
        lambda grp: grp.loc[grp['inch'].idxmin(), 'half_hour']
    )

    return {
        "max_half_times_mean": max_half_times_mean,
        "min_half_times_mean": min_half_times_mean,
        "max_half_times_median": max_half_times_median,
        "min_half_times_median": min_half_times_median
    }
