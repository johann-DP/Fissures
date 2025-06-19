import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime
import os

# --- Chargement des données (inch & timestamp) ---
def load_data(local_path, start_day_str, end_day_str):
    start = pd.to_datetime(start_day_str)
    end   = pd.to_datetime(end_day_str) + pd.Timedelta(days=1)
    df = pd.read_csv(local_path, header=None)
    df.columns = ['timestamp','inch']
    df['inch'] = pd.to_numeric(df['inch'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
    df = df[(df['timestamp'].diff() >= pd.Timedelta(seconds=10)) | df['timestamp'].diff().isna()]
    df['day'] = df['timestamp'].dt.date
    return df

# --- Raffinement du palier autour de chaque extrémum initial ---
def refine_window_extremum(res, vals, times,
                           window_min: float,
                           tol_cand: float,
                           tol_plateau: float):
    """
    Raffine les extrêmes LOWESS en :
      - n'utilisant ±window_min minutes que pour déceler un candidat v_cand,
      - quand pas de v_cand, sélectionne le run global de v0 contenant raw_i,
      - centre l'horodatage sur le milieu exact du run retenu.
    """

    def center_of_run(s, e):
        """Retourne le timestamp milieu du run [s,e]."""
        t0, t1 = times.iloc[s], times.iloc[e]
        return t0 + (t1 - t0) / 2

    def find_runs(idxs):
        """Transforme une liste triée d’indices en runs contigus [(s,e),…]."""
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

    def refine_one(key_t, key_v, is_max):
        # 1) position initiale
        diffs = (times - res[key_t]).abs().values
        raw_i = int(np.argmin(diffs))
        v0    = res[key_v]

        # 2) définir la fenêtre locale pour chercher v_cand
        t_center = res[key_t]
        t_lower  = t_center - pd.Timedelta(minutes=window_min)
        t_upper  = t_center + pd.Timedelta(minutes=window_min)
        within   = (times >= t_lower) & (times <= t_upper)
        win_idxs = np.nonzero(within)[0]

        # 3) détecter un candidat plus extrême dans la fenêtre
        v_cand = None
        if win_idxs.size:
            seg = vals[win_idxs]
            extremum = seg.max() if is_max else seg.min()
            cond = (is_max and extremum > v0 + tol_cand) \
                or (not is_max and extremum < v0 - tol_cand)
            if cond:
                v_cand = extremum

        if v_cand is not None:
            # 4a) nouveau palier → toutes occurrences dans la fenêtre
            cand_mask = within & np.isclose(vals, v_cand, atol=tol_plateau)
            idxs_ext  = np.nonzero(cand_mask)[0].tolist()
            runs      = find_runs(idxs_ext)
            # choisir run contenant raw_i ou, sinon, le plus long
            run = next((r for r in runs if r[0] <= raw_i <= r[1]), None)
            if run is None and runs:
                run = max(runs, key=lambda r: r[1] - r[0])
            if run:
                s, e = run
                res[key_v] = v_cand
                res[key_t] = center_of_run(s, e)

        else:
            # 4b) pas de v_cand → **run global de v0** (toute la série)
            plateaus = np.nonzero(np.isclose(vals, v0, atol=tol_plateau))[0].tolist()
            runs0    = find_runs(plateaus)
            # run global contenant raw_i ou, sinon, le plus long
            run = next((r for r in runs0 if r[0] <= raw_i <= r[1]), None)
            if run is None and runs0:
                run = max(runs0, key=lambda r: r[1] - r[0])
            if run:
                s, e = run
                res[key_v] = v0
                res[key_t] = center_of_run(s, e)
        # sinon on garde LOWESS

    # raffinement du max puis du min
    refine_one("t_max", "v_max", True)
    refine_one("t_min", "v_min", False)
    return res


def method_lowess(vals, times, frac,
                  tol_cand, tol_plateau,
                  margin, window_min):
    trend = lowess(vals, np.arange(len(vals)), frac=frac, return_sorted=False)
    d = np.diff(trend)
    idx_max = [i+1 for i in range(len(d)-1) if d[i]>0 and d[i+1]<0]
    idx_min = [i+1 for i in range(len(d)-1) if d[i]<0 and d[i+1]>0]
    interior = range(margin, len(vals)-margin)
    peaks   = [i for i in idx_max if i in interior]
    valleys = [i for i in idx_min if i in interior]
    if not peaks or not valleys:
        return None

    max_idx = max(peaks, key=lambda i: vals[i])
    min_idx = min(valleys, key=lambda i: vals[i])
    res = {
        't_max': times.iloc[max_idx],
        'v_max': vals[max_idx],
        't_min': times.iloc[min_idx],
        'v_min': vals[min_idx],
        'trend': trend
    }
    # appel avec les deux tolérances
    return refine_window_extremum(res, vals, times,
                                  window_min,
                                  tol_cand,
                                  tol_plateau)


if __name__ == "__main__":
    SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
    CSV_PATH    = os.path.join(SCRIPT_DIR, "measurements.csv")
    START, END  = "2025-03-15", datetime.now().strftime("%Y-%m-%d")
    TOL         = 1e-3
    TOL_PLATEAU = 1e-6
    LOWESS_FRAC = 0.05
    MARGIN      = 10
    WINDOW_MIN  = 20
    OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "lowess_refined")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(CSV_PATH, START, END)
    for day, sub in df.groupby("day"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        vals  = sub["inch"].to_numpy()
        times = sub["timestamp"]
        if len(vals) < MARGIN*2 + 3: continue

        res = method_lowess(
             vals,
             times,
             frac = LOWESS_FRAC,
             tol_cand = TOL,
             tol_plateau = TOL_PLATEAU,
             margin = MARGIN,
             window_min = WINDOW_MIN
        )
        if res is None:
            print(f"{day}: pas d'extrêmes détectés")
            continue

        print(f"{day} → Max raffiné @{res['t_max']} ({res['v_max']:.4f}), "
              f"Min raffiné @{res['t_min']} ({res['v_min']:.4f})")

        plt.figure(figsize=(10,4))
        plt.plot(times, vals, '-', color='grey', alpha=0.6, label='Mesures')
        plt.plot(times, res['trend'], 'k--', label='Trend LOWESS')
        plt.scatter([res['t_max']], [res['v_max']], color='red',   s=80, label='Max raffiné')
        plt.scatter([res['t_min']], [res['v_min']], color='blue',  s=80, label='Min raffiné')
        plt.title(f"{day} - LOWESS only + window refinement")
        plt.xlabel("Timestamp"); plt.ylabel("inch")
        plt.legend(loc='upper left', fontsize='small')
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"{day}.png")
        plt.savefig(out)
        print("Figure enregistrée:", out)
        #plt.show()
