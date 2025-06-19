# -*- coding: utf-8 -*-
from __future__ import annotations
import math, warnings
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import special as sps
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.stats import truncnorm, logistic, weibull_min, norm

# -------------------------------------------------------------------- config
A, B = 0.0, 24.0
TWOPI_24 = 2 * math.pi / 24
warnings.filterwarnings("ignore", category=RuntimeWarning)

LAWS: Dict[str, st.rv_continuous] = {
    # "uniform": st.uniform,
    "vonmises": st.vonmises,
    "truncnorm": st.truncnorm,
    "logistic": st.logistic,
    "weibull_min": st.weibull_min,
}


# ----------------------------------------------------------------- helpers
def _loglik_trunc(dist, x: np.ndarray, p: Tuple) -> Tuple[float, float]:
    m = dist.cdf(B, *p) - dist.cdf(A, *p)
    if m <= .90:  # ≥10 % hors support ⇒ éliminé
        return -math.inf, math.inf
    return dist.logpdf(x, *p).sum() - len(x) * math.log(m), -math.log(m)


def _dir_est_vonmises(theta: np.ndarray) -> Tuple[float, float]:
    C, S = np.cos(theta).mean(), np.sin(theta).mean()
    mu = math.atan2(S, C)
    R = math.hypot(C, S)
    if R < 1e-6:
        return 0.0, mu
    k = (2 * R + R ** 3 + 5 * R ** 5 / 6) if R < .53 else \
        (-.4 + 1.39 / (1 - R) + .43 / (1 - R) ** 2) if R < .85 else \
            1 / (R ** 3 - 4 * R ** 2 + 3 * R)
    I0, I1 = sps.iv(0, k), sps.iv(1, k)
    k -= (I1 / I0 - R) / (1 - (I1 / I0) ** 2 - I1 / (I0 * k))
    return max(k, 0.0), mu


# ---------------------------------------------------------------- fit (1 loi)
def _fit_one(name: str, x: np.ndarray) -> Tuple[str, Optional[float], Optional[Tuple]]:
    n = len(x)
    dist = LAWS[name]
    try:
        if name == "uniform":
            p, k = (0.0, 24.0), 0
            logL, tail = _loglik_trunc(dist, x, p)
        elif name == "truncnorm":
            # Ajustement d'une normale tronquée sur [A,B] = [0,24]
            mu, sigma = st.norm.fit(x)
            a = (A - mu) / sigma
            b = (B - mu) / sigma
            p = (a, b, mu, sigma)
            k = 2  # mu et sigma estimés
            logL, tail = _loglik_trunc(dist, x, p)
        elif name == "vonmises":
            th = x * TWOPI_24
            κ, μ = _dir_est_vonmises(th)
            if κ < .1:  # quasi-uniforme
                return name, None, None
            p = (κ, μ, 1.0)
            k = 2
            logL = κ * np.cos(th - μ).sum() - n * math.log(2 * math.pi * sps.iv(0, κ)) + n * math.log(TWOPI_24)
            tail = 0.0
        elif name == "logistic":
            p = dist.fit(x)  # loc & scale libres
            if not (0.1 <= p[1] <= 10):
                raise ValueError
            k = len(p)
            logL, tail = _loglik_trunc(dist, x, p)
        elif name == "weibull_min":
            p = dist.fit(x, floc=0.0)
            k = len(p)
            logL, tail = _loglik_trunc(dist, x, p)
        else:
            return name, None, None
        if not math.isfinite(logL):
            return name, None, None
        bic = k * math.log(n) - 2 * logL
        ks = st.kstest(x, dist.cdf, args=p).statistic
        score = bic + 0.5 * n * ks + 50 * tail
        return name, score, p
    except Exception:
        return name, None, None


# ------------------------------------------------------------ fit (global)
def fit_distributions_hours(hours: np.ndarray) -> Tuple[str, Tuple]:
    x = np.asarray(hours[np.isfinite(hours)])
    if len(x) < 10:
        raise ValueError("échantillon trop petit (<10)")
    best_name, best_score, best_params = None, math.inf, None
    for name in LAWS:
        _, score, pars = _fit_one(name, x)
        if score is not None and score < best_score:
            best_name, best_score, best_params = name, score, pars
    if best_name is None:
        raise RuntimeError("Aucune loi valide")
    return best_name, best_params


# --------------------------------------------------- bootstrap (mean, sigma)
def _bootstrap_mean_sigma(
        sample: np.ndarray,
        dist: st.rv_continuous,
        params: Tuple,
        law: str,
        n_rep: int = 2000,
        seed: int = 0
) -> dict[str, float]:
    """
    Bootstrap pour estimer IC95 de la moyenne, de la médiane et de l'écart-type
    d'un échantillon simulé à partir d'une loi tronquée ou circulaire.
    - sample : données originales (1D array)
    - dist   : objet scipy.stats (rv_continuous) de la loi ajustée
    - params : paramètres de la loi ajustée
    - law    : nom de la loi ("vonmises", "truncnorm", etc.)
    - n_rep  : nombre de répétitions bootstrap
    - seed   : graine pour la reproductibilité

    Retourne un dict avec les bornes inférieure/supérieure à 95 % pour mean, median, sigma.
    """
    n = len(sample)
    rng = np.random.default_rng(seed)

    def _rvs(size: int) -> np.ndarray:
        # Tirages respectant la tronquature sur [0,24] ou la circularité pour vonmises
        if law == "vonmises":
            # convertit de radians (0–2π) en heures (0–24)
            return dist.rvs(*params, size=size) * 24 / (2 * math.pi)
        out = []
        while len(out) < size:
            d = dist.rvs(*params, size=size)
            # on ne conserve que les valeurs dans [0,24)
            valid = d[(0 <= d) & (d < 24)]
            out.extend(valid.tolist())
        return np.asarray(out[:size])

    means, medians, sigmas = [], [], []
    for _ in range(n_rep):
        s = _rvs(n)
        means.append(s.mean())
        medians.append(np.median(s))
        sigmas.append(s.std(ddof=1))

    pct = np.percentile
    return {
        "mean_lo": pct(means, 2.5),
        "mean_hi": pct(means, 97.5),
        "median_lo": pct(medians, 2.5),
        "median_hi": pct(medians, 97.5),
        "sigma_lo": pct(sigmas, 2.5),
        "sigma_hi": pct(sigmas, 97.5),
    }


# stats_calculator.py (suite après _bootstrap_mean_sigma)

# --------------------------------------------------------------- IC (½-heure)
def compute_confidence_intervals_hours(hours: np.ndarray,
                                       n_boot: int = 2000,
                                       seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(hours)
    if n < 5:
        return {"ic_boot_mean_lo": np.nan, "ic_boot_mean_hi": np.nan}

    means = [rng.choice(hours, size=n, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"ic_boot_mean_lo": lo, "ic_boot_mean_hi": hi}


# ------------------------------------------------------- extrêmes journaliers
def extract_extreme_hours(df: pd.DataFrame):
    """Heures (décimales) des max/min journaliers hors 00:00/24:00."""
    from time_calculator import compute_daily_extrema_timestamps

    ext = compute_daily_extrema_timestamps(df)
    to_hours = lambda ts: ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
    h_max = to_hours(ext['time_max'])
    h_min = to_hours(ext['time_min'])
    return h_max.to_numpy(), h_min.to_numpy()


def compute_extremes_stats(df: pd.DataFrame,
                           n_boot: int = 2000) -> Dict[str, dict]:
    h_max, h_min = extract_extreme_hours(df)
    out: Dict[str, dict] = {}
    for lab, arr in (("max", h_max), ("min", h_min)):
        law, pars = fit_distributions_hours(arr)
        ci = _bootstrap_mean_sigma(arr, LAWS[law], pars, law, n_rep=n_boot)
        out[lab] = {"law": law, "params": np.round(np.asarray(pars), 3), "ci": ci}
    return out


def compute_parametric_ci(law: str, params: Tuple, alpha: float = 0.05) -> dict[str, float]:
    """
    Pour la loi `law` ajustée avec `params`, renvoie l'IC central [alpha/2,1-alpha/2]
    et le mode analytique.
    """
    from scipy.stats import truncnorm, logistic, vonmises, uniform, weibull_min, norm

    dist_map = {
        "truncnorm": truncnorm,
        "logistic": logistic,
        "vonmises": vonmises,
        "uniform": uniform,
        "weibull_min": weibull_min
    }
    dist = dist_map[law]
    lo, hi = dist.ppf([alpha / 2, 1 - alpha / 2], *params)

    # Mode analytique selon la loi
    if law == "uniform":
        loc, scale = params
        mode = loc + scale / 2
    elif law == "truncnorm":
        mode = params[2]
    elif law == "logistic":
        mode = params[0]
    elif law == "weibull_min":
        c, loc, scale = params
        mode = loc + scale * ((c - 1) / c) ** (1 / c) if c > 1 else loc
    elif law == "vonmises":
        κ, μ, _ = params
        mode = (μ * 24 / (2 * math.pi)) % 24
    else:
        mode = float('nan')

    return {"lo": float(lo), "hi": float(hi), "mode": float(mode)}


def compute_hdi(sample: np.ndarray, cred_mass: float = 0.95) -> dict[str, float]:
    """
    Calcule l'intervalle HDI (Highest Density Interval) de masse cred_mass
    et le mode principal (via KDE).
    """
    arr = np.sort(sample[~np.isnan(sample)])
    N = len(arr)
    if N < 3:
        return {"lo": np.nan, "hi": np.nan, "mode": np.nan}

    k = int(np.floor(cred_mass * N))
    widths = arr[k:] - arr[:N - k]
    idx = np.argmin(widths)
    lo, hi = arr[idx], arr[idx + k]

    kde = gaussian_kde(arr)
    grid = np.linspace(lo, hi, 200)
    dens = kde(grid)
    mode = grid[np.argmax(dens)]

    return {"lo": float(lo), "hi": float(hi), "mode": float(mode)}


def get_primary_peak_hours(hours: np.ndarray,
                           grid_steps: int = 1000,
                           window_half_width: float = 1.0) -> np.ndarray:
    """
    Isole le cluster d'heures autour du pic principal de la densité (KDE),
    puis renvoie uniquement les heures situées à ±window_half_width/2 h autour de ce pic.
    """
    arr = np.sort(hours[~np.isnan(hours)])
    if len(arr) < 3:
        return arr

    kde = gaussian_kde(arr)
    grid = np.linspace(0, 24, grid_steps)
    density = kde(grid)

    peaks, _ = find_peaks(density)
    center = grid[peaks[np.argmax(density[peaks])]] if len(peaks) else np.mean(arr)

    diff = np.abs(((arr - center + 12) % 24) - 12)
    mask = diff <= (window_half_width / 2)
    return arr[mask]


def compute_value_extremes_stats(daily_stats: pd.DataFrame) -> dict[str, dict]:
    """
    Ajuste les meilleures lois (par critère BIC) aux valeurs max & min journalières.
    Renvoie {'max': {'law': nom, 'params': tuple}, 'min': {...}}.
    """

    def _fit_best(data: np.ndarray):
        best_name, best_bic, best_params = None, float('inf'), None
        n = len(data)
        for name, dist in {
            'norm': norm,
            'logistic': logistic,
            'weibull_min': weibull_min,
            'truncnorm': truncnorm
        }.items():
            try:
                params = dist.fit(data)
                logL = np.sum(dist.logpdf(data, *params))
                bic = len(params) * np.log(n) - 2 * logL
                if bic < best_bic:
                    best_name, best_bic, best_params = name, bic, params
            except Exception:
                continue
        return best_name, best_params

    arr_max = daily_stats['max'].dropna().values
    arr_min = daily_stats['min'].dropna().values

    law_max, params_max = _fit_best(arr_max)
    law_min, params_min = _fit_best(arr_min)

    return {
        'max': {'law': law_max, 'params': params_max},
        'min': {'law': law_min, 'params': params_min}
    }


def compute_value_clusters(daily_stats: pd.DataFrame, window_half_width: float) -> Dict[str, object]:
    """
    Pour les valeurs max & min journalières,
    - isole le cluster autour du pic principal de densité (KDE),
    - calcule le centre (moyenne du cluster) et la demi-largeur.
    """
    arr_max = daily_stats['max'].dropna().values
    arr_min = daily_stats['min'].dropna().values

    cluster_max = get_primary_peak_hours(arr_max, window_half_width=window_half_width)
    cluster_min = get_primary_peak_hours(arr_min, window_half_width=window_half_width)

    center_max = float(np.mean(cluster_max)) if cluster_max.size else np.nan
    center_min = float(np.mean(cluster_min)) if cluster_min.size else np.nan
    margin = window_half_width / 2

    return {
        'center_max': center_max, 'margin_max': margin, 'cluster_max': cluster_max,
        'center_min': center_min, 'margin_min': margin, 'cluster_min': cluster_min
    }


def compute_value_cluster_stats(daily_stats: pd.DataFrame,
                                window_half_width: float) -> dict[str, dict]:
    """
    Ajuste les meilleures lois aux clusters de valeurs max & min journalières.
    Renvoie :
      {'max': {'law': nom_loi, 'params': tuple}, 'min': {...}}
    """
    clusters = compute_value_clusters(daily_stats, window_half_width)
    arr_max_cl = clusters['cluster_max']
    arr_min_cl = clusters['cluster_min']

    def _fit_best(data: np.ndarray):
        best_name, best_bic, best_params = None, float('inf'), None
        n = len(data)
        for name, dist in {
            'norm': norm,
            'logistic': logistic,
            'weibull_min': weibull_min,
            'truncnorm': truncnorm
        }.items():
            try:
                params = dist.fit(data)
                logL = np.sum(dist.logpdf(data, *params))
                bic = len(params) * np.log(n) - 2 * logL
                if bic < best_bic:
                    best_name, best_bic, best_params = name, bic, params
            except Exception:
                continue
        return best_name, best_params

    law_max_cl, params_max_cl = _fit_best(arr_max_cl)
    law_min_cl, params_min_cl = _fit_best(arr_min_cl)

    return {
        'max': {'law': law_max_cl, 'params': params_max_cl},
        'min': {'law': law_min_cl, 'params': params_min_cl}
    }
