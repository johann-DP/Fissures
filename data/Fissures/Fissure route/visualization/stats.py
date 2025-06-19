import pandas as pd
from scipy.stats import linregress, spearmanr
import pymannkendall as mk
from rich.table import Table
from rich.console import Console

console = Console()


def global_regression_stats(df_date):
    """
    Régression linéaire globale sur 'inch' en fonction du temps (time_numeric).
    Affiche un petit tableau avec slope, intercept, R², p-value, std_err.
    Retourne le dictionnaire {slope, intercept, r_value, p_value, std_err}.
    """
    slope, intercept, r_value, p_value, std_err = linregress(df_date['time_numeric'], df_date['inch'])
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Paramètre")
    table.add_column("Valeur", justify="right")
    table.add_row("Slope", f"{slope:.6f}")
    table.add_row("Intercept", f"{intercept:.6f}")
    table.add_row("R-squared", f"{r_value ** 2:.3f}")
    table.add_row("p-value", f"{p_value:.3f}")
    table.add_row("Std Err", f"{std_err:.6f}")
    console.print(
        "[bold underline]Test de régression linéaire globale sur l'évolution de l'inclinaison[/bold underline]")
    console.print(table)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err
    }


def mann_kendall_test(df_date):
    """
    Test de Mann-Kendall pour la tendance (croissante/décroissante) de 'inch'.
    Affiche le résultat + renvoie l'objet mk_result.
    """
    try:
        result = mk.original_test(df_date['inch'])
        console.print("[bold underline]Test Mann-Kendall sur l'inclinaison[/bold underline]")
        # Limiter l'affichage
        console.print(
            f"trend={result.trend}, h={result.h}, p={result.p:.4f}, z={result.z:.4f}, slope={result.slope:.6f}")
        return result
    except Exception as e:
        console.print(f"[bold red]Erreur lors du test Mann-Kendall:[/bold red] {e}")
        return None


def local_regression_analysis(df_date, window_size=5):
    """
    Régression locale (fenêtre glissante) pour 'inch' en fonction de time_numeric.
    Affiche un DataFrame (start_index, slope, p_value).
    """
    from scipy.stats import linregress
    rolling_results = []
    for i in range(len(df_date) - window_size + 1):
        window = df_date.iloc[i:i + window_size]
        res = linregress(window['time_numeric'], window['inch'])
        rolling_results.append((window.index[0], res.slope, res.pvalue))
    df_rolling = pd.DataFrame(rolling_results, columns=["start_index", "slope", "p_value"])
    console.print("[bold underline]Analyse de régression locale (fenêtre glissante, 5 points)[/bold underline]")
    console.print(df_rolling)

    return df_rolling


def compare_tests(df_date, lr_res, mk_res):
    """
    Compare LR (linregress) et MK (Mann-Kendall), ainsi qu'un Spearman,
    et affiche un tableau style:

        Trend MK |     MK     |   Spearman    |        LR
           Décr.   N. Sign.      Croiss. Sign.   Croiss. Sign.

    - Trend MK : signe du slope Mann-Kendall (croiss/décr/no trend)
    - MK : 'Sign.' ou 'N. Sign.' selon p < 0.05
    - Spearman : slope sign + signification
    - LR : slope sign + signification
    """

    # Mann-Kendall
    if mk_res is not None:
        mk_slope = mk_res.slope
        mk_p = mk_res.p
        if mk_slope > 0:
            mk_trend = "Croiss."
        elif mk_slope < 0:
            mk_trend = "Décr."
        else:
            mk_trend = "No trend"
        mk_sign = "Sign." if mk_p < 0.05 else "N. Sign."
    else:
        mk_trend = "N/A"
        mk_sign = "N/A"

    # LR
    lr_slope = lr_res["slope"]
    lr_p = lr_res["p_value"]
    if lr_slope > 0:
        lr_trend = "Croiss."
    elif lr_slope < 0:
        lr_trend = "Décr."
    else:
        lr_trend = "No trend"
    lr_sign = "Sign." if lr_p < 0.05 else "N. Sign."

    # Spearman
    corr, p_spear = spearmanr(df_date['time_numeric'], df_date['inch'])
    if corr > 0:
        spear_trend = "Croiss."
    elif corr < 0:
        spear_trend = "Décr."
    else:
        spear_trend = "No trend"
    spear_sign = "Sign." if p_spear < 0.05 else "N. Sign."

    # Construire un petit tableau
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Trend MK", justify="center")
    t.add_column("MK", justify="center")
    t.add_column("Spearman", justify="center")
    t.add_column("LR", justify="center")

    # Sur une seule ligne
    row = [
        mk_trend,
        mk_sign,
        f"{spear_trend} {spear_sign}",
        f"{lr_trend} {lr_sign}"
    ]
    t.add_row(*row)

    console.print("\n[bold underline]Comparaison Mann-Kendall / Spearman / LR[/bold underline]")
    console.print(t)
