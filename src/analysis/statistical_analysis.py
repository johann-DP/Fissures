import numpy as np
import pandas as pd
import pymannkendall as mk
from IPython.display import display
from scipy.stats import linregress, spearmanr


def tests_statistiques(df):
    """Applique les tests statistiques et affiche les résultats."""
    results = []
    for i in range(2, len(df) + 1):
        subset = df.iloc[:i]
        mk_result = mk.original_test(subset["Bureau"])
        spearman_corr, spearman_p = spearmanr(subset["Days"], subset["Bureau"])
        slope, intercept, r_value, p_value, std_err = linregress(
            subset["Days"], subset["Bureau"]
        )

        # TODO valider le calcul pour LR
        results.append(
            {
                "p-value MK": round(mk_result.p, 2),
                "Corr. Sp.": round(spearman_corr, 2),
                "p-value Sp.": round(spearman_p, 2),
                "Slope LR": round(slope, 3),
                "Corr. LR": round(r_value, 2),
                "p-value LR": round(p_value, 2),
                "Trend MK": "Croiss." if mk_result.trend == "increasing" else "Décr.",
                "MK": "Sign.  " if mk_result.p < 0.05 else "N. Sign.  ",
                "Spearman": "Croiss. Sign.  " if spearman_p < 0.05 else "N. Sign.  ",
                "LR": (
                    "Croiss. Sign."
                    if p_value < 0.05 and slope > 0
                    else "Décr. Sign." if p_value < 0.05 and slope < 0 else "N. Sign."
                ),
            }
        )
    results_df = pd.DataFrame(results)
    results_df.index = np.arange(1, len(results) + 1)
    display(results_df.iloc[:, :6])
    print("")
    display(results_df.iloc[:, 6:])
    return results_df
