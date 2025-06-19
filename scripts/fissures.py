import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
from IPython.display import display
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess


def chargement_donnees(chemin):
    """Charge les données depuis un fichier Excel."""

    df = pd.read_excel(chemin)
    df.columns = [
        "Date",
        "Bureau",
        "Mur extérieur",
        "Variation Bureau",
        "Variation Mur",
    ]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days
    return df


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


def dataviz_evolution(df):
    """
    Génère deux graphiques distincts : un pour les mesures normalisées et un pour les variations.
    Inclut les coefficients de corrélation de Pearson et de Spearman dans les légendes.

    Arguments:
    df : DataFrame contenant les données avec les colonnes 'Date', 'Bureau', 'Mur extérieur',
         'Variation Bureau', et 'Variation Mur'.
    """
    # Configuration de la figure principale avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, dpi=300, sharex=True, figsize=(12, 6)
    )

    # Premier axe pour les valeurs normalisées
    color_ax1 = "tab:blue"
    ax1.set_ylabel("Valeur (%)", color=color_ax1)
    ax1.plot(
        df["Date"],
        df["Bureau"] / df["Bureau"].mean() * 100 - 100,
        color=color_ax1,
        label="Bureau",
        alpha=0.5,
    )
    ax1.plot(
        df["Date"],
        df["Mur extérieur"] / df["Mur extérieur"].mean() * 100 - 100,
        color="tab:green",
        label="Mur Extérieur",
        alpha=0.5,
    )
    ax1.tick_params(axis="y", labelcolor=color_ax1)

    # Calcul des coefficients de corrélation pour le premier subplot
    corr_pearson = pearsonr(df["Bureau"], df["Mur extérieur"])
    corr_spearman = spearmanr(df["Bureau"], df["Mur extérieur"])
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), title="", frameon=False)
    ax1.text(
        0.01,
        0.01,
        f"Pearson: {corr_pearson[0]:.2f}\nSpearman: {corr_spearman[0]:.2f}",
        transform=ax1.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    plt.xticks(rotation=45)

    # Second axe pour les variations
    color_ax2 = "tab:red"
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Variation (en mm)", color=color_ax2)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.plot(
        df["Date"],
        df["Variation Bureau"],
        color=color_ax2,
        label="Variations\nBureau",
        alpha=0.75,
    )
    ax2.plot(
        df["Date"],
        df["Variation Mur"],
        color="tab:orange",
        label="Variations\nMur",
        alpha=0.75,
    )
    ax2.tick_params(axis="y", labelcolor=color_ax2)

    # Calcul des coefficients de corrélation pour le second subplot
    corr_pearson2 = pearsonr(
        df["Variation Bureau"].dropna(), df["Variation Mur"].dropna()
    )
    corr_spearman2 = spearmanr(
        df["Variation Bureau"].dropna(), df["Variation Mur"].dropna()
    )
    ax2.text(
        0.01,
        0.99,
        f"Pearson: {corr_pearson2[0]:.2f}\nSpearman: {corr_spearman2[0]:.2f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1), title="", frameon=False)

    # Ajouter un titre global pour les graphiques
    plt.suptitle("Evolution des écarts et de leurs variations", fontsize=16)

    # Ajustement de l'espacement pour éviter les chevauchements
    plt.tight_layout()  # rect=[0, 0.03, 1, 0.95]

    # Afficher le graphique
    plt.show()

    return fig


def loess_regression(df):
    """
    Effectue une régression LOESS sur les données et affiche un graphique avec la RMSE.

    Arguments:
    df : DataFrame contenant les données avec les colonnes 'Days' et 'Bureau\n(mm)'.

    Retourne:
    fig : La figure matplotlib de la régression LOESS.
    second_phase_data : DataFrame contenant les données de la deuxième phase.
    """
    # Diviser les données en deux parties
    threshold_day = 55
    first_phase_data = df[df["Days"] <= threshold_day]
    second_phase_data = df[df["Days"] > threshold_day]

    # Définir les valeurs à tester pour les paramètres it et delta
    it_values = np.linspace(3, 50, 5, dtype=int)
    delta_values = np.linspace(0, 100, 5)

    best_rmse = float("inf")
    best_params = {}

    # Effectuer la recherche par grille
    for it, delta in itertools.product(it_values, delta_values):
        # Régression LOESS pour la première phase
        loess_smoothed_first = lowess(
            first_phase_data["Bureau"], first_phase_data["Days"], it=it, delta=delta
        )

        # Régression LOESS pour la deuxième phase
        loess_smoothed_second = lowess(
            second_phase_data["Bureau"], second_phase_data["Days"], it=it, delta=delta
        )

        # Concaténer les prédictions des deux phases
        y_pred = np.concatenate(
            [loess_smoothed_first[:, 1], loess_smoothed_second[:, 1]]
        )

        # Calculer la RMSE
        rmse = np.sqrt(mean_squared_error(df["Bureau"], y_pred))

        # Vérifier si cette combinaison de paramètres donne une meilleure RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"it": it, "delta": delta}

    # Régression LOESS pour la première phase avec les meilleurs paramètres
    loess_smoothed_first = lowess(
        first_phase_data["Bureau"],
        first_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    # Régression LOESS pour la deuxième phase avec les meilleurs paramètres
    loess_smoothed_second = lowess(
        second_phase_data["Bureau"],
        second_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Première phase
    ax.scatter(
        first_phase_data["Days"],
        first_phase_data["Bureau"],
        label="Première phase",
        color="blue",
    )
    ax.plot(first_phase_data["Days"], loess_smoothed_first[:, 1], color="red")

    # Deuxième phase
    ax.scatter(
        second_phase_data["Days"],
        second_phase_data["Bureau"],
        label="Deuxième phase",
        color="green",
    )
    ax.plot(second_phase_data["Days"], loess_smoothed_second[:, 1], color="orange")
    second_phase_data["LOESS Bureau"] = loess_smoothed_second[:, 1]

    # Ajout de la RMSE en légende
    ax.legend(title=f"RMSE Ph.2 : {best_rmse:.2f} mm")
    ax.set_title(
        "Régression LOESS pour la série chronologique Bureau pour chaque phase"
    )
    ax.set_xlabel("Jours depuis le début de la série")
    ax.set_ylabel("Bureau (mm)")
    ax.grid(True)

    plt.show()

    return fig, second_phase_data


def linear_regression(df):
    """
    Effectue et trace la régression linéaire cumulative et prévisionnelle pour les données fournies.

    Arguments:
    df : DataFrame contenant les données avec les colonnes 'Days' et 'Bureau\n(mm)'.

    Retourne:
    fig1, fig2 : Les deux figures matplotlib de régression linéaire.
    regression_results : DataFrame contenant les résultats de la régression pour chaque sous-ensemble.
    """
    n = len(df)  # Nombre total de points de données
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))  # Utilise le colormap 'coolwarm'
    regression_results = []

    # Création de la première figure pour l'évolution de la régression linéaire
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    for i in range(2, n + 1):
        subset = df.iloc[:i]
        slope, intercept, r_value, p_value, std_err = linregress(
            subset["Days"], subset["Bureau"]
        )
        x = np.array([min(subset["Days"]), max(subset["Days"])])
        y = slope * x + intercept
        alpha = i / n
        ax1.plot(x, y, color=colors[i - 2], alpha=alpha)
        regression_results.append([i, slope, intercept, r_value, p_value, std_err])

    ax1.scatter(df["Days"], df["Bureau"], color="blue")
    ax1.set_xlabel("Jours depuis le début de la série")
    ax1.set_ylabel("Bureau (mm)")
    ax1.set_title("Évolution de la régression linéaire")

    # Création de la deuxième figure pour la régression linéaire avec prévision
    fig2, ax2 = plt.subplots(dpi=300)
    sns.regplot(x="Days", y="Bureau", data=df, ax=ax2)
    x = np.array([0, 365])
    y_model = np.polyval(np.polyfit(df["Days"], df["Bureau"], 1), x)
    ax2.plot(x, y_model, "--", color="grey")
    ax2.scatter(50, y_model[1], color="red")
    ax2.text(
        50,
        y_model[1],
        f"  Prévision à 365 jours : {y_model[1]:.2f} mm ",
        color="red",
        ha="left",
    )
    delta_y = y_model[1] - df["Bureau"].iloc[0]
    ax2.text(
        50,
        y_model[1] - 0.05,
        f"           écart : {delta_y:.2f} mm",
        color="red",
        ha="left",
    )
    ax2.set_xlabel("Jours depuis le début de la série")
    ax2.set_ylabel("Bureau (mm)")
    ax2.set_title("Régression linéaire avec prévision pour 365 jours")
    ax2.set_xlim(0, max(df["Days"]) + 10)
    ax2.set_ylim(
        min(df["Bureau"]) - 0.02, max(max(df["Bureau"]) + 0.02, y_model[1] + 0.05)
    )
    plt.show()

    regression_results_df = pd.DataFrame(
        regression_results,
        columns=["Week", "Slope", "Intercept", "R_value", "P_value", "Std_err"],
    )
    return fig1, fig2, regression_results_df


def regression_comparison(df, loess_data, regression_results_df):
    """
    Compare les régressions LOESS et linéaire avec visualisation des prévisions à 365 jours.

    Arguments:
    df : DataFrame original contenant 'Days' et 'Bureau'.
    loess_data : DataFrame contenant les données de la deuxième phase et la régression LOESS.
    regression_results_df : DataFrame contenant les résultats de la régression linéaire sur tout le dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Couleurs pour différentes phases
    phase_color_map = {"Phase 1": "blue", "Phase 2": "green"}

    # Tracer les données brutes avec couleurs pour chaque phase
    threshold_day = 55
    ax.scatter(
        df[df["Days"] <= threshold_day]["Days"],
        df[df["Days"] <= threshold_day]["Bureau"],
        color=phase_color_map["Phase 1"],
        label="Phase 1",
    )
    ax.scatter(
        df[df["Days"] > threshold_day]["Days"],
        df[df["Days"] > threshold_day]["Bureau"],
        color=phase_color_map["Phase 2"],
        label="Phase 2",
    )

    # Régression LOESS pour la deuxième phase
    if "LOESS Bureau" in loess_data.columns:
        ax.plot(
            loess_data["Days"],
            loess_data["LOESS Bureau"],
            color="orange",
            label="LOESS Phase 2",
        )
    else:
        print(
            "Column 'LOESS Bureau' does not exist in loess_data DataFrame. Check column names."
        )

    # Régression linéaire linéaire globale et prévision à 365 jours
    slope, intercept, r_value, p_value, std_err = linregress(df["Days"], df["Bureau"])
    x_global = np.array([0, df["Days"].iloc[-1]])
    y_global = slope * x_global + intercept
    ax.plot(x_global, y_global, "--", color="red", label="Régression Linéaire Globale")

    # Prévision à 365 jours pour la régression globale
    y_pred_365 = slope * 365 + intercept
    delta_y = y_pred_365 - df["Bureau"].iloc[0]
    ax.scatter(50, y_pred_365, color="red")
    ax.text(
        55,
        y_pred_365,
        f"Prévision à 365 jours : {y_pred_365:.2f} mm ; écart : {delta_y:.2f} mm",
        color="red",
        ha="left",
    )

    # Prévision à 365 jours pour la régression linéaire sur la phase 2
    slope, intercept, r_value, p_value, std_err = linregress(
        df[df["Days"] > threshold_day]["Days"], df[df["Days"] > threshold_day]["Bureau"]
    )
    y_pred_365 = slope * 365 + intercept
    delta_y = y_pred_365 - df["Bureau"].iloc[0]
    ax.scatter(50, y_pred_365, color="purple")
    x_loess = np.array([threshold_day, df["Days"].iloc[-1]])
    y_loess = slope * x_loess + intercept
    ax.plot(x_loess, y_loess, "--", color="purple", label="Régression Linéaire Phase 2")
    ax.text(
        55,
        y_pred_365,
        f"Prévision à 365 jours : {y_pred_365:.2f} mm ; écart : {delta_y:.2f} mm",
        color="purple",
        ha="left",
    )

    # Configuration du graphique
    ax.set_xlabel("Jours depuis le début de la série")
    ax.set_ylabel("Bureau (mm)")
    ax.set_title("Comparaison des régressions et prévisions")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    ax.grid(True)

    plt.show()

    return fig


def main():
    chemin = "C:/Users/johan/Bureau/Fissure_2.xlsx"
    df = chargement_donnees(chemin)

    print("Bureau\n")
    print("Tests globaux\n")
    tests_statistiques(df)

    _ = dataviz_evolution(df)
    _, second_phase_data = loess_regression(df)
    fig1, fig2, regression_results_df = linear_regression(df)
    _ = regression_comparison(df, second_phase_data, regression_results_df)

    print("\nTests phase 2\n")
    tests_statistiques(df.iloc[9:, :])


if __name__ == "__main__":
    main()
