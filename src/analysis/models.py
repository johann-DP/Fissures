import itertools
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import (het_breuschpagan,
                                          linear_harvey_collier)

# def loess_regression(df):
#     """
#     Effectue une régression LOESS sur les données et affiche un graphique avec la RMSE.
#
#     Arguments:
#     df : DataFrame contenant les données avec les colonnes 'Days' et 'Bureau\n(mm)'.
#
#     Retourne:
#     fig : La figure plotly de la régression LOESS.
#     second_phase_data : DataFrame contenant les données de la deuxième phase.
#     """
#     # Diviser les données en trois parties
#     threshold_day_1 = 55
#     threshold_day_2 = 210
#     first_phase_data = df[df["Days"] <= threshold_day_1]
#     second_phase_data = df[
#         (df["Days"] > threshold_day_1) & (df["Days"] <= threshold_day_2)
#         ]
#     third_phase_data = df[df["Days"] > threshold_day_2]
#
#     # Définir les valeurs à tester pour les paramètres it et delta
#     it_values = np.linspace(3, 50, 5, dtype=int)
#     delta_values = np.linspace(0, 100, 5)
#
#     best_rmse = float("inf")
#     best_params = {}
#
#     # Effectuer la recherche par grille
#     for it, delta in itertools.product(it_values, delta_values):
#         # Régression LOESS pour la première phase
#         loess_smoothed_first = lowess(
#             first_phase_data["Bureau"], first_phase_data["Days"], it=it, delta=delta
#         )
#
#         # Régression LOESS pour la deuxième phase
#         loess_smoothed_second = lowess(
#             second_phase_data["Bureau"], second_phase_data["Days"], it=it, delta=delta
#         )
#
#         # Régression LOESS pour la troisième phase
#         loess_smoothed_third = lowess(
#             third_phase_data["Bureau"], third_phase_data["Days"], it=it, delta=delta
#         )
#
#         # Concaténer les prédictions des trois phases
#         y_pred = np.concatenate(
#             [
#                 loess_smoothed_first[:, 1],
#                 loess_smoothed_second[:, 1],
#                 loess_smoothed_third[:, 1],
#             ]
#         )
#
#         # Calculer la RMSE
#         rmse = np.sqrt(mean_squared_error(df["Bureau"], y_pred))
#
#         # Vérifier si cette combinaison de paramètres donne une meilleure RMSE
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_params = {"it": it, "delta": delta}
#
#     # Régression LOESS pour la première phase avec les meilleurs paramètres
#     loess_smoothed_first = lowess(
#         first_phase_data["Bureau"],
#         first_phase_data["Days"],
#         it=best_params["it"],
#         delta=best_params["delta"],
#     )
#
#     # Régression LOESS pour la deuxième phase avec les meilleurs paramètres
#     loess_smoothed_second = lowess(
#         second_phase_data["Bureau"],
#         second_phase_data["Days"],
#         it=best_params["it"],
#         delta=best_params["delta"],
#     )
#
#     # Régression LOESS pour la troisième phase avec les meilleurs paramètres
#     loess_smoothed_third = lowess(
#         third_phase_data["Bureau"],
#         third_phase_data["Days"],
#         it=best_params["it"],
#         delta=best_params["delta"],
#     )
#
#     # Création du graphique avec Plotly
#     fig = go.Figure()
#
#     # Première phase
#     fig.add_trace(
#         go.Scatter(
#             x=first_phase_data["Days"],
#             y=first_phase_data["Bureau"],
#             mode="markers",
#             name="Première phase",
#             marker=dict(color="blue"),
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=first_phase_data["Days"],
#             y=loess_smoothed_first[:, 1],
#             mode="lines",
#             name="LOESS Première phase",
#             line=dict(color="red"),
#         )
#     )
#
#     # Deuxième phase
#     fig.add_trace(
#         go.Scatter(
#             x=second_phase_data["Days"],
#             y=second_phase_data["Bureau"],
#             mode="markers",
#             name="Deuxième phase",
#             marker=dict(color="green"),
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=second_phase_data["Days"],
#             y=loess_smoothed_second[:, 1],
#             mode="lines",
#             name="LOESS Deuxième phase",
#             line=dict(color="orange"),
#         )
#     )
#     second_phase_data["LOESS Bureau"] = loess_smoothed_second[:, 1]
#
#     # Troisième phase
#     fig.add_trace(
#         go.Scatter(
#             x=third_phase_data["Days"],
#             y=third_phase_data["Bureau"],
#             mode="markers",
#             name="Troisième phase",
#             marker=dict(color="purple"),
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=third_phase_data["Days"],
#             y=loess_smoothed_third[:, 1],
#             mode="lines",
#             name="LOESS Troisième phase",
#             line=dict(color="brown"),
#         )
#     )
#     third_phase_data["LOESS Bureau"] = loess_smoothed_third[:, 1]
#
#     # Mise à jour de la mise en page du graphique
#     fig.update_layout(
#         title="Régression LOESS pour la série chronologique Bureau pour chaque phase",
#         font=dict(size=20),
#         xaxis_title="Jours depuis le début de la série",
#         yaxis_title="Bureau (mm)",
#         legend_title_text=f"RMSE : {best_rmse:.2f} mm",
#         autosize=True,
#         height=None,
#         width=None,
#     )
#
#     return second_phase_data, third_phase_data, fig


def loess_regression(df):
    """
    Effectue une régression LOESS sur les données et affiche un graphique avec la RMSE.

    Arguments:
    df : DataFrame contenant les données avec les colonnes 'Days' et 'Bureau\n(mm)'.

    Retourne:
    fig : La figure plotly de la régression LOESS.
    second_phase_data : DataFrame contenant les données de la deuxième phase.
    third_phase_data : DataFrame contenant les données de la troisième phase.
    fourth_phase_data : DataFrame contenant les données de la quatrième phase.
    fifth_phase_data : DataFrame contenant les données de la cinquième phase.
    sixth_phase_data : DataFrame contenant les données de la sixième phase.
    """
    # Diviser les données en phases
    threshold_day_1 = 0
    threshold_day_2 = 55
    threshold_day_3 = 210
    threshold_day_4 = 259
    threshold_day_5 = 287

    first_phase_data = df[df["Days"] <= threshold_day_1]
    second_phase_data = df[
        (df["Days"] > threshold_day_1) & (df["Days"] <= threshold_day_2)
    ]
    third_phase_data = df[
        (df["Days"] > threshold_day_2) & (df["Days"] <= threshold_day_3)
    ]
    fourth_phase_data = df[
        (df["Days"] > threshold_day_3) & (df["Days"] <= threshold_day_4)
    ]
    fifth_phase_data = df[
        (df["Days"] > threshold_day_4) & (df["Days"] <= threshold_day_5)
    ]
    sixth_phase_data = df[df["Days"] > threshold_day_5]  # Nouvelle phase ajoutée

    # Définir les valeurs à tester pour les paramètres it et delta
    it_values = np.linspace(3, 50, 5, dtype=int)
    delta_values = np.linspace(0, 100, 5)

    best_rmse = float("inf")
    best_params = {}

    # Effectuer la recherche par grille
    for it, delta in itertools.product(it_values, delta_values):
        # Régression LOESS pour chaque phase
        loess_smoothed_first = lowess(
            first_phase_data["Bureau"], first_phase_data["Days"], it=it, delta=delta
        )

        loess_smoothed_second = lowess(
            second_phase_data["Bureau"], second_phase_data["Days"], it=it, delta=delta
        )

        loess_smoothed_third = lowess(
            third_phase_data["Bureau"], third_phase_data["Days"], it=it, delta=delta
        )

        loess_smoothed_fourth = lowess(
            fourth_phase_data["Bureau"], fourth_phase_data["Days"], it=it, delta=delta
        )

        loess_smoothed_fifth = lowess(
            fifth_phase_data["Bureau"], fifth_phase_data["Days"], it=it, delta=delta
        )

        loess_smoothed_sixth = lowess(
            sixth_phase_data["Bureau"], sixth_phase_data["Days"], it=it, delta=delta
        )

        # Concaténer les prédictions des six phases
        y_pred = np.concatenate(
            [
                loess_smoothed_first[:, 1],
                loess_smoothed_second[:, 1],
                loess_smoothed_third[:, 1],
                loess_smoothed_fourth[:, 1],
                loess_smoothed_fifth[:, 1],
                loess_smoothed_sixth[:, 1],
            ]
        )

        # Calculer la RMSE
        rmse = np.sqrt(mean_squared_error(df["Bureau"], y_pred))

        # Vérifier si cette combinaison de paramètres donne une meilleure RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"it": it, "delta": delta}

    # Régression LOESS finale avec les meilleurs paramètres
    loess_smoothed_first = lowess(
        first_phase_data["Bureau"],
        first_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    loess_smoothed_second = lowess(
        second_phase_data["Bureau"],
        second_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    loess_smoothed_third = lowess(
        third_phase_data["Bureau"],
        third_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    loess_smoothed_fourth = lowess(
        fourth_phase_data["Bureau"],
        fourth_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    loess_smoothed_fifth = lowess(
        fifth_phase_data["Bureau"],
        fifth_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    loess_smoothed_sixth = lowess(
        sixth_phase_data["Bureau"],
        sixth_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Première phase
    fig.add_trace(
        go.Scatter(
            x=first_phase_data["Days"],
            y=first_phase_data["Bureau"],
            mode="markers",
            name="Première phase",
            marker=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=first_phase_data["Days"],
            y=loess_smoothed_first[:, 1],
            mode="lines",
            name="LOESS Première phase",
            line=dict(color="red"),
        )
    )

    # Deuxième phase
    fig.add_trace(
        go.Scatter(
            x=second_phase_data["Days"],
            y=second_phase_data["Bureau"],
            mode="markers",
            name="Deuxième phase",
            marker=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=second_phase_data["Days"],
            y=loess_smoothed_second[:, 1],
            mode="lines",
            name="LOESS Deuxième phase",
            line=dict(color="orange"),
        )
    )
    second_phase_data["LOESS Bureau"] = loess_smoothed_second[:, 1]

    # Troisième phase
    fig.add_trace(
        go.Scatter(
            x=third_phase_data["Days"],
            y=third_phase_data["Bureau"],
            mode="markers",
            name="Troisième phase",
            marker=dict(color="purple"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=third_phase_data["Days"],
            y=loess_smoothed_third[:, 1],
            mode="lines",
            name="LOESS Troisième phase",
            line=dict(color="brown"),
        )
    )
    third_phase_data["LOESS Bureau"] = loess_smoothed_third[:, 1]

    # Quatrième phase
    fig.add_trace(
        go.Scatter(
            x=fourth_phase_data["Days"],
            y=fourth_phase_data["Bureau"],
            mode="markers",
            name="Quatrième phase",
            marker=dict(color="cyan"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fourth_phase_data["Days"],
            y=loess_smoothed_fourth[:, 1],
            mode="lines",
            name="LOESS Quatrième phase",
            line=dict(color="magenta"),
        )
    )
    fourth_phase_data["LOESS Bureau"] = loess_smoothed_fourth[:, 1]

    # Cinquième phase
    fig.add_trace(
        go.Scatter(
            x=fifth_phase_data["Days"],
            y=fifth_phase_data["Bureau"],
            mode="markers",
            name="Cinquième phase",
            marker=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fifth_phase_data["Days"],
            y=loess_smoothed_fifth[:, 1],
            mode="lines",
            name="LOESS Cinquième phase",
            line=dict(color="gray"),
        )
    )
    fifth_phase_data["LOESS Bureau"] = loess_smoothed_fifth[:, 1]

    # Sixième phase
    fig.add_trace(
        go.Scatter(
            x=sixth_phase_data["Days"],
            y=sixth_phase_data["Bureau"],
            mode="markers",
            name="Sixième phase",
            marker=dict(color="yellow"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sixth_phase_data["Days"],
            y=loess_smoothed_sixth[:, 1],
            mode="lines",
            name="LOESS Sixième phase",
            line=dict(color="blue"),
        )
    )
    sixth_phase_data["LOESS Bureau"] = loess_smoothed_sixth[:, 1]

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        title="Régression LOESS pour la série chronologique Bureau pour chaque phase",
        font=dict(size=20),
        xaxis_title="Jours depuis le début de la série",
        yaxis_title="Bureau (mm)",
        legend_title_text=f"RMSE : {best_rmse:.2f} mm",
        autosize=True,
        height=None,
        width=None,
    )

    return (
        second_phase_data,
        third_phase_data,
        fourth_phase_data,
        fifth_phase_data,
        sixth_phase_data,
        fig,
    )


def linear_regression(df):
    """
    Effectue et trace la régression linéaire cumulative et prévisionnelle pour les données fournies.

    Arguments:
    df : DataFrame contenant les données avec les colonnes 'Days' et 'Bureau'.

    Retourne:
    fig : La figure plotly contenant les deux sous-graphiques de régression linéaire.
    regression_results : DataFrame contenant les résultats de la régression pour chaque sous-ensemble.
    """
    n = len(df)
    colors = px.colors.sequential.Blues
    regression_results = []

    # Créez une figure avec deux sous-graphiques (subplots)
    fig = make_subplots(rows=2, cols=1)

    # Premier subplot (fig1)
    for i in range(2, n + 1):
        subset = df.iloc[:i]
        slope, intercept, r_value, p_value, std_err = linregress(
            subset["Days"], subset["Bureau"]
        )
        x = np.array([min(subset["Days"]), max(subset["Days"])])
        y = slope * x + intercept
        alpha = i / n
        color_index = int((i - 2) / (n - 2) * (len(colors) - 1))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=colors[color_index], width=1),
                opacity=0.8,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        regression_results.append([i, slope, intercept, r_value, p_value, std_err])

    fig.add_trace(
        go.Scatter(
            x=df["Days"],
            y=df["Bureau"],
            mode="markers",
            marker=dict(color="blue", opacity=0.5),
            name="Bureau",
        ),
        row=1,
        col=1,
    )

    # Deuxième subplot (fig2)
    x = np.array([0, 365])
    y_model = np.polyval(np.polyfit(df["Days"], df["Bureau"], 1), x)
    fig.add_trace(
        go.Scatter(
            x=df["Days"],
            y=df["Bureau"],
            mode="markers",
            marker=dict(color="blue", opacity=0.5),
            name="Bureau",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_model,
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Modèle",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[365],
            y=[y_model[1]],
            mode="markers",
            marker=dict(color="red", opacity=0.6),
            name="Prévision à 365 jours",
        ),
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=365,
        y=y_model[1],
        text=f"Prévision à 365 jours : {y_model[1]:.2f} mm",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-50,
        font=dict(size=12, color="red"),
        row=2,
        col=1,
    )
    delta_y = y_model[1] - df["Bureau"].iloc[0]
    fig.add_annotation(
        x=365,
        y=y_model[1] - 0.025,
        text=f"Écart : {delta_y:.2f} mm",
        showarrow=False,
        font=dict(size=12, color="red"),
        row=2,
        col=1,
    )

    # Mise à jour de la disposition
    fig.update_layout(
        title="Régression linéaire : évolution et prévision",
        font=dict(size=20),
        xaxis_title="Jours depuis le début de la série",
        yaxis_title="Bureau (mm)",
        autosize=True,
        height=None,
        width=None,
    )

    regression_results_df = pd.DataFrame(
        regression_results,
        columns=["Week", "Slope", "Intercept", "R_value", "P_value", "Std_err"],
    )

    return regression_results_df, fig


def regression_comparison(df):
    """
    Compare les régressions LOESS et linéaire avec visualisation des prévisions à 365 jours, placées à l'abscisse du jour 25.

    Arguments:
    df : DataFrame original contenant 'Days' et 'Bureau'.
    loess_data : DataFrame contenant les données des phases 1, 2, et 3 avec les régressions LOESS.
    regression_results_df : DataFrame contenant les résultats de la régression linéaire sur tout le dataset.
    """
    fig = go.Figure()

    # Définition des seuils pour les phases
    threshold_day_1 = 55
    threshold_day_2 = 210
    prediction_day = 365
    annotation_day = 25

    # Diviser les données en trois phases
    first_phase_data = df[df["Days"] <= threshold_day_1]
    second_phase_data = df[
        (df["Days"] > threshold_day_1) & (df["Days"] <= threshold_day_2)
    ]
    third_phase_data = df[df["Days"] > threshold_day_2]

    # Tracer les données brutes pour chaque phase
    fig.add_trace(
        go.Scatter(
            x=first_phase_data["Days"],
            y=first_phase_data["Bureau"],
            mode="markers",
            name="Phase 1",
            marker=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=second_phase_data["Days"],
            y=second_phase_data["Bureau"],
            mode="markers",
            name="Phase 2",
            marker=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=third_phase_data["Days"],
            y=third_phase_data["Bureau"],
            mode="markers",
            name="Phase 3",
            marker=dict(color="purple"),
        )
    )

    # Recalculer les régressions LOESS avec les meilleurs paramètres trouvés dans loess_regression
    it_values = np.linspace(3, 50, 5, dtype=int)
    delta_values = np.linspace(0, 100, 5)

    best_rmse = float("inf")
    best_params = {}

    # Recherche des meilleurs paramètres
    for it, delta in itertools.product(it_values, delta_values):
        loess_smoothed_first = lowess(
            first_phase_data["Bureau"], first_phase_data["Days"], it=it, delta=delta
        )
        loess_smoothed_second = lowess(
            second_phase_data["Bureau"], second_phase_data["Days"], it=it, delta=delta
        )
        loess_smoothed_third = lowess(
            third_phase_data["Bureau"], third_phase_data["Days"], it=it, delta=delta
        )

        y_pred = np.concatenate(
            [
                loess_smoothed_first[:, 1],
                loess_smoothed_second[:, 1],
                loess_smoothed_third[:, 1],
            ]
        )
        rmse = np.sqrt(mean_squared_error(df["Bureau"], y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"it": it, "delta": delta}

    # Appliquer les meilleurs paramètres trouvés
    loess_smoothed_first = lowess(
        first_phase_data["Bureau"],
        first_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )
    loess_smoothed_second = lowess(
        second_phase_data["Bureau"],
        second_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )
    loess_smoothed_third = lowess(
        third_phase_data["Bureau"],
        third_phase_data["Days"],
        it=best_params["it"],
        delta=best_params["delta"],
    )

    # Afficher les régressions LOESS pour chaque phase en arrière-plan avec transparence
    fig.add_trace(
        go.Scatter(
            x=first_phase_data["Days"],
            y=loess_smoothed_first[:, 1],
            mode="lines",
            name="LOESS Phase 1",
            line=dict(color="red"),
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=second_phase_data["Days"],
            y=loess_smoothed_second[:, 1],
            mode="lines",
            name="LOESS Phase 2",
            line=dict(color="orange"),
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=third_phase_data["Days"],
            y=loess_smoothed_third[:, 1],
            mode="lines",
            name="LOESS Phase 3",
            line=dict(color="brown"),
            opacity=0.5,
        )
    )

    # Régression linéaire globale et prévision à 365 jours
    slope, intercept, r_value, p_value, std_err = linregress(df["Days"], df["Bureau"])
    x_global = np.array([df["Days"].min(), df["Days"].max()])
    y_global = slope * x_global + intercept
    fig.add_trace(
        go.Scatter(
            x=x_global,
            y=y_global,
            mode="lines",
            name="Régression Linéaire Globale",
            line=dict(dash="dash", color="red"),
        )
    )

    # Prévision globale à l'abscisse spécifiée (jour 25)
    y_pred_global = slope * prediction_day + intercept
    fig.add_trace(
        go.Scatter(
            x=[annotation_day],
            y=[y_pred_global],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Prévision à l'abscisse spécifiée (Globale)",
        )
    )
    fig.add_annotation(
        x=annotation_day,
        y=y_pred_global,
        text=f"Prévision globale à 365 jours : {y_pred_global:.2f} mm",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-50,
        font=dict(color="red", size=10),
    )

    # Régressions linéaires et prévisions pour les phases 2 et 3
    for phase_data, color, phase_name in zip(
        [second_phase_data, third_phase_data],
        ["green", "purple"],
        ["Phase 2", "Phase 3"],
    ):
        slope, intercept, r_value, p_value, std_err = linregress(
            phase_data["Days"], phase_data["Bureau"]
        )
        x_phase = np.array([phase_data["Days"].min(), phase_data["Days"].max()])
        y_phase = slope * x_phase + intercept

        fig.add_trace(
            go.Scatter(
                x=x_phase,
                y=y_phase,
                mode="lines",
                name=f"Régression Linéaire {phase_name}",
                line=dict(dash="dash", color=color),
            )
        )

        # Prévision pour chaque phase à l'abscisse du jour 25
        y_pred = slope * prediction_day + intercept
        fig.add_trace(
            go.Scatter(
                x=[
                    (
                        annotation_day
                        if phase_name is None
                        else (
                            annotation_day * 5
                            if phase_name == "Phase 2"
                            else annotation_day * 10
                        )
                    )
                ],
                y=[y_pred],
                mode="markers",
                marker=dict(size=10, color=color),
                name=f"Prévision à l'abscisse spécifiée ({phase_name})",
            )
        )
        fig.add_annotation(
            x=(
                annotation_day
                if phase_name is None
                else (
                    annotation_day * 5
                    if phase_name == "Phase 2"
                    else annotation_day * 10
                )
            ),
            y=y_pred,
            text=f"Prévision pour la {phase_name} à 365 jours : {y_pred:.2f} mm",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-50 if phase_name == "Phase 2" else 50,
            font=dict(color=color, size=10),
        )

    # Configuration du graphique
    fig.update_layout(
        title="Comparaison des régressions et prévisions",
        font=dict(size=20),
        xaxis_title="Jours depuis le début de la série",
        yaxis_title="Bureau (mm)",
        autosize=True,
        height=None,
        width=None,
        legend=dict(orientation="h", x=0, y=-0.2),
    )

    return fig


def select_weekly_variables(df_cleaned):
    """
    Selects and returns the weekly variables from the cleaned dataframe.

    Parameters:
    df_cleaned (DataFrame): The cleaned dataframe with both daily and weekly data_route.

    Returns:
    DataFrame: A dataframe containing only the weekly variables.
    """
    logging.info("Select weekly feat")

    # Assurer que la colonne 'Time' est au format datetime
    df_cleaned["Time"] = pd.to_datetime(df_cleaned["Time"])

    # Filtrer les colonnes hebdomadaires
    weekly_columns = [
        col
        for col in df_cleaned.columns
        if any(
            stat in col
            for stat in ["mean", "median", "std", "skew", "calculate_kurtosis"]
        )
    ]

    # Garder uniquement les lignes pour chaque dimanche à midi
    df_weekly = df_cleaned[df_cleaned["Time"].dt.weekday == 6]
    df_weekly = df_weekly[df_weekly["Time"].dt.hour == 12]

    # Sélectionner les colonnes hebdomadaires
    df_weekly = df_weekly[["Time"] + weekly_columns]

    return df_weekly


def create_features(df):
    logging.info("Create new feat")
    df["lag_1"] = df["Variation Bureau"].shift(1)
    df["lag_2"] = df["Variation Bureau"].shift(2)
    df["lag_3"] = df["Variation Bureau"].shift(3)
    df["lag_4"] = df["Variation Bureau"].shift(4)
    df["lag_5"] = df["Variation Bureau"].shift(5)
    df["lag_6"] = df["Variation Bureau"].shift(6)
    df["lag_7"] = df["Variation Bureau"].shift(7)
    df["lag_7"] = df["Variation Bureau"].shift(8)  # 2 mois de lags
    df["Indoor_Tem_mean_x_Outdoor_Tem_mean"] = (
        df["Indoor Tem(°C)_mean"] * df["Outdoor Tem(°C)_mean"]
    )
    df["Indoor_Hum_mean_x_Outdoor_Hum_mean"] = (
        df["Indoor Hum(%)_mean"] * df["Outdoor Hum(%)_mean"]
    )
    df = df.dropna()
    return df


def prepare_data(df_weekly, df_fissures):
    logging.info("Data prep: join")
    df_fissures["Date"] = pd.to_datetime(df_fissures["Date"]).dt.normalize()
    df_weekly["Time"] = (
        pd.to_datetime(df_weekly["Time"]).dt.tz_convert(None).dt.normalize()
    )
    df_weekly = df_weekly.rename(columns={"Time": "Date"})

    df_joined = pd.merge(df_fissures, df_weekly, how="inner", on="Date")
    df_joined = create_features(df_joined)

    return df_joined


def regression_model(X_train, y_train, model_type="Ridge"):
    """Create and train a Ridge or Lasso regression model"""
    logging.info(f"{model_type} regression")

    # Standardize the data_route
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    logging.info("X: scaled")

    # Create a regression model with cross-validation
    if model_type == "Ridge":
        model = RidgeCV(
            alphas=np.logspace(-6, 6, 13), scoring="neg_mean_squared_error", cv=5
        )
    elif model_type == "Lasso":
        model = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5)
    else:
        raise ValueError("model_type should be either 'Ridge' or 'Lasso'")

    # Create a pipeline with the scaler and the regression model
    regression_pipeline = Pipeline([("scaler", scaler), ("regression", model)])

    # Fit the pipeline on the training data_route
    regression_pipeline.fit(X_train, y_train)
    logging.info(f"{model_type} regression: fitted")

    return regression_pipeline


def train_models(X, y, model_type="Lasso"):
    logging.info("Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipelines pour la mise à l'échelle et l'entraînement
    rf_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    gb_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]
    )

    rf_pipeline.fit(X_train, y_train)
    logging.info("Random Forest: trained")
    gb_pipeline.fit(X_train, y_train)
    logging.info("Gradient boosting: trained")

    # Ridge or Lasso Regression Pipeline
    regression_pipeline = regression_model(X_train, y_train, model_type=model_type)

    return regression_pipeline, rf_pipeline, gb_pipeline, X_test, y_test


def plot_feature_importance(
    model, feature_names, title, rmse, bp_test=None, non_regression_test=None, top_n=10
):
    logging.info("Dataviz: feat importance")

    steps = model.named_steps

    # Identifier et récupérer les importances des caractéristiques en fonction du modèle
    if "rf" in steps and hasattr(steps["rf"], "feature_importances_"):
        importance = steps["rf"].feature_importances_
    elif "gb" in steps and hasattr(steps["gb"], "feature_importances_"):
        importance = steps["gb"].feature_importances_
    elif "regression" in steps and hasattr(steps["regression"], "coef_"):
        importance = np.abs(steps["regression"].coef_)
    elif "ridge" in steps and hasattr(steps["ridge"].best_estimator_, "coef_"):
        importance = np.abs(steps["ridge"].best_estimator_.coef_)
    else:
        raise ValueError("Le modèle ne possède ni 'feature_importances_' ni 'coef_'.")

    # Tri des importances et sélection des principales caractéristiques
    feature_importance = pd.Series(importance, index=feature_names).sort_values(
        ascending=False
    )
    top_features = feature_importance.head(top_n)

    # Création de la figure
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_features.values, y=top_features.index, orientation="h"))

    # Titre avec RMSE, Breusch-Pagan, et autres tests
    title_text = f"{title}\nRMSE: {rmse:.2g} %"
    if bp_test is not None:
        homoscedasticity = (
            "homoscedastiques" if bp_test > 0.05 else "hétéroscédastiques"
        )
        title_text += f"    -    Résidus {homoscedasticity} (Breusch-Pagan p-value: {bp_test:.2g})"
    if non_regression_test is not None:
        non_regression_result = (
            "non régression" if non_regression_test > 0.05 else "régression présente"
        )
        title_text += f"    -    Test de non régression: {non_regression_result} (p-value: {non_regression_test:.2g})"

    # Mise à jour du layout pour un affichage similaire entre les deux modèles
    fig.update_layout(
        title=title_text,
        title_font_size=20,
        xaxis_title="Importance",
        xaxis_title_font_size=20,
        yaxis_title="Caractéristiques",
        yaxis_title_font_size=20,
        yaxis_tickfont_size=18,
        xaxis_tickfont_size=18,
        autosize=True,
        height=None,
        width=None,
    )

    return fig


def visualize_model_results(
    regression_model, rf_model, gb_model, X_test, y_test, delta_y
):
    # Prédictions et calcul de l'erreur pour chaque modèle
    logging.info("Regression model: RMSE")
    y_pred_regression = regression_model.predict(X_test)
    rmse_regression = (
        np.sqrt(mean_squared_error(y_test, y_pred_regression)) / delta_y * 100
    )
    logging.info("Random Forest: RMSE")
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf)) / delta_y * 100
    logging.info("Gradient Boosting: RMSE")
    y_pred_gb = gb_model.predict(X_test)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb)) / delta_y * 100

    # Test de Breusch-Pagan pour le modèle de régression
    logging.info("Breusch-Pagan test")
    scaler_step = regression_model.named_steps.get("scaler")
    if scaler_step:
        X_test_scaled = scaler_step.transform(X_test)
    else:
        logging.error(
            "The scaler step is not found in the pipeline. Skipping scaling step."
        )
        X_test_scaled = (
            X_test.values
        )  # Convertir en array numpy si X_test est un DataFrame

    X_test_selected = X_test_scaled

    X_test_selected = sm.add_constant(X_test_selected)
    bp_test = het_breuschpagan(y_test - y_pred_regression, X_test_selected)[1]

    # Test de non régression (Harvey-Collier)
    # logging.info("Harvey-Collier test")
    # if np.linalg.matrix_rank(X_test_selected) < X_test_selected.shape[1]:
    #     logging.error("The initial regressor matrix is singular. Adjusting skip parameter.")
    #     skip_value = X_test_selected.shape[1] + 1
    # else:
    #     skip_value = 10
    #
    # non_regression_test = linear_harvey_collier(sm.OLS(y_test, X_test_selected).fit(), skip=skip_value)[1]
    non_regression_test = None

    # Importance des caractéristiques
    plotly_FIregression = plot_feature_importance(
        regression_model,
        X_test.columns,
        "Importance des caractéristiques (Regression)",
        rmse_regression,
        bp_test,
        non_regression_test,
        top_n=10,
    )
    plotly_FIrf = plot_feature_importance(
        rf_model,
        X_test.columns,
        "Importance des caractéristiques (Random Forest)",
        rmse_rf,
        top_n=10,
    )
    plotly_FIgb = plot_feature_importance(
        gb_model,
        X_test.columns,
        "Importance des caractéristiques (Gradient Boosting)",
        rmse_gb,
        top_n=10,
    )

    return plotly_FIregression, plotly_FIrf, plotly_FIgb


def model_fissures_with_explanatory_vars(
    df_paliers_old, df_paliers_new, target="Valeur moyenne"
):
    """
    Modélisation des paliers avec des variables explicatives supplémentaires basées sur la résistance des matériaux,
    les caractéristiques des IPN, et le tassement différentiel.

    Arguments:
    - df_paliers_old: DataFrame des paliers pour la première phase
    - df_paliers_new: DataFrame des paliers pour la deuxième phase
    - target: 'Valeur moyenne' pour modéliser l'écartement, ou 'Palier_Duration' pour modéliser la durée des paliers.

    Retourne:
    - Les résultats de modélisation (pipelines et scores) avec les RMSE en pourcentage, et les importances des variables.
    """

    # Combiner les deux séries de paliers dans un DataFrame
    df_paliers_combined = pd.concat([df_paliers_old, df_paliers_new])

    if "Début" not in df_paliers_combined.columns:
        print("Erreur: La colonne 'Début' est absente du DataFrame combiné.")

    # Calculer l'âge du bâtiment pour chaque palier
    construction_year = 1959
    df_paliers_combined["Building_Age"] = (
        df_paliers_combined["Début"].dt.year - construction_year
    )
    df_paliers_combined["Building_Age"] = df_paliers_combined["Building_Age"].apply(
        lambda x: max(x, 1)
    )  # Évite âge nul

    ### Calcul du moment d'inertie des IPN (section en H) ###
    b_aile = 0.15  # longueur des ailes en mètre
    h_aile = 0.01  # épaisseur des ailes en mètre
    A_aile = b_aile * h_aile  # Aire des ailes
    d_aile = 0.15  # demi-hauteur des ailes (en mètre)

    I_aile = 2 * ((b_aile * h_aile**3) / 12 + A_aile * d_aile**2)
    b_central = 0.30  # largeur de la barre centrale en mètre
    h_central = 0.015  # épaisseur de la barre centrale en mètre
    I_central = (b_central * h_central**3) / 12
    df_paliers_combined["IPN_Moment_Inertia"] = I_aile + I_central

    E_acier = 210 * 10**9  # module de Young de l'acier en Pascals
    df_paliers_combined["IPN_Rigidite_Flexion"] = (
        E_acier * df_paliers_combined["IPN_Moment_Inertia"]
    )
    df_paliers_combined["IPN_Section"] = b_aile * h_aile + b_central * h_central
    df_paliers_combined["IPN_Stress_Factor"] = 1 / df_paliers_combined["IPN_Section"]

    ### Tassement différentiel ###
    installation_year_ipn = 2016
    df_paliers_combined["IPN_Age"] = (
        df_paliers_combined["Début"].dt.year - installation_year_ipn
    )
    df_paliers_combined["IPN_Age"] = df_paliers_combined["IPN_Age"].apply(
        lambda x: max(x, 1)
    )
    df_paliers_combined["Tassement_Differentiel_IPN"] = np.log1p(
        df_paliers_combined["IPN_Age"]
    )
    df_paliers_combined["Tassement_Mur"] = (
        np.log1p(df_paliers_combined["Building_Age"]) * 0.5
    )
    df_paliers_combined["Tassement_Colline"] = (
        np.log1p(df_paliers_combined["Building_Age"]) * 0.1
    )

    ### Variables explicatives existantes ###
    df_paliers_combined["Corrosion_Index"] = np.log1p(
        df_paliers_combined["Building_Age"]
    )
    df_paliers_combined["Fatigue_Factor"] = np.sqrt(df_paliers_combined["Building_Age"])
    df_paliers_combined["Degradation_Factor"] = np.exp(
        -0.01 * df_paliers_combined["Building_Age"]
    )
    df_paliers_combined["Palier_Duration"] = (
        df_paliers_combined["Fin"] - df_paliers_combined["Début"]
    ).dt.days

    ### Variable cible (target) ###
    X = df_paliers_combined[
        [
            "Building_Age",
            "Corrosion_Index",
            "Fatigue_Factor",
            "Degradation_Factor",
            "IPN_Rigidite_Flexion",
            "IPN_Stress_Factor",
            "Tassement_Differentiel_IPN",
            "Tassement_Mur",
            "Tassement_Colline",
        ]
    ]
    y = df_paliers_combined[target]

    # Vérifier et nettoyer les valeurs infinies ou NaN dans X
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # Séparation en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modèles permettant l'interprétation des variables
    models = {
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    GridSearchCV(
                        Ridge(max_iter=10000),
                        param_grid={"alpha": np.logspace(-6, 6, 13)},
                        cv=5,
                        scoring="neg_mean_squared_error",
                    ),
                ),
            ]
        ),
    }

    # Entraînement des modèles et stockage des résultats avec RMSE en pourcentage et importance des caractéristiques
    model_results = {}
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = (
            np.sqrt(mean_squared_error(y_test, y_pred)) / (y.max() - y.min()) * 100
        )  # RMSE en pourcentage

        # Extraire les importances des variables pour les modèles qui le permettent
        if model_name == "Random Forest":
            feature_importances = pipeline.named_steps["rf"].feature_importances_
        elif model_name == "Gradient Boosting":
            feature_importances = pipeline.named_steps["gb"].feature_importances_
        elif model_name == "Ridge":
            feature_importances = np.abs(
                pipeline.named_steps["ridge"].best_estimator_.coef_
            )

        model_results[model_name] = {
            "pipeline": pipeline,
            "rmse": rmse,
            "feature_importances": feature_importances,
            "features": X.columns.tolist(),
        }

        # Après entraînement des modèles :
        model_results = {}
        for model_name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rmse = (
                np.sqrt(mean_squared_error(y_test, y_pred)) / (y.max() - y.min()) * 100
            )  # RMSE en pourcentage

            # Extraire les importances des variables pour les modèles qui le permettent
            if model_name == "Random Forest":
                feature_importances = pipeline.named_steps["rf"].feature_importances_
            elif model_name == "Gradient Boosting":
                feature_importances = pipeline.named_steps["gb"].feature_importances_
            elif model_name == "Ridge":
                feature_importances = np.abs(
                    pipeline.named_steps["ridge"].best_estimator_.coef_
                )

            # Utilisation de plot_feature_importance
            feature_importance_plot = plot_feature_importance(
                pipeline,
                X.columns,
                title=f"Importance des variables - {model_name}",
                rmse=rmse,
            )

            model_results[model_name] = {
                "pipeline": pipeline,
                "rmse": rmse,
                "feature_importances": feature_importances,
                "features": X.columns.tolist(),
                "plot": feature_importance_plot,
            }

        return model_results
