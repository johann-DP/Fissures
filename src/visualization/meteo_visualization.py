import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import sem, t


def visualize_normalized_boxplots(df_normalized):
    # Définir une palette de couleurs plus neutre
    neutral_color_scale = [
        "#4C78A8",  # Blue
        "#F58518",  # Orange
        "#E45756",  # Red
        "#72B7B2",  # Teal
        "#54A24B",  # Green
        "#ECAE8C",  # Sand
    ]

    # Transformer le DataFrame pour avoir une colonne pour les catégories
    df_melt = df_normalized.melt(var_name="Variables", value_name="Valeurs")

    # Créer la figure de boxplot avec Plotly Express
    fig = px.box(
        df_melt,
        x="Variables",
        y="Valeurs",
        color="Variables",
        points="all",
        color_discrete_sequence=neutral_color_scale,
    )

    # Mettre à jour les titres et les étiquettes des axes
    fig.update_layout(
        title="Boxplots des variables météorologiques normalisées",
        font=dict(size=20),
        xaxis_title="Variables",
        yaxis_title="Valeurs",
        autosize=True,
        legend_title_font=dict(size=18),
        legend=dict(font=dict(size=10)),
    )

    # Mettre à jour la taille des markers
    fig.update_traces(marker=dict(size=4))

    return fig


def plot_humidity(df_cleaned):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    # Subplot pour les humidités intérieure et extérieure
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Hum(%)"],
            mode="lines",
            name="Indoor Humidity",
            line=dict(color="blue", width=1),
            opacity=0.3,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Hum(%) MA"],
            mode="lines",
            name="Indoor Humidity MA",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Hum(%)"],
            mode="lines",
            name="Outdoor Humidity",
            line=dict(color="green", width=1),
            opacity=0.3,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Hum(%) MA"],
            mode="lines",
            name="Outdoor Humidity MA",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
    )

    # Subplot pour les quantités d'eau par mètre cube intérieur
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Water Content (g/m³)"],
            mode="lines",
            name="Indoor Water Content",
            line=dict(color="lightblue", width=1),
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Water Content (g/m³) MA"],
            mode="lines",
            name="Indoor Water Content MA",
            line=dict(color="lightblue", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Water Content Max (g/m³)"],
            mode="lines",
            name="Indoor Water Content Max",
            line=dict(color="orange", width=1),
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Water Content Max (g/m³) MA"],
            mode="lines",
            name="Indoor Water Content Max MA",
            line=dict(color="orange", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Water Content Min (g/m³)"],
            mode="lines",
            name="Indoor Water Content Min",
            line=dict(color="darkblue", width=1),
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Water Content Min (g/m³) MA"],
            mode="lines",
            name="Indoor Water Content Min MA",
            line=dict(color="darkblue", width=2),
        ),
        row=2,
        col=1,
    )

    # Subplot pour les quantités d'eau par mètre cube extérieur
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Water Content (g/m³)"],
            mode="lines",
            name="Outdoor Water Content",
            line=dict(color="lightgreen", width=1),
            opacity=0.3,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Water Content (g/m³) MA"],
            mode="lines",
            name="Outdoor Water Content MA",
            line=dict(color="lightgreen", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Water Content Max (g/m³)"],
            mode="lines",
            name="Outdoor Water Content Max",
            line=dict(color="red", width=1),
            opacity=0.3,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Water Content Max (g/m³) MA"],
            mode="lines",
            name="Outdoor Water Content Max MA",
            line=dict(color="red", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Water Content Min (g/m³)"],
            mode="lines",
            name="Outdoor Water Content Min",
            line=dict(color="darkgreen", width=1),
            opacity=0.3,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Water Content Min (g/m³) MA"],
            mode="lines",
            name="Outdoor Water Content Min MA",
            line=dict(color="darkgreen", width=2),
        ),
        row=3,
        col=1,
    )

    # Mise en forme de la disposition de la figure
    fig.update_layout(
        title="Évolution de l'humidité et des quantités d'eau par mètre cube d'air",
        height=1100,
        showlegend=True,
        legend=dict(font=dict(size=10)),
        font=dict(size=20),
    )

    fig.update_yaxes(title_text="Humidity (%)", title_font=dict(size=18), row=1, col=1)
    fig.update_yaxes(title_text="Indoor Water Content (g/m³)", title_font=dict(size=18), row=2, col=1)
    fig.update_yaxes(title_text="Outdoor Water Content (g/m³)", title_font=dict(size=18), row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_layout(
        annotations=[
            dict(
                x="2024-03-01",  # Date pour le 1er mars 2024
                y=10,  # Valeur de Water Content
                text="Chauffage \"hors gel\"",
                showarrow=True,
                arrowhead=2,
                ax=0,  # Décalage horizontal du texte par rapport à la flèche
                ay=-80,  # Décalage vertical du texte par rapport à la flèche
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
                xref="x2", yref="y2",
            ),
            dict(
                x="2024-07-01",  # Date pour le 1er juillet 2024
                y=8,  # Valeur de Water Content
                text="Coupure internet",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=40,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
                xref="x2", yref="y2",
            ),
            dict(
                x="2024-11-21",  # Date pour le 21 novembre 2024
                y=12,  # Valeur de Water Content
                text="Panne fibre",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
                xref="x2", yref="y2",
            ),
        ]
    )

    return fig


def plot_temperature_extremes(df_cleaned):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Tem.Max(°C)"],
            mode="lines",
            name="Indoor Max Temperature",
            line=dict(color="blue", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Tem.Max(°C) MA"],
            mode="lines",
            name="Indoor Max Temperature MA",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Tem.Min(°C)"],
            mode="lines",
            name="Indoor Min Temperature",
            line=dict(color="green", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Tem.Min(°C) MA"],
            mode="lines",
            name="Indoor Min Temperature MA",
            line=dict(color="green", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Tem.Max(°C)"],
            mode="lines",
            name="Outdoor Max Temperature",
            line=dict(color="purple", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Tem.Max(°C) MA"],
            mode="lines",
            name="Outdoor Max Temperature MA",
            line=dict(color="purple", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Tem.Min(°C)"],
            mode="lines",
            name="Outdoor Min Temperature",
            line=dict(color="orange", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Tem.Min(°C) MA"],
            mode="lines",
            name="Outdoor Min Temperature MA",
            line=dict(color="orange", width=2),
        )
    )

    fig.update_layout(
        title="Évolution des températures maximales et minimales",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        font=dict(size=20),
        legend=dict(font=dict(size=10)),
        autosize=True,
        height=None,
        width=None,
    )

    fig.update_layout(
        annotations=[
            dict(
                x="2024-03-01",  # Date pour le 1er mars 2024
                y=20,  # Valeur de température
                text="Chauffage \"hors gel\"",
                showarrow=True,
                arrowhead=2,
                ax=0,  # Décalage horizontal du texte par rapport à la flèche
                ay=-120,  # Décalage vertical du texte par rapport à la flèche
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
            ),
            dict(
                x="2024-07-01",  # Date pour le 1er juillet 2024
                y=27,  # Valeur de température
                text="Coupure internet",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-220,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
            ),
            dict(
                x="2024-11-21",  # Date pour le 21 novembre 2024
                y=22,  # Valeur de température
                text="Panne fibre",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-100,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
            ),
        ]
    )

    return fig


def plot_precipitation(df_cleaned):
    # Création des subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Moyennes hebdomadaires des précipitations (échelle normale)",
            "Évolution des précipitations (échelle log)",
        ],
        vertical_spacing=0.05,
    )

    # Calcul des moyennes hebdomadaires
    weekly_means = (
        df_cleaned.groupby(df_cleaned["Time"].dt.to_period("W"))["Rainfull(Day)(mm)"]
        .mean()
        .reset_index()
    )
    # Décalage de 2 jours pour la courbe du subplot du haut
    weekly_means["Time"] = weekly_means["Time"].dt.start_time + pd.Timedelta(days=2)

    # Supprimer la première moyenne hebdomadaire (semaine incomplète)
    weekly_means = weekly_means.iloc[1:]

    # Calcul des moyennes hebdomadaires sans décalage pour le subplot du bas
    weekly_means_no_shift = (
        df_cleaned.groupby(df_cleaned["Time"].dt.to_period("W"))["Rainfull(Day)(mm)"]
        .mean()
        .reset_index()
    )
    weekly_means_no_shift["Time"] = weekly_means_no_shift["Time"].dt.start_time
    # Supprimer la première moyenne hebdomadaire (semaine incomplète)
    weekly_means_no_shift = weekly_means_no_shift.iloc[1:]

    # Définir les limites de l'axe des abscisses
    x_min = df_cleaned["Time"].min()
    x_max = df_cleaned["Time"].max()

    # Subplot row=1 : Moyennes hebdomadaires en échelle normale (avec décalage)
    fig.add_trace(
        go.Scatter(
            x=weekly_means["Time"],
            y=weekly_means["Rainfull(Day)(mm)"],
            mode="lines+markers",
            name="Weekly Means (Normal Scale)",
            line=dict(color="coral", width=3),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    # Subplot row=2 : Évolution des précipitations en échelle logarithmique
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Rainfull(Hour)(mm)"],
            mode="lines",
            name="Rainfall per Hour",
            line=dict(width=1, color="rgba(0, 0, 255, 0.5)"),
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Rainfull(Day)(mm)"],
            mode="lines",
            name="Rainfall per Day",
            line=dict(width=2, color="rgba(0, 0, 139, 0.5)"),
        ),
        row=2,
        col=1,
    )

    # Moyennes hebdomadaires pour le subplot en échelle logarithmique secondaire (sans décalage)
    for i in range(len(weekly_means_no_shift) - 1):
        # Ligne horizontale pour une semaine (échelle logarithmique secondaire)
        fig.add_trace(
            go.Scatter(
                x=[
                    weekly_means_no_shift["Time"].iloc[i],
                    weekly_means_no_shift["Time"].iloc[i + 1],
                ],
                y=[
                    weekly_means_no_shift["Rainfull(Day)(mm)"].iloc[i],
                    weekly_means_no_shift["Rainfull(Day)(mm)"].iloc[i],
                ],
                mode="lines",
                line=dict(color="coral", width=3),
                name="Weekly Mean (Log Scale)",
                showlegend=(i == 0),
                yaxis="y3",  # Associer à l'échelle secondaire à droite
            ),
            row=2,
            col=1,
        )

        # Ligne verticale pour relier les paliers (échelle logarithmique secondaire)
        fig.add_trace(
            go.Scatter(
                x=[
                    weekly_means_no_shift["Time"].iloc[i + 1],
                    weekly_means_no_shift["Time"].iloc[i + 1],
                ],
                y=[
                    weekly_means_no_shift["Rainfull(Day)(mm)"].iloc[i],
                    weekly_means_no_shift["Rainfull(Day)(mm)"].iloc[i + 1],
                ],
                mode="lines",
                line=dict(color="coral", width=3, dash="dot"),
                showlegend=False,
                yaxis="y3",  # Associer à l'échelle secondaire à droite
            ),
            row=2,
            col=1,
        )

    # Ajout des shapes (bandes gris clair) pour chaque subplot
    months = pd.date_range(start=x_min, end=x_max, freq="MS")
    shapes = []
    for i in range(0, len(months) - 1, 2):  # Un mois sur deux
        # Bandes pour le subplot du haut
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=months[i],
                x1=months[i + 1],
                y0=0.525,  # Limite basse du subplot du haut
                y1=1.0,  # Limite haute du subplot du haut
                fillcolor="rgba(211, 211, 211, 0.3)",
                line=dict(width=0),
                layer="below",
            )
        )
        # Bandes pour le subplot du bas
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=months[i],
                x1=months[i + 1],
                y0=0.0,  # Limite basse du subplot du bas
                y1=0.472,  # Limite haute du subplot du bas
                fillcolor="rgba(211, 211, 211, 0.3)",
                line=dict(width=0),
                layer="below",
            )
        )

    # Mise en page finale
    fig.update_layout(
        shapes=shapes,
        title="Analyse des précipitations hebdomadaires et journalières",
        font=dict(size=20),
        xaxis=dict(
            range=[x_min, x_max],
        ),
        xaxis2=dict(title="Time", range=[x_min, x_max]),  # Titre uniquement sur le subplot du bas
        yaxis=dict(title="Rainfall (mm)", type="linear"),
        yaxis2=dict(title="Rainfall (mm)", type="log"),
        yaxis3=dict(
            title="Logarithmic Scale (Coral)",
            type="log",
            overlaying="y2",
            side="right",
        ),
        legend=dict(font=dict(size=10)),
        autosize=True,
        margin=dict(t=100, l=80),  # Ajuster les marges pour aligner le titre et les sous-titres
    )

    # Aligner sur les sous-titres à gauche
    fig.update_annotations(dict(xanchor="left", x=0.015))

    # Ajouter l'annotation pour "Panne fibre" sans remplacer les sous-titres
    fig.add_annotation(
        x="2024-11-21",  # Date pour le 21 novembre 2024
        y=np.log10(10),  # Valeur arbitraire de précipitation
        text="Panne fibre",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-80,
        font=dict(size=14, color="gray"),
        arrowcolor="gray",
        align="center",  # Centrer le texte
        xanchor="center",  # Centrer par rapport à x
        xref="x",  # Référencer x en coordonnées données
        yref="y2",  # Référencer y dans le subplot en échelle log
    )

    return fig


def plot_wind_speed_direction(df_cleaned):
    # Création des subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Moyennes hebdomadaires de la vitesse du vent (échelle normale)",
            "Évolution de la vitesse du vent (échelle log)",
        ],
        vertical_spacing=0.05,
    )

    # Calcul des moyennes hebdomadaires
    weekly_means = (
        df_cleaned.groupby(df_cleaned["Time"].dt.to_period("W"))["Wind speed(km/h)"]
        .mean()
        .reset_index()
    )
    # Décalage de 2 jours pour la courbe purple uniquement dans le subplot du haut
    weekly_means["Time"] = weekly_means["Time"].dt.start_time + pd.Timedelta(days=2)

    # Supprimer la première moyenne hebdomadaire (semaine incomplète)
    weekly_means = weekly_means.iloc[1:]

    # Calcul des moyennes hebdomadaires sans décalage
    weekly_means_no_shift = (
        df_cleaned.groupby(df_cleaned["Time"].dt.to_period("W"))["Wind speed(km/h)"]
        .mean()
        .reset_index()
    )
    weekly_means_no_shift["Time"] = weekly_means_no_shift["Time"].dt.start_time
    # Supprimer la première moyenne hebdomadaire (semaine incomplète)
    weekly_means_no_shift = weekly_means_no_shift.iloc[1:]

    # Définir les limites de l'axe des abscisses
    x_min = df_cleaned["Time"].min()
    x_max = df_cleaned["Time"].max()

    # Subplot row=1 : Moyennes hebdomadaires en échelle normale (avec décalage)
    fig.add_trace(
        go.Scatter(
            x=weekly_means["Time"],
            y=weekly_means["Wind speed(km/h)"],
            mode="lines+markers",
            name="Weekly Means (Normal Scale)",
            line=dict(color="purple", width=3),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    # Subplot row=2 : Évolution de la vitesse du vent en échelle logarithmique
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Wind speed(km/h)"],
            mode="lines",
            name="Wind Speed",
            line=dict(color="red", width=1),
            opacity=0.15,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Wind speed(km/h) MA"],
            mode="lines",
            name="Wind Speed MA",
            line=dict(color="red", width=2),
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    # Moyennes hebdomadaires pour le subplot en échelle logarithmique secondaire (sans décalage)
    for i in range(len(weekly_means_no_shift) - 1):
        # Ligne horizontale pour une semaine (échelle logarithmique secondaire)
        fig.add_trace(
            go.Scatter(
                x=[
                    weekly_means_no_shift["Time"].iloc[i],
                    weekly_means_no_shift["Time"].iloc[i + 1],
                ],
                y=[
                    weekly_means_no_shift["Wind speed(km/h)"].iloc[i],
                    weekly_means_no_shift["Wind speed(km/h)"].iloc[i],
                ],
                mode="lines",
                line=dict(color="purple", width=3),
                name="Weekly Mean (Log Scale)",
                showlegend=(i == 0),
                yaxis="y3",  # Associer à l'échelle secondaire à droite
            ),
            row=2,
            col=1,
        )

        # Ligne verticale pour relier les paliers (échelle logarithmique secondaire)
        fig.add_trace(
            go.Scatter(
                x=[
                    weekly_means_no_shift["Time"].iloc[i + 1],
                    weekly_means_no_shift["Time"].iloc[i + 1],
                ],
                y=[
                    weekly_means_no_shift["Wind speed(km/h)"].iloc[i],
                    weekly_means_no_shift["Wind speed(km/h)"].iloc[i + 1],
                ],
                mode="lines",
                line=dict(color="purple", width=3, dash="dot"),
                showlegend=False,
                yaxis="y3",  # Associer à l'échelle secondaire à droite
            ),
            row=2,
            col=1,
        )

    # Ajout des shapes (bandes gris clair) pour chaque subplot
    months = pd.date_range(start=x_min, end=x_max, freq="MS")
    shapes = []
    for i in range(0, len(months) - 1, 2):  # Un mois sur deux
        # Bandes pour le subplot du haut
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=months[i],
                x1=months[i + 1],
                y0=0.525,  # Limite basse du subplot du haut
                y1=1.0,  # Limite haute du subplot du haut
                fillcolor="rgba(211, 211, 211, 0.3)",
                line=dict(width=0),
                layer="below",
            )
        )
        # Bandes pour le subplot du bas
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=months[i],
                x1=months[i + 1],
                y0=0.0,  # Limite basse du subplot du bas
                y1=0.472,  # Limite haute du subplot du bas
                fillcolor="rgba(211, 211, 211, 0.3)",
                line=dict(width=0),
                layer="below",
            )
        )

    # Mise en page finale
    fig.update_layout(
        shapes=shapes,
        title="Analyse des vitesses hebdomadaires et journalières du vent",
        font=dict(size=20),
        xaxis=dict(
            range=[x_min, x_max],
        ),
        xaxis2=dict(title="Time", range=[x_min, x_max]),  # Titre uniquement sur le subplot du bas
        yaxis=dict(title="Wind Speed (km/h)", type="linear"),
        yaxis2=dict(title="Wind Speed (km/h)", type="log"),
        yaxis3=dict(
            title="Logarithmic Scale (Purple)",
            type="log",
            overlaying="y2",
            side="right",
        ),
        legend=dict(font=dict(size=10)),
        autosize=True,
        margin=dict(t=100, l=80),  # Ajuster les marges pour aligner le titre et les sous-titres
    )

    # Aligner sur les sous-titres à gauche
    fig.update_annotations(dict(xanchor="left", x=0.015))

    # Ajouter les annotations pour "Coupure internet" et "Panne fibre" sans remplacer les sous-titres
    fig.add_annotation(
        x="2024-07-01",  # Date pour le 1er juillet 2024
        y=np.log10(5),  # Valeur logarithmique de la vitesse
        text="Coupure internet",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(size=14, color="gray"),
        arrowcolor="gray",
        align="center",  # Centrer le texte
        xanchor="center",  # Centrer par rapport à x
        xref="x",  # Référencer x en coordonnées données
        yref="y2",  # Référencer y dans le subplot en échelle log
    )

    fig.add_annotation(
        x="2024-11-21",  # Date pour le 21 novembre 2024
        y=np.log10(7),  # Valeur logarithmique de la vitesse
        text="Panne fibre",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-80,
        font=dict(size=14, color="gray"),
        arrowcolor="gray",
        align="center",  # Centrer le texte
        xanchor="center",  # Centrer par rapport à x
        xref="x",  # Référencer x en coordonnées données
        yref="y2",  # Référencer y dans le subplot en échelle log
    )

    # # Graphique pour la direction du vent avec un subplot supplémentaire pour l'angle
    # fig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
    #
    # # Traces pour sinus et cosinus
    # fig2.add_trace(
    #     go.Scatter(
    #         x=df_cleaned["Time"],
    #         y=df_cleaned["Wind direction sin"],
    #         mode="lines",
    #         name="Wind Direction Sin",
    #         line=dict(color="blue", width=1),
    #         opacity=0.15,
    #     ),
    #     row=2,
    #     col=1,
    # )
    # fig2.add_trace(
    #     go.Scatter(
    #         x=df_cleaned["Time"],
    #         y=df_cleaned["Wind direction sin MA"],
    #         mode="lines",
    #         name="Wind Direction Sin MA",
    #         line=dict(color="blue", width=2),
    #     ),
    #     row=2,
    #     col=1,
    # )
    # fig2.add_trace(
    #     go.Scatter(
    #         x=df_cleaned["Time"],
    #         y=df_cleaned["Wind direction cos"],
    #         mode="lines",
    #         name="Wind Direction Cos",
    #         line=dict(color="green", width=1),
    #         opacity=0.15,
    #     ),
    #     row=2,
    #     col=1,
    # )
    # fig2.add_trace(
    #     go.Scatter(
    #         x=df_cleaned["Time"],
    #         y=df_cleaned["Wind direction cos MA"],
    #         mode="lines",
    #         name="Wind Direction Cos MA",
    #         line=dict(color="green", width=2),
    #     ),
    #     row=2,
    #     col=1,
    # )
    #
    # # Trace pour l'angle de direction du vent
    # fig2.add_trace(
    #     go.Scatter(
    #         x=df_cleaned["Time"],
    #         y=df_cleaned["Wind direction"],
    #         mode="lines",
    #         name="Wind Direction",
    #         line=dict(color="darkblue", width=2),
    #         opacity=0.15,
    #     ),
    #     row=1,
    #     col=1,
    # )
    # fig2.add_trace(
    #     go.Scatter(
    #         x=df_cleaned["Time"],
    #         y=df_cleaned["Wind direction MA"],
    #         mode="lines",
    #         name="Wind Direction MA",
    #         line=dict(color="darkblue", width=2),
    #         opacity=0.5,
    #     ),
    #     row=1,
    #     col=1,
    # )
    #
    # fig2.update_layout(
    #     title="Analyse détaillée de la direction du vent",
    #     font=dict(size=20),
    #     legend=dict(font=dict(size=10)),
    #     autosize=True,
    # )
    #
    # fig2.update_xaxes(title_text="Date", row=2, col=1)
    # fig2.update_yaxes(title_text="Direction (sin et cos)", row=2, col=1)
    # fig2.update_yaxes(title_text="Angle (°)", row=1, col=1)

    # Creating the plotly wind rose figure

    fig2 = go.Figure()

    # Conversion des degrés en radians pour la visualisation
    df_cleaned["Wind direction radians"] = np.deg2rad(df_cleaned["Wind direction"])

    # Ajouter la trace principale pour la direction et la vitesse du vent
    fig2.add_trace(
        go.Scatterpolar(
            r=np.log1p(
                df_cleaned["Wind speed(km/h)"]
            ),  # Échelle logarithmique pour la vitesse du vent
            theta=-np.degrees(
                df_cleaned["Wind direction radians"]
            ),  # Conversion en degrés avec inversion de l'axe
            mode="markers",
            name="Wind Speed and Direction",
            marker=dict(
                color=df_cleaned[
                    "Wind speed(km/h)"
                ],  # Couleur en fonction de la vitesse du vent
                colorscale="thermal",
                reversescale=True,
                cmin=0,
                cmax=df_cleaned["Wind speed(km/h)"].max(),
                size=10,  # Taille des marqueurs
                opacity=(
                    df_cleaned["Wind speed(km/h)"]
                    - df_cleaned["Wind speed(km/h)"].min()
                )
                / (
                    df_cleaned["Wind speed(km/h)"].max()
                    - df_cleaned["Wind speed(km/h)"].min()
                )
                * (1 - np.exp(-0.6 * df_cleaned["Wind speed(km/h)"])),
                colorbar=dict(
                    title=dict(
                        text="Wind Speed (km/h)", side="right", font=dict(size=18)
                    ),  # Titre de la barre de couleur
                    tickvals=[1, 2, 5, 10, 15],  # Valeurs correspondant à l'échelle log
                    ticktext=[" 1", " 2", " 5", " 10", "15"],  # Textes correspondants
                    thickness=50,  # Épaisseur de la barre de couleur
                    len=0.8,  # Longueur de la barre de couleur
                    x=0.85,
                ),
            ),
        )
    )

    # CALCULS POUR LA MOYENNE CLASSIQUE
    mean_direction = np.degrees(
        np.arctan2(
            np.sin(df_cleaned["Wind direction radians"]).mean(),
            np.cos(df_cleaned["Wind direction radians"]).mean(),
        )
    )
    mean_speed = df_cleaned["Wind speed(km/h)"].mean()

    # Coordonnées et marges pour la moyenne classique
    r_center = mean_speed
    theta_center = -mean_direction
    percentile_2_5 = np.percentile(df_cleaned["Wind speed(km/h)"], 2.5)
    percentile_97_5 = np.percentile(df_cleaned["Wind speed(km/h)"], 97.5)
    r_lower = max(percentile_2_5, 0)
    r_upper = percentile_97_5
    mean_direction_x = np.cos(df_cleaned["Wind direction radians"]).mean()
    mean_direction_y = np.sin(df_cleaned["Wind direction radians"]).mean()
    R = np.sqrt(mean_direction_x**2 + mean_direction_y**2)
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else 360  # Gestion des cas limites
    margin_direction = np.degrees(circular_std)
    theta_lower = theta_center - margin_direction
    theta_upper = theta_center + margin_direction
    theta_range = np.linspace(theta_lower, theta_upper, 100)
    r_values = np.concatenate(
        [np.full_like(theta_range, r_lower), np.full_like(theta_range, r_upper)[::-1]]
    )
    theta_values = np.concatenate([theta_range, theta_range[::-1]])

    # CALCULS POUR LA MOYENNE PONDÉRÉE
    weights = df_cleaned["Wind speed(km/h)"] / df_cleaned["Wind speed(km/h)"].max()

    # Calcul de la direction moyenne pondérée
    mean_direction_weighted = np.degrees(
        np.arctan2(
            np.sum(np.sin(df_cleaned["Wind direction radians"]) * weights),
            np.sum(np.cos(df_cleaned["Wind direction radians"]) * weights),
        )
    )

    # Calcul de la vitesse moyenne pondérée
    mean_speed_weighted = np.sum(df_cleaned["Wind speed(km/h)"] * weights) / np.sum(
        weights
    )
    r_center_weighted = mean_speed_weighted
    theta_center_weighted = -mean_direction_weighted

    # Calcul des percentiles pondérés pour les marges radiales
    sorted_speeds = np.sort(df_cleaned["Wind speed(km/h)"])
    sorted_weights = weights[np.argsort(df_cleaned["Wind speed(km/h)"])]
    cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    percentile_2_5_weighted = sorted_speeds[np.searchsorted(cumulative_weights, 0.025)]
    percentile_97_5_weighted = sorted_speeds[np.searchsorted(cumulative_weights, 0.975)]
    r_lower_weighted = max(percentile_2_5_weighted, 0)
    r_upper_weighted = percentile_97_5_weighted

    # Normalisation des composantes directionnelles
    mean_direction_x_weighted = np.sum(
        np.cos(df_cleaned["Wind direction radians"]) * weights
    ) / np.sum(weights)
    mean_direction_y_weighted = np.sum(
        np.sin(df_cleaned["Wind direction radians"]) * weights
    ) / np.sum(weights)

    # Calcul du facteur de concentration R_weighted
    R_weighted = np.sqrt(mean_direction_x_weighted**2 + mean_direction_y_weighted**2)

    # Gestion des cas limites pour la dispersion circulaire
    if R_weighted > 0:
        circular_std_weighted = np.sqrt(-2 * np.log(R_weighted))
    else:
        circular_std_weighted = (
            360  # Dispersion circulaire maximale dans les cas limites
        )

    # Calcul des marges angulaires
    margin_direction_weighted = np.degrees(circular_std_weighted)
    theta_lower_weighted = theta_center_weighted - margin_direction_weighted
    theta_upper_weighted = theta_center_weighted + margin_direction_weighted

    # Recalcul des coordonnées pour la zone bleue
    theta_range_weighted = np.linspace(theta_lower_weighted, theta_upper_weighted, 100)
    r_values_weighted = np.concatenate(
        [
            np.full_like(theta_range_weighted, r_lower_weighted),
            np.full_like(theta_range_weighted, r_upper_weighted)[::-1],
        ]
    )
    theta_values_weighted = np.concatenate(
        [theta_range_weighted, theta_range_weighted[::-1]]
    )

    # Dimensions fixes pour les triangles d'extrémité
    triangle_length = 0.01  # Longueur de la base (fixe)
    triangle_height = 0.1  # Hauteur du triangle (fixe)

    # Fonction pour ajouter un triangle d'extrémité
    def add_arrow_head(fig, r_base, theta_base, color):
        """
        Ajoute un triangle d'extrémité à une flèche.

        Args:
        - fig: Figure Plotly (go.Figure).
        - r_base: Rayon du point de base de la flèche.
        - theta_base: Angle de la flèche en degrés.
        - color: Couleur du triangle.
        """
        # Conversion de l'angle pour les coordonnées polaires
        theta_left = theta_base + np.degrees(
            np.arctan(triangle_length / (2 * triangle_height))
        )
        theta_right = theta_base - np.degrees(
            np.arctan(triangle_length / (2 * triangle_height))
        )

        # Coordonnées radiales du triangle
        r_tip = r_base + triangle_height * r_base
        r_side = r_base

        fig.add_trace(
            go.Scatterpolar(
                r=[r_tip, r_side, r_side, r_tip],
                theta=[theta_base, theta_left, theta_right, theta_base],
                mode="lines",
                fill="toself",
                line=dict(color=color, width=1),
                fillcolor=color,
                showlegend=False,
            )
        )

    # Ajout de la flèche rouge (classique)
    fig2.add_trace(
        go.Scatterpolar(
            r=[0, np.log1p(r_center)],
            theta=[0, theta_center],
            mode="lines",
            name="Mean Wind Vector",
            line=dict(color="red", width=4),
            showlegend=False,
        )
    )
    add_arrow_head(fig2, np.log1p(r_center), theta_center, "red")

    # Ajout de la zone d'incertitude rouge
    fig2.add_trace(
        go.Scatterpolar(
            r=np.log1p(r_values),
            theta=-theta_values,
            mode="lines",
            fill="toself",
            name="Confidence Zone",
            line=dict(color="rgba(255, 192, 203, 0.5)"),
            fillcolor="rgba(255, 192, 203, 0.4)",
            showlegend=False,
        )
    )

    # Ajout de la flèche bleu moyen (pondérée)
    fig2.add_trace(
        go.Scatterpolar(
            r=[0, np.log1p(r_center_weighted)],
            theta=[0, theta_center_weighted],
            mode="lines",
            name="Weighted Mean Wind Vector",
            line=dict(color="mediumblue", width=6),
            showlegend=False,
        )
    )
    add_arrow_head(
        fig2, np.log1p(r_center_weighted), theta_center_weighted, "mediumblue"
    )

    # Ajout de la zone d'incertitude bleu moyen
    fig2.add_trace(
        go.Scatterpolar(
            r=np.log1p(r_values_weighted),
            theta=-theta_values_weighted,
            mode="lines",
            fill="toself",
            name="Weighted Confidence Zone",
            line=dict(color="rgba(70, 130, 180, 0.3)"),
            fillcolor="rgba(70, 130, 180, 0.2)",
            showlegend=False,
        )
    )

    # Identification des 5 valeurs les plus élevées de vitesse du vent
    top_5_speeds = df_cleaned.nlargest(5, "Wind speed(km/h)")

    # Ajout d'une trace Scatterpolar pour les annotations des 5 valeurs les plus fortes
    fig2.add_trace(
        go.Scatterpolar(
            r=np.log1p(
                top_5_speeds["Wind speed(km/h)"]
            ),  # Appliquer une échelle logarithmique pour le rayon
            theta=-np.degrees(
                top_5_speeds["Wind direction radians"]
            ),  # Convertir les radians en degrés
            mode="text",
            text=[
                f"<br>{speed:.1f} km/h<br>{date.strftime('%Y-%m-%d')}"
                for speed, date in zip(
                    top_5_speeds["Wind speed(km/h)"], top_5_speeds["Time"]
                )
            ],  # Texte de l'annotation
            textposition="bottom left",
            textfont=dict(size=12, color="black"),
            showlegend=False,
        )
    )

    # Mise à jour de la mise en page pour une meilleure lisibilité et esthétique
    fig2.update_layout(
        width=2800,
        height=1350,
        title=dict(
            text="Visualisation radiale de la direction et de la vitesse du vent",
            font=dict(size=26),  # Taille du titre
            # x=0.5,  # Centrer le titre
            y=0.98,
        ),
        polar=dict(
            domain=dict(x=[0, 1], y=[0, 0.95]),
            radialaxis=dict(
                range=[
                    0,
                    np.log1p(df_cleaned["Wind speed(km/h)"].max()),
                ],  # Échelle logarithmique pour l'axe radial
                tickvals=np.log1p(
                    [1, 2, 5, 10, 20, 50]
                ),  # Ticks correspondant à une échelle logarithmique
                ticktext=[
                    " 1",
                    " 2",
                    " 5",
                    " 10",
                    " 20",
                    " 50",
                ],  # Textes des ticks pour plus de clarté
                showline=True,
                linewidth=1.5,
                tickangle=45,  # Orientation des étiquettes des ticks
                tickfont=dict(
                    size=14, color="blue"
                ),  # Taille et couleur des étiquettes des ticks
                gridcolor="blue",  # Couleur de la grille
                gridwidth=0.7,  # Épaisseur de la grille
            ),
            angularaxis=dict(
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],  # Points cardinaux
                ticktext=[
                    "N",
                    "NE",
                    "E",
                    "SE",
                    "S",
                    "SW",
                    "W",
                    "NW",
                ],  # Étiquettes des directions
                direction="clockwise",  # Sens de rotation conforme à la direction du vent
                rotation=90,  # Rotation pour aligner le 0° vers le haut
                tickfont=dict(
                    size=20, color="blue"
                ),  # Taille et couleur des étiquettes des points cardinaux
                gridcolor="blue",
                gridwidth=0.7,  # Épaisseur de la grille angulaire
            ),
        ),
        margin=dict(l=0, r=20, b=0, t=35),
        legend=dict(
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        autosize=False,
    )

    # Ajout des annotations pour une légende personnalisée sous le titre
    fig2.add_annotation(
        x=0.05,
        y=0.95,  # Position centrale sous le titre
        xref="paper",
        yref="paper",  # Coordonnées relatives à la figure
        text=(
            "<span style='color:rgba(229, 100, 84, 1.0); font-size:20px; text-shadow: 1px 0, -1px 0, 0 1px, 0 -1px;'>⟶</span> Mean Wind Vector<br>"
            "<span style='color:rgba(235, 219, 229, 1.0); font-size:18px;'>██</span> Confidence Zone<br>"
            "<span style='font-size:24px;'><b> </b></span><br>"
            "<span style='color:mediumblue; font-size:20px; text-shadow: 1px 0, -1px 0, 0 1px, 0 -1px;'>⟶</span> Weighted Mean Wind Vector<br>"
            "<span style='color:rgba(203, 215, 233, 1.0); font-size:18px;'>██</span> Weighted Confidence Zone"
        ),
        showarrow=False,  # Pas de flèche
        align="left",  # Aligner le texte à gauche
        font=dict(size=16),  # Taille du texte principal
    )

    return fig, fig2


def plot_light_uv(df_cleaned):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    # Trace for Light Intensity and its moving average
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Light intensity"],
            mode="lines",
            name="Light Intensity",
            line=dict(color="blue", width=1),
            opacity=0.3,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Light intensity MA"],
            mode="lines",
            name="Light Intensity MA",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Trace for UV Rating and its moving average
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["UV rating"],
            mode="lines",
            name="UV Rating",
            line=dict(color="green", width=1),
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["UV rating MA"],
            mode="lines",
            name="UV Rating MA",
            line=dict(color="green", width=2),
        ),
        row=2,
        col=1,
    )

    # Calculate and plot the ratio differences
    mean_light_intensity = df_cleaned["Light intensity"].mean()
    mean_uv_rating = df_cleaned["UV rating"].mean()
    df_cleaned["light_ratio"] = df_cleaned["Light intensity"] / mean_light_intensity
    df_cleaned["uv_ratio"] = df_cleaned["UV rating"] / mean_uv_rating
    df_cleaned["ratio_diff"] = df_cleaned["light_ratio"] - df_cleaned["uv_ratio"]
    df_cleaned["ratio_diff MA"] = df_cleaned["ratio_diff"].rolling(window=30).mean()

    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["ratio_diff"],
            mode="lines",
            name="Ratio Difference",
            line=dict(color="red", width=1),
            opacity=0.3,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["ratio_diff MA"],
            mode="lines",
            name="Ratio Difference MA",
            line=dict(color="red", width=2),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title="Évolution de l'intensité lumineuse, de l'indice UV et de leur différence de ratio",
        font=dict(size=20),
        autosize=True,
        height=None,
        width=None,
    )

    fig.update_yaxes(title_text="Light Intensity", row=1, col=1)
    fig.update_yaxes(title_text="UV Rating", row=2, col=1)
    fig.update_yaxes(title_text="Ratio Difference", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_layout(
        annotations=[
            dict(
                x="2024-07-01",  # Date pour le 1er juillet 2024
                y=8,  # Valeur d'UV Rating
                text="Coupure internet",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-70,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
                xref="x2", yref="y2",
            ),
            dict(
                x="2024-11-21",  # Date pour le 21 novembre 2024
                y=1,  # Valeur d'UV Rating
                text="Panne fibre",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-50,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
                xref="x2", yref="y2",
            ),
        ]
    )

    return fig


def plot_moving_averages(df_cleaned):
    fig = go.Figure()

    # Traces for Indoor and Outdoor Temperature
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Tem(°C)"],
            mode="lines",
            name="Indoor Temperature",
            line=dict(color="blue", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Indoor Tem(°C) MA"],
            mode="lines",
            name="Indoor Temperature MA",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Tem(°C)"],
            mode="lines",
            name="Outdoor Temperature",
            line=dict(color="green", width=1),
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_cleaned["Time"],
            y=df_cleaned["Outdoor Tem(°C) MA"],
            mode="lines",
            name="Outdoor Temperature MA",
            line=dict(color="green", width=2),
        )
    )

    fig.update_layout(
        title="Évolution de la température intérieure et extérieure",
        font=dict(size=20),
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        legend=dict(font=dict(size=10)),
        autosize=True,
        height=None,
        width=None,
    )

    fig.update_layout(
        annotations=[
            dict(
                x="2024-03-01",  # Date pour le 1er mars 2024
                y=20,  # Valeur de température
                text="Chauffage \"hors gel\"",
                showarrow=True,
                arrowhead=2,
                ax=0,  # Décalage horizontal du texte par rapport à la flèche
                ay=-120,  # Décalage vertical du texte par rapport à la flèche
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
            ),
            dict(
                x="2024-07-01",  # Date pour le 1er juillet 2024
                y=25,  # Valeur de température
                text="Coupure internet",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-180,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
            ),
            dict(
                x="2024-11-21",  # Date pour le 21 novembre 2024
                y=22,  # Valeur de température
                text="Panne fibre",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-100,
                font=dict(size=14, color="gray"),
                arrowcolor="gray",
            ),
        ]
    )

    return fig


def plot_weekly_statistics(df_cleaned, variable_base_names):
    plotly_figures = []

    for base_name in variable_base_names:
        # Filter columns corresponding to statistics of base_name
        stats_columns = [
            col
            for col in df_cleaned.columns
            if "dt" not in col
            and base_name in col
            and any(
                stat in col
                for stat in ["mean", "median", "std", "skew", "calculate_kurtosis"]
            )
        ]
        n = len(stats_columns) // 2
        stats_columns = stats_columns[:n]

        # Create subplots for means, medians, std, skewness, and kurtosis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

        # Plot mean and median on subplot 1
        for col in stats_columns:
            if "mean" in col or "median" in col:
                fig.add_trace(
                    go.Scatter(
                        x=df_cleaned["Time"],
                        y=df_cleaned[col],
                        mode="lines",
                        name=col.split("_")[1],
                    ),
                    row=1,
                    col=1,
                )

        # Plot std on subplot 2
        for col in stats_columns:
            if "std" in col:
                fig.add_trace(
                    go.Scatter(
                        x=df_cleaned["Time"],
                        y=df_cleaned[col],
                        mode="lines",
                        name=col.split("_")[1],
                        line=dict(color="blue"),
                    ),
                    row=2,
                    col=1,
                )

        # Plot skewness and kurtosis on subplot 3
        for col in stats_columns:
            if "skew" in col:
                fig.add_trace(
                    go.Scatter(
                        x=df_cleaned["Time"],
                        y=df_cleaned[col],
                        mode="lines",
                        name=col.split("_")[1],
                        line=dict(dash="dash", color="orange"),
                    ),
                    row=3,
                    col=1,
                )
            if "calculate_kurtosis" in col:
                fig.add_trace(
                    go.Scatter(
                        x=df_cleaned["Time"],
                        y=df_cleaned[col],
                        mode="lines",
                        name=col.split("_")[2],
                        line=dict(dash="dash", color="red"),
                    ),
                    row=3,
                    col=1,
                )

        # Update layout
        fig.update_layout(
            height=1100,
            showlegend=True,
            title_text=f"Évolution des statistiques pour {base_name}",
            font=dict(size=20),
        )

        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)
        fig.update_yaxes(title_text="Skewness", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Kurtosis", row=3, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Time", row=3, col=1)

        plotly_figures.append(fig)

    return plotly_figures


def plot_correlation_matrix(df_filtered):
    corr = df_filtered.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Apply mask to correlation matrix
    corr = corr.mask(mask)

    # Create a heatmap with Plotly
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale="RdBu",
        showscale=True,
        reversescale=True,
        zmin=-1,
        zmax=1,
    )

    # Update layout for better appearance
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_nticks=36,
        autosize=True,
        width=None,
        height=None,
    )

    return fig


def plot_pairplot(
    df, title="Matrices des scatterplots et distributions des variables de base"
):
    variables = df.columns
    num_vars = len(variables)
    fig = make_subplots(
        rows=num_vars,
        cols=num_vars,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    # Palette de couleurs
    color_palette = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#ECAE8C"]

    for i, var1 in enumerate(variables):
        color = color_palette[i % len(color_palette)]  # Cycle through colors
        for j, var2 in enumerate(variables):
            if i == j:  # Diagonal, plot histograms
                hist = go.Histogram(
                    x=df[var1], nbinsx=20, name=var1, marker_color=color
                )
                fig.add_trace(hist, row=i + 1, col=j + 1)
            else:  # Off-diagonal, plot scatterplots
                scatter = go.Scattergl(
                    x=df[var2],
                    y=df[var1],
                    mode="markers",
                    marker=dict(size=3, opacity=0.6, color=color),
                    name=f"{var1} vs {var2}",
                )
                fig.add_trace(scatter, row=i + 1, col=j + 1)

                # Update y-axes range for scatterplots based on y data
                y_range = [df[var1].min(), df[var1].max()]
                fig.update_yaxes(range=y_range, row=i + 1, col=j + 1)

            # Update axis labels
            if i == num_vars - 1:
                fig.update_xaxes(title_text=var2, row=i + 1, col=j + 1)
            if j == 0:
                fig.update_yaxes(title_text=var1, row=i + 1, col=j + 1)

    fig.update_layout(height=1100, width=2500, title_text=title, title_font_size=20)
    return fig
