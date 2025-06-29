# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from stats_calculator import get_primary_peak_hours, compute_hdi
from typing import List, Tuple


def filter_constant_plateaus(data: List[Tuple[pd.Timestamp, float]]) -> List[Tuple[pd.Timestamp, float]]:
    """
    Élimine les points redondants à l’intérieur d’un palier constant.
    data est une liste triée [(timestamp, valeur), …].
    On conserve :
      - le tout premier point du palier,
      - le tout dernier point du palier,
      - puis, dès que la valeur change, on réitère ces deux points.
    """
    if not data:
        return []

    filtered = []
    plateau_start_idx = 0
    filtered.append(data[0])  # toujours conserver le premier point global

    for i in range(1, len(data)):
        if data[i][1] != data[i - 1][1]:
            plateau_end_idx = i - 1
            if plateau_start_idx != plateau_end_idx:
                filtered.append(data[plateau_end_idx])
            filtered.append(data[i])
            plateau_start_idx = i

    last_idx = len(data) - 1
    if plateau_start_idx != last_idx:
        filtered.append(data[last_idx])

    return filtered


def create_fig_main(df, daily_stats, global_min, global_max, colors):
    """
    Figure principale :
    - la série brute “Mesures” est allégée aux seuls débuts/fins de palier constant,
    - les plateaux min/max journaliers sont calculés via ``compute_daily_extrema_timestamps``
      (appelé dans ``calculate_daily_stats``), garantissant qu’aucun palier traversant
      n’est pris en compte.
    """

    fig = go.Figure()

    # Correction 1 : Réduire les points affichés aux seuls changements de valeur (début/fin de plateau)
    if not df.empty:
        if len(df) > 1:
            changes = df['inch'].values[1:] != df['inch'].values[:-1]
            mask1 = np.concatenate(([True], changes))
            mask2 = np.concatenate((changes, [True]))
            mask_points = mask1 | mask2
        else:
            mask_points = np.array([True], dtype=bool)
        df_points = df.iloc[mask_points]
    else:
        df_points = df

    # Nuage de points des mesures (points de changement, transparence pour densité)
    fig.add_trace(go.Scatter(
        x=df_points['timestamp'], y=df_points['inch'],
        mode='markers',
        marker=dict(color='black', size=5, opacity=0.1),
        name='Mesures',
        customdata=np.stack([df_points['mm']], axis=-1),
        hovertemplate='Time: %{x}<br>Inclinaison: %{y:.3f} inch<br>%{customdata[0]:.1f} mm'
    ))

    # Correction 2 : les valeurs min/max quotidiennes sont désormais calculées
    # uniquement via ``compute_daily_extrema_timestamps`` dans
    # ``calculate_daily_stats``. On se contente donc d'utiliser les colonnes
    # 'min' et 'max' déjà présentes dans ``daily_stats``.
    daily_stats = daily_stats.copy()

    # Lignes horizontales pour min, max, mean, median de chaque jour + trace de ces stats
    for stat in ['min', 'max', 'mean', 'median']:
        for _, row in daily_stats.iterrows():
            if not np.isnan(row[stat]):
                fig.add_shape(
                    type="line",
                    x0=row['day_start'], x1=row['day_end'],
                    y0=row[stat], y1=row[stat],
                    line=dict(color=colors[stat], width=2, dash='solid'),
                    opacity=0.8, xref="x", yref="y"
                )
        fig.add_trace(go.Scatter(
            x=daily_stats['noon'], y=daily_stats[stat],
            mode='lines+markers',
            marker=dict(color=colors[stat]),
            line=dict(color=colors[stat]),
            name=stat.capitalize(),
            hovertemplate=f"{stat.capitalize()}<br>%{{y:.3f}} inch"
        ))
    # Intervalle de confiance (vert si normal assumé, violet sinon)
    for _, row in daily_stats.iterrows():

        if (
                row.get('normal', False)  # Student‑t
                and not np.isnan(row.get('ci_lower'))
                and not np.isnan(row.get('ci_upper'))
        ):
            fig.add_shape(
                type="rect",
                x0=row['day_start'], x1=row['day_end'],
                y0=row['ci_lower'], y1=row['ci_upper'],
                fillcolor='rgba(0,128,0,0.20)',  # vert
                line=dict(width=0), xref="x", yref="y"
            )

        elif (
                (not row.get('normal', True))  # bootstrap
                and not np.isnan(row.get('ci_lower_np'))
                and not np.isnan(row.get('ci_upper_np'))
        ):
            fig.add_shape(
                type="rect",
                x0=row['day_start'], x1=row['day_end'],
                y0=row['ci_lower_np'], y1=row['ci_upper_np'],
                fillcolor='rgba(128,0,128,0.20)',  # violet
                line=dict(width=0), xref="x", yref="y"
            )

    # Lignes pointillées pour global min et global max
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(), x1=df['timestamp'].max(),
        y0=global_max, y1=global_max,
        xref="x", yref="y",
        line=dict(color=colors['max'], dash='dot', width=2),
        opacity=0.8, layer="above"
    )
    fig.add_shape(
        type="line",
        x0=df['timestamp'].min(), x1=df['timestamp'].max(),
        y0=global_min, y1=global_min,
        xref="x", yref="y",
        line=dict(color=colors['min'], dash='dot', width=2),
        opacity=0.8, layer="above"
    )
    # Annotations pour les écarts en mm (différences quotidienne et vs global)
    for _, row in daily_stats.iterrows():
        if not np.isnan(row.get('diff_mm')):
            fig.add_annotation(
                x=row['noon'],
                y=row['mean'] + 0.0035,
                text=f"Δ = {row['diff_mm']:.1f} mm",
                showarrow=False,
                font=dict(color=colors['mean'], size=12),
                xref="x", yref="y"
            )
        if not np.isnan(row.get('diff_global_max_mm')):
            fig.add_annotation(
                x=row['noon'],
                y=row['max'] + 0.0035,
                text=f"Δ max = {row['diff_global_max_mm']:.1f} mm",
                showarrow=False,
                font=dict(color=colors['max'], size=12),
                xref="x", yref="y"
            )
        if not np.isnan(row.get('diff_global_min_mm')):
            fig.add_annotation(
                x=row['noon'],
                y=row['min'] - 0.0035,
                text=f"Δ min = {row['diff_global_min_mm']:.1f} mm",
                showarrow=False,
                font=dict(color=colors['min'], size=12),
                xref="x", yref="y"
            )
    fig.update_layout(
        title="Figure Principale - Évolution de l'inclinaison",
        xaxis_title="Temps",
        yaxis_title="Inclinaison (inch)",
        hovermode="closest"
    )
    return fig


def create_fig_values(daily_extrema_df, nbinsy_values, colors):
    """
    Crée la figure des distributions des valeurs extrêmes (onglet 2), en se basant
    EXCLUSIVEMENT sur les extrema journaliers provenant de `daily_extrema_df` (non sur
    l'ancien `daily_stats`). daily_extrema_df doit comporter au moins ces colonnes :
        - 'val_max' : float, valeur du maximum absolu de chaque jour (hors 24 h)
        - 'val_min' : float, valeur du minimum absolu de chaque jour (hors 0 h)
    Les autres éléments (clusters, histogrammes, KDE) sont conservés tels quels.
    """

    import numpy as np
    import plotly.graph_objs as go
    from stats_calculator import compute_value_clusters

    # 1) On récupère directement valeurs max et min corrigées (hors 0 h/24 h)
    max_vals = daily_extrema_df['val_max']
    min_vals = daily_extrema_df['val_min']

    # 2) Calcul de la largeur de bin pour convertir densité KDE en probabilité par bin
    value_min = float(min_vals.min())
    value_max = float(max_vals.max())
    bin_width = (value_max - value_min) / nbinsy_values if nbinsy_values > 0 else 1.0

    fig = go.Figure()

    # 3) Bandes de clusters (valeurs extrêmes)
    #    On recrée un DataFrame minimal { 'min':…, 'max':… } pour que compute_value_clusters
    #    fonctionne sans toucher à son code :
    df_for_clusters = {
        'min': min_vals,
        'max': max_vals
    }
    window_half_width = 0.1
    clusters = compute_value_clusters(df_for_clusters, window_half_width)
    # Cluster MAX (bordeaux)
    fig.add_shape(
        type="rect", xref="paper", yref="y",
        x0=0, x1=1,
        y0=clusters['center_max'] - clusters['margin_max'],
        y1=clusters['center_max'] + clusters['margin_max'],
        fillcolor='rgba(128,0,32,0.2)',
        line_width=0, layer="below"
    )
    # Cluster MIN (gris-bleu)
    fig.add_shape(
        type="rect", xref="paper", yref="y",
        x0=0, x1=1,
        y0=clusters['center_min'] - clusters['margin_min'],
        y1=clusters['center_min'] + clusters['margin_min'],
        fillcolor='rgba(119,136,184,0.2)',
        line_width=0, layer="below"
    )
    # Légende « Cluster MAX »
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(128,0,32,0.2)', size=10),
        showlegend=True,
        name='Cluster MAX'
    ))
    # Légende « Cluster MIN »
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(119,136,184,0.2)', size=10),
        showlegend=True,
        name='Cluster MIN'
    ))

    # 4) Histogrammes des valeurs MAX et MIN
    fig.add_trace(go.Histogram(
        y=max_vals,
        nbinsy=nbinsy_values,
        orientation='h',
        histnorm='probability',
        marker_color='rgba(255,0,0,0.8)',
        name='MAX',
        ybins=dict(size=bin_width),
        hovertemplate="Valeur: %{y:.3f} inch<br>Probabilité: %{x:.3f}"
    ))
    fig.add_trace(go.Histogram(
        y=min_vals,
        nbinsy=nbinsy_values,
        orientation='h',
        histnorm='probability',
        marker_color='rgba(0,0,255,0.8)',
        name='MIN',
        ybins=dict(size=bin_width),
        hovertemplate="Valeur: %{y:.3f} inch<br>Probabilité: %{x:.3f}"
    ))

    # 5) Layout initial
    fig.update_layout(
        barmode='overlay',
        yaxis_title="Valeur (inch)",
        xaxis_title="Probabilité",
        title="Distributions des valeurs extrêmes (inch) avec KDE",
        hovermode="closest"
    )
    fig.update_traces(opacity=0.7)

    # 6) Ajout des KDE (courbes de densité lissées) si échantillon > 1
    if len(max_vals) > 1:
        from scipy.stats import gaussian_kde
        kde_max = gaussian_kde(max_vals)
        kde_min = gaussian_kde(min_vals)
        y_range = np.linspace(value_min, value_max, 200)
        # Mise à l’échelle des densités pour obtenir une probabilité par bin
        kde_max_vals = kde_max(y_range) * bin_width
        kde_min_vals = kde_min(y_range) * bin_width
        fig.add_trace(go.Scatter(
            x=kde_max_vals, y=y_range, mode='lines',
            line=dict(color='rgba(255,0,0,0.5)'),
            name='KDE MAX'
        ))
        fig.add_trace(go.Scatter(
            x=kde_min_vals, y=y_range, mode='lines',
            line=dict(color='rgba(0,0,255,0.5)'),
            name='KDE MIN'
        ))

    return fig


def create_fig_hours(min_times, max_times, nbinsx_hours):
    """
    Crée la figure des heures des extrema journaliers (onglet 3), en utilisant
    exclusivement les timestamps corrigés (hors 0 h/24 h).
    min_times et max_times doivent être des np.ndarray de float (heures décimales),
    afin que get_primary_peak_hours puisse y appliquer np.isnan() sans erreur.
    """

    from stats_calculator import get_primary_peak_hours

    fig = go.Figure()

    # 1) Conversion éventuelle de min_times/max_times en heures décimales
    #    Si l'argument est déjà un np.ndarray, on le renvoie tel quel.
    def to_decimal_hours(arr):
        if isinstance(arr, np.ndarray):
            return arr
        # Si c'est une liste de floats, on la convertit en np.array
        if isinstance(arr, list) and isinstance(arr[0], float):
            return np.array(arr, dtype=float)
        # Sinon, on suppose que ce sont des pd.Timestamp
        return np.array([t.hour + t.minute / 60 + t.second / 3600 for t in arr], dtype=float)

    # Convertir en np.ndarray (heures décimales) si besoin
    hrs_min = to_decimal_hours(min_times)
    hrs_max = to_decimal_hours(max_times)

    x_range_hours = np.linspace(0, 24, 200)
    window_half_width = 3.0

    # 2) Bandes de clusters en arrière-plan (heures des extrêmes)
    # Cluster MAX (bordeaux)
    cluster_max = get_primary_peak_hours(hrs_max, window_half_width=window_half_width)
    if len(cluster_max) > 0:
        center_max = float(np.mean(cluster_max))
        margin = window_half_width / 2
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=center_max - margin, x1=center_max + margin,
            y0=0, y1=1,
            fillcolor='rgba(128,0,32,0.2)', line_width=0,
            name="MAX IC's cluster", showlegend=True
        )

    # Cluster MIN (gris-bleu)
    cluster_min = get_primary_peak_hours(hrs_min, window_half_width=window_half_width)
    if len(cluster_min) > 0:
        center_min = float(np.mean(cluster_min))
        margin = window_half_width / 2
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=center_min - margin, x1=center_min + margin,
            y0=0, y1=1,
            fillcolor='rgba(119,136,184,0.2)', line_width=0,
            name="MIN IC's cluster", showlegend=True
        )

    # 3) Histogrammes des heures MIN et MAX (densité)
    fig.add_trace(go.Histogram(
        x=hrs_min[~np.isnan(hrs_min)],
        nbinsx=nbinsx_hours,
        histnorm='probability density',
        marker_color='rgba(0,0,255,0.8)',
        name="Heures MIN",
        hovertemplate="Heure: %{x:.2f} h<br>Densité: %{y:.3f}"
    ))
    fig.add_trace(go.Histogram(
        x=hrs_max[~np.isnan(hrs_max)],
        nbinsx=nbinsx_hours,
        histnorm='probability density',
        marker_color='rgba(255,0,0,0.8)',
        name="Heures MAX",
        hovertemplate="Heure: %{x:.2f} h<br>Densité: %{y:.3f}"
    ))

    fig.update_layout(
        barmode='overlay',
        xaxis_title="Heure (h)",
        yaxis_title="Densité",
        title="Distributions des heures des extrêmes (h) avec KDE",
        hovermode="closest",
        xaxis=dict(range=[0, 24])
    )
    fig.update_traces(opacity=0.7)

    # 4) Ajout des KDE si données suffisantes
    from scipy.stats import gaussian_kde
    valid_min = hrs_min[~np.isnan(hrs_min)]
    valid_max = hrs_max[~np.isnan(hrs_max)]
    if len(valid_min) > 1 and len(valid_max) > 1:
        kde_hours_min = gaussian_kde(valid_min)
        kde_hours_max = gaussian_kde(valid_max)
        kde_hours_min_vals = kde_hours_min(x_range_hours)
        kde_hours_max_vals = kde_hours_max(x_range_hours)
        fig.add_trace(go.Scatter(
            x=x_range_hours, y=kde_hours_min_vals, mode='lines',
            line=dict(color='rgba(0,0,255,0.5)'),
            name='KDE Heures MIN'
        ))
        fig.add_trace(go.Scatter(
            x=x_range_hours, y=kde_hours_max_vals, mode='lines',
            line=dict(color='rgba(255,0,0,0.5)'),
            name='KDE Heures MAX'
        ))

    return fig


def create_fig_jour(
        data,
        label,
        max_half,  # inutilisé
        min_half,  # inutilisé
        annot_max,
        annot_min,
        center_max_time,
        margin_max_time,
        center_min_time,
        margin_min_time,
        colors,
        format_half_hour_func,  # inutilisée dans la figure finale
        base_data: pd.DataFrame = None
):
    """
    Construit la figure pour "Jour Moyen" ou "Jour Médian" (variation intra-journalière).
    """
    # Identifie s'il s'agit du jour moyen ou médian
    if "moyen" in label.lower():
        color_key = "mean"
        axis_title = "Inclinaison moyenne (inch)"
        hover_label = "Moyenne"
        full_title = "Jour Moyen - Moyenne par demi-heure"
    elif "médian" in label.lower():
        color_key = "median"
        axis_title = "Inclinaison médiane (inch)"
        hover_label = "Médiane"
        full_title = "Jour Médian - Médiane par demi-heure"
    else:
        color_key = label.lower()
        axis_title = "Inclinaison (inch)"
        hover_label = label
        full_title = label + " - par demi-heure"
    fig = go.Figure()
    # Courbe en marches (palier) pour l'évolution sur 24h
    fig.add_trace(go.Scatter(
        x=data['half_hour'],
        y=data['inch'],
        mode='lines',
        line_shape='hv',
        name=label,
        line=dict(color=colors[color_key]),
        hovertemplate="Période %{x:.1f} h<br>" + hover_label + ": %{y:.3f} inch",
        showlegend=False
    ))
    fig.update_layout(
        title=full_title,
        xaxis_title="Heure (h)",
        yaxis_title=axis_title,
        xaxis=dict(range=[0, 24]),
        margin=dict(t=120),
        showlegend=False
    )
    # pour le jour médian seulement : barplot différence moyenne–médiane
    if "médian" in label.lower() and base_data is not None:
        diff = base_data['inch'] - data['inch']
        fig.add_trace(go.Bar(
            x=data['half_hour'],
            y=diff,
            marker=dict(color='rgba(200,200,200,0.5)'),
            yaxis='y2',
            showlegend=False
        ))
        fig.update_layout(
            yaxis2=dict(
                title='moyenne – médiane',
                overlaying='y',
                side='right',
                showticklabels=True
            )
        )
        # annotation sans légende
        fig.add_annotation(
            x=0, y=1.08, xref='paper', yref='paper',
            text='moyenne – médiane',
            showarrow=False,
            font=dict(color='lightgrey', size=12)
        )
    # Deux annotations séparées en haut (pour jour moyen uniquement)
    if "moyen" in label.lower():
        fig.add_annotation(
            x=0, y=1.12, xref="paper", yref="paper",
            text=annot_max,
            showarrow=False, align="left",
            font=dict(color='rgba(128,0,32,0.8)', size=12)
        )
        fig.add_annotation(
            x=0, y=1.08, xref="paper", yref="paper",
            text=annot_min,
            showarrow=False, align="left",
            font=dict(color='rgba(119,136,184,0.8)', size=12)
        )
    # Bandes IC sur les heures de max (rouge) et min (bleu) du cluster principal
    fig.add_shape(
        type="rect", xref="x", yref="paper",
        x0=center_max_time - margin_max_time,
        x1=center_max_time + margin_max_time,
        y0=0, y1=1,
        fillcolor=colors['max'], opacity=0.2, line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="paper",
        x0=center_min_time - margin_min_time,
        x1=center_min_time + margin_min_time,
        y0=0, y1=1,
        fillcolor=colors['min'], opacity=0.2, line_width=0
    )
    return fig


def create_fig_hours_law_comparison(
        emp_max: np.ndarray,
        emp_min: np.ndarray,
        law_max: str,
        params_max: tuple,
        law_min: str,
        params_min: tuple,
        nbinsx: int,
        title: str
):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import truncnorm, logistic, vonmises, weibull_min
    # Map des lois utilisées
    dist_map = {
        "truncnorm": truncnorm,
        "logistic": logistic,
        "vonmises": vonmises,
        "weibull_min": weibull_min
    }
    dist_max = dist_map[law_max]
    dist_min = dist_map[law_min]
    # Histogrammes empiriques (densité) pour emp_max / emp_min
    hist_max, bin_edges = np.histogram(emp_max[~np.isnan(emp_max)], bins=nbinsx, range=(0, 24), density=True)
    hist_min, _ = np.histogram(emp_min[~np.isnan(emp_min)], bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    # Densités théoriques correspondantes (par différence de CDF pour chaque bin)
    cdf_e_max = dist_max.cdf(bin_edges, *params_max)
    theo_max = np.diff(cdf_e_max) / bin_width
    cdf_e_min = dist_min.cdf(bin_edges, *params_min)
    theo_min = np.diff(cdf_e_min) / bin_width
    # Différences empirique – théorique
    diff_max = hist_max - theo_max
    diff_min = hist_min - theo_min
    # Tracé figure
    fig = go.Figure()
    # Barres de Δ densité
    fig.add_trace(go.Bar(
        x=bin_centers, y=diff_max,
        name="Δ MAX",
        marker_color='rgba(255,0,0,0.6)', opacity=0.6
    ))
    fig.add_trace(go.Bar(
        x=bin_centers, y=diff_min,
        name="Δ MIN",
        marker_color='rgba(0,0,255,0.6)', opacity=0.6
    ))
    # Courbes PDF théoriques sur [0,24]
    xg = np.linspace(0, 24, 200)
    fig.add_trace(go.Scatter(
        x=xg, y=dist_max.pdf(xg, *params_max),
        mode='lines', name=f"{law_max} PDF",
        line=dict(color='rgba(128,0,32,0.6)')
    ))
    fig.add_trace(go.Scatter(
        x=xg, y=dist_min.pdf(xg, *params_min),
        mode='lines', name=f"{law_min} PDF",
        line=dict(color='rgba(119,136,184,0.6)')
    ))
    fig.update_layout(
        barmode='overlay',
        title=title,
        xaxis_title="Heure (h)",
        yaxis_title="Δ densité (empirique – théorique)",
        hovermode="closest"
    )
    return fig


def create_fig_qq_dual_laws(
        emp_max: np.ndarray,
        emp_min: np.ndarray,
        law_gmax: str, params_gmax: tuple,
        law_gmin: str, params_gmin: tuple,
        law_cmax: str, params_cmax: tuple,
        law_cmin: str, params_cmin: tuple,
        title: str
):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import truncnorm, logistic, vonmises, weibull_min
    # Map des lois
    dist_map = {
        "truncnorm": truncnorm,
        "logistic": logistic,
        "vonmises": vonmises,
        "weibull_min": weibull_min
    }
    gmax = dist_map[law_gmax]
    gmin = dist_map[law_gmin]
    cmax = dist_map[law_cmax]
    cmin = dist_map[law_cmin]
    fig = go.Figure()
    # 1) QQ-plot Global MAX (rouge)
    d_max = np.sort(emp_max[~np.isnan(emp_max)])
    if len(d_max) >= 2:
        p = np.linspace(0.01, 0.99, len(d_max))
        xg = gmax.ppf(p, *params_gmax)
        fig.add_trace(go.Scatter(
            x=xg, y=d_max, mode='markers',
            name="QQ MAX global",
            marker=dict(color='rgba(255,0,0,0.6)')
        ))
    # 2) QQ-plot Global MIN (bleu)
    d_min = np.sort(emp_min[~np.isnan(emp_min)])
    if len(d_min) >= 2:
        p = np.linspace(0.01, 0.99, len(d_min))
        xg = gmin.ppf(p, *params_gmin)
        fig.add_trace(go.Scatter(
            x=xg, y=d_min, mode='markers',
            name="QQ MIN global",
            marker=dict(color='rgba(0,0,255,0.6)')
        ))
    # 3) QQ-plot Cluster MAX (bordeaux)
    if len(d_max) >= 2:
        p = np.linspace(0.01, 0.99, len(d_max))
        xc = cmax.ppf(p, *params_cmax)
        fig.add_trace(go.Scatter(
            x=xc, y=d_max, mode='markers',
            name="QQ MAX cluster",
            marker=dict(color='rgba(128,0,32,0.4)')
        ))
    # 4) QQ-plot Cluster MIN (gris-bleu)
    if len(d_min) >= 2:
        p = np.linspace(0.01, 0.99, len(d_min))
        xc = cmin.ppf(p, *params_cmin)
        fig.add_trace(go.Scatter(
            x=xc, y=d_min, mode='markers',
            name="QQ MIN cluster",
            marker=dict(color='rgba(119,136,184,0.6)')
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Quantiles Théoriques",
        yaxis_title="Quantiles Empiriques",
        hovermode="closest"
    )
    return fig


def create_fig_value_step(
        df_agg,
        label: str,
        color: str,
        bin_col: str = "inch_bin",
        hour_col: str = "hour"
):
    """
    Construit une figure en escaliers (verticale) représentant la valeur atteinte en fonction de l'heure.
    - df_agg : DataFrame avec les colonnes `hour` (ou `hour_col`) et `inch_bin` (ou `bin_col`).
    - label  : Titre à afficher sur la figure.
    - color  : Couleur de la ligne (format CSS ou rgba).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_agg[hour_col],
        y=df_agg[bin_col],
        mode='lines',
        line_shape='vh',
        line=dict(color=color, width=2),
        name=label,
        hovertemplate=f"{label}<br>Heure moyenne: %{{x:.2f}} h<br>Valeur: %{{y:.2f}} inch"
    ))
    fig.update_layout(
        title=label,
        xaxis_title="Heure (h)",
        yaxis_title="Valeur (inch)",
        xaxis=dict(range=[0, 24]),
        hovermode="closest"
    )
    return fig


def create_fig_values_timeseries(daily_stats, colors):
    fig = go.Figure()
    ds = daily_stats.sort_values('day_start')

    # Remplissage gris entre min et max quotidiens
    fig.add_trace(go.Scatter(
        x=ds['day_start'],
        y=ds['max'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=ds['day_start'],
        y=ds['min'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Courbes pour min, mean, median, max au fil des jours
    for stat in ['min', 'mean', 'median', 'max']:
        fig.add_trace(go.Scatter(
            x=ds['day_start'],
            y=ds[stat],
            mode='lines',
            line=dict(color=colors[stat]),
            name=stat.capitalize()
        ))

    fig.update_layout(
        title="Évolution journalière des valeurs extrêmes (inch)",
        xaxis_title="Date",
        yaxis_title="Valeur (inch)",
        hovermode="closest"
    )
    return fig


def create_fig_diff_min_max_timeseries(daily_stats, gray_color):
    # Calcul de la différence quotidienne Max–Min
    diff = daily_stats['max'] - daily_stats['min']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stats['day_start'],
        y=diff,
        mode='lines+markers',
        marker=dict(color=gray_color),
        line=dict(color=gray_color),
        name='Δ Max–Min'
    ))
    fig.update_layout(
        title="Écart quotidien entre Max et Min",
        xaxis_title="Date",
        yaxis_title="Δ (inch)",
        hovermode="closest"
    )
    return fig


def create_fig_values_law_comparison(
        emp_max: np.ndarray,
        emp_min: np.ndarray,
        law_max: str,
        params_max: tuple,
        law_min: str,
        params_min: tuple,
        nbins: int,
        title: str
):
    """
    Comme create_fig_hours_law_comparison, mais pour les valeurs max/min journalières.
    Trace les Δ de probabilités par bin (empirique – théorique) et les PDF pour référence.
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import norm, logistic, weibull_min, truncnorm
    # Map des lois
    dist_map = {
        'norm': norm,
        'logistic': logistic,
        'weibull_min': weibull_min,
        'truncnorm': truncnorm
    }
    # Bornes des bins
    all_vals = np.concatenate([emp_min, emp_max])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    bin_edges = np.linspace(vmin, vmax, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    # Histogrammes empiriques normalisés en probabilités
    counts_max, _ = np.histogram(emp_max[~np.isnan(emp_max)], bins=bin_edges, density=False)
    counts_min, _ = np.histogram(emp_min[~np.isnan(emp_min)], bins=bin_edges, density=False)
    hist_max = counts_max / counts_max.sum() if counts_max.sum() > 0 else np.zeros_like(counts_max)
    hist_min = counts_min / counts_min.sum() if counts_min.sum() > 0 else np.zeros_like(counts_min)
    # Probabilités théoriques par bin via CDF
    cdf_max = dist_map[law_max].cdf(bin_edges, *params_max)
    cdf_min = dist_map[law_min].cdf(bin_edges, *params_min)
    theo_max = np.diff(cdf_max)
    theo_min = np.diff(cdf_min)
    # Δ probabilités
    diff_max = hist_max - theo_max
    diff_min = hist_min - theo_min
    # PDF continues mises à l’échelle en probas par bin
    xg = np.linspace(vmin, vmax, 200)
    pdf_max = dist_map[law_max].pdf(xg, *params_max) * bin_width
    pdf_min = dist_map[law_min].pdf(xg, *params_min) * bin_width
    # Tracé
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=diff_max, y=bin_centers, orientation='h',
        name='Δ MAX', marker_color='rgba(255,0,0,0.6)', opacity=0.6
    ))
    fig.add_trace(go.Bar(
        x=diff_min, y=bin_centers, orientation='h',
        name='Δ MIN', marker_color='rgba(0,0,255,0.6)', opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=pdf_max, y=xg, mode='lines',
        name=f'{law_max} PDF', line=dict(color='rgba(128,0,32,0.6)')
    ))
    fig.add_trace(go.Scatter(
        x=pdf_min, y=xg, mode='lines',
        name=f'{law_min} PDF', line=dict(color='rgba(119,136,184,0.6)')
    ))
    fig.update_layout(
        barmode='overlay',
        title=title,
        xaxis_title='Δ probabilité (emp – théor)',
        yaxis_title='Valeur (inch)',
        hovermode='closest'
    )
    return fig


def create_fig_qq_values(
        emp_max: np.ndarray,
        emp_min: np.ndarray,
        cluster_max: np.ndarray,
        cluster_min: np.ndarray,
        law_max_global: str,
        params_max_global: tuple,
        law_min_global: str,
        params_min_global: tuple,
        law_max_cluster: str,
        params_max_cluster: tuple,
        law_min_cluster: str,
        params_min_cluster: tuple,
        title: str
):
    """
    Génère 4 QQ-plots pour les valeurs max/min :
      1) max globaux vs loi_max_global
      2) min globaux vs loi_min_global
      3) max du cluster principal vs loi_max_cluster
      4) min du cluster principal vs loi_min_cluster
    Inclut des annotations détaillées sur les clusters (paramètres, IC95, mode).
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import truncnorm, logistic, weibull_min, norm
    # Mapping des lois
    dist_map = {
        'truncnorm': truncnorm,
        'logistic': logistic,
        'weibull_min': weibull_min,
        'norm': norm
    }
    # Formatage des paramètres en chaînes
    params_max_global_str = '[' + ', '.join(f"{p:.2f}" for p in params_max_global) + ']'
    params_min_global_str = '[' + ', '.join(f"{p:.2f}" for p in params_min_global) + ']'
    params_max_cluster_str = '[' + ', '.join(f"{p:.2f}" for p in params_max_cluster) + ']'
    params_min_cluster_str = '[' + ', '.join(f"{p:.2f}" for p in params_min_cluster) + ']'
    # Calcul des annotations (IC95 et mode) pour clusters
    hdi_max = compute_hdi(cluster_max)
    annot_max = (
        f"Cluster MAX → {law_max_cluster} params={params_max_cluster_str}, "
        f"IC95=[{hdi_max['lo']:.2f}, {hdi_max['hi']:.2f}], mode={hdi_max['mode']:.2f}"
    )
    hdi_min = compute_hdi(cluster_min)
    annot_min = (
        f"Cluster MIN → {law_min_cluster} params={params_min_cluster_str}, "
        f"IC95=[{hdi_min['lo']:.2f}, {hdi_min['hi']:.2f}], mode={hdi_min['mode']:.2f}"
    )
    fig = go.Figure()
    # Annotations des lois de cluster
    fig.add_annotation(
        x=0, y=1.10, xref='paper', yref='paper',
        text=annot_max,
        showarrow=False, align='left',
        font=dict(color='rgba(128,0,32,0.8)', size=12)
    )
    fig.add_annotation(
        x=0, y=1.06, xref='paper', yref='paper',
        text=annot_min,
        showarrow=False, align='left',
        font=dict(color='rgba(119,136,184,0.8)', size=12)
    )
    # QQ-plot Global MAX
    data_gmax = np.sort(emp_max[~np.isnan(emp_max)])
    if len(data_gmax) >= 3:
        probs = np.linspace(0.01, 0.99, len(data_gmax))
        theo_gmax = dist_map[law_max_global].ppf(probs, *params_max_global)
        fig.add_trace(go.Scatter(
            x=theo_gmax, y=data_gmax,
            mode='markers', name='QQ MAX global',
            marker=dict(color='rgba(255,0,0,0.6)')
        ))
    # QQ-plot Global MIN
    data_gmin = np.sort(emp_min[~np.isnan(emp_min)])
    if len(data_gmin) >= 3:
        probs = np.linspace(0.01, 0.99, len(data_gmin))
        theo_gmin = dist_map[law_min_global].ppf(probs, *params_min_global)
        fig.add_trace(go.Scatter(
            x=theo_gmin, y=data_gmin,
            mode='markers', name='QQ MIN global',
            marker=dict(color='rgba(0,0,255,0.6)')
        ))
    # QQ-plot Cluster MAX
    data_cmax = np.sort(cluster_max[~np.isnan(cluster_max)])
    if len(data_cmax) >= 3:
        probs = np.linspace(0.01, 0.99, len(data_cmax))
        theo_cmax = dist_map[law_max_cluster].ppf(probs, *params_max_cluster)
        fig.add_trace(go.Scatter(
            x=theo_cmax, y=data_cmax,
            mode='markers', name='QQ MAX cluster',
            marker=dict(color='rgba(128,0,32,0.4)')
        ))
    # QQ-plot Cluster MIN
    data_cmin = np.sort(cluster_min[~np.isnan(cluster_min)])
    if len(data_cmin) >= 3:
        probs = np.linspace(0.01, 0.99, len(data_cmin))
        theo_cmin = dist_map[law_min_cluster].ppf(probs, *params_min_cluster)
        fig.add_trace(go.Scatter(
            x=theo_cmin, y=data_cmin,
            mode='markers', name='QQ MIN cluster',
            marker=dict(color='rgba(119,136,184,0.6)')
        ))
    fig.update_layout(
        title=title,
        xaxis_title='Quantiles théoriques',
        yaxis_title='Quantiles empiriques (Valeurs inch)',
        hovermode='closest'
    )
    return fig


def create_fig_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, horizon_days: int):
    """
    Construit la figure de prévision sur `horizon_days` jours à partir du modèle Prophet.
    Affiche les données historiques et la projection avec intervalle de confiance à 95%.
    """
    # Données historiques (réelles)
    actual_time = df['timestamp']
    actual_inch = df['inch']
    actual_mm = df['mm'] if 'mm' in df.columns else df['inch'] * 25.4
    # Données de prévision (issues de Prophet)
    forecast_time = forecast_df['ds']
    yhat = forecast_df['yhat']
    yhat_lower = forecast_df['yhat_lower']
    yhat_upper = forecast_df['yhat_upper']
    yhat_mm = yhat * 25.4
    fig = go.Figure()
    # Trace des mesures historiques
    fig.add_trace(go.Scatter(
        x=actual_time, y=actual_inch,
        mode='lines',
        line=dict(color='black', width=1),
        name='Mesures',
        customdata=np.array(actual_mm).reshape(-1, 1),
        hovertemplate='Date: %{x}<br>Inclinaison: %{y:.3f} inch<br>%{customdata[0]:.1f} mm'
    ))
    # Intervalle de confiance 95% (bande entre yhat_lower et yhat_upper)
    fig.add_trace(go.Scatter(
        x=forecast_time, y=yhat_lower,
        mode='lines', line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_time, y=yhat_upper,
        mode='lines', line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty', fillcolor='rgba(255,0,255,0.2)',
        name='IC 95%', hoverinfo='skip'
    ))
    # Trace de la prévision (valeur moyenne prédite)
    fig.add_trace(go.Scatter(
        x=forecast_time, y=yhat,
        mode='lines',
        line=dict(color='magenta', width=2),
        name='Prévision',
        customdata=np.array(yhat_mm).reshape(-1, 1),
        hovertemplate='Date: %{x}<br>Prévision: %{y:.3f} inch<br>%{customdata[0]:.1f} mm'
    ))
    fig.update_layout(
        title=f"Prévisions sur {horizon_days} jours",
        xaxis_title="Date",
        yaxis_title="Inclinaison (inch)",
        hovermode="closest"
    )
    return fig
