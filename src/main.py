import logging

import dash
import plotly.io as pio
from dash import dcc, html

from analysis.models import (linear_regression, loess_regression,
                             model_fissures_with_explanatory_vars,
                             prepare_data, regression_comparison,
                             select_weekly_variables, train_models,
                             visualize_model_results)
from analysis.statistical_analysis import tests_statistiques
from data_processing.fissures_processing import chargement_donnees
from data_processing.meteo_processing import (add_moving_averages,
                                              add_temporal_derivatives,
                                              add_weekly_stats,
                                              calculate_water_content,
                                              clean_data,
                                              load_and_concat_excel_files,
                                              normalize_data,
                                              save_cleaned_data)
from visualization.fissures_visualization import (dataviz_evolution,
                                                  dataviz_forecast,
                                                  dataviz_old_new,
                                                  preprocessing_old_new)
from visualization.meteo_visualization import (plot_humidity, plot_light_uv,
                                               plot_moving_averages,
                                               plot_pairplot,
                                               plot_precipitation,
                                               plot_temperature_extremes,
                                               plot_weekly_statistics,
                                               plot_wind_speed_direction,
                                               visualize_normalized_boxplots)
from visualization.structure_visualization import (generate_boxplot_figure,
                                                   generate_building_plan,
                                                   generate_dual_axis_figure,
                                                   generate_scatterplot_grid,
                                                   return_df_paliers_combined)

logging.basicConfig(level=logging.DEBUG)


def prepare_meteo_data():
    """
    Prépare les données météorologiques, effectue les calculs nécessaires et retourne les DataFrames nettoyés et normalisés.
    """
    data_dir = "data/Meteo/"
    meteo_output_path = "data/Meteo/artifact/df_cleaned_with_stats.csv"

    # Chargement et traitement des données
    df_combined = load_and_concat_excel_files(data_dir)
    df_cleaned = clean_data(df_combined)
    df_cleaned = calculate_water_content(df_cleaned)
    df_cleaned = add_temporal_derivatives(df_cleaned)
    df_cleaned = add_moving_averages(df_cleaned)
    df_cleaned = add_weekly_stats(df_cleaned)

    # Sauvegarde des données nettoyées
    save_cleaned_data(df_cleaned, meteo_output_path)
    logging.info("Saved data")

    # Normalisation des données pour les visualisations
    columns_to_normalize = [
        col
        for col in df_cleaned.columns
        if not (
            col.endswith("MA")
            or col.startswith("CH1")
            or col.endswith(("mean", "median", "std", "skew", "kurtosis"))
        )
    ]
    df_normalized = normalize_data(df_cleaned[columns_to_normalize])
    logging.info("Normalized data")

    return df_cleaned, df_normalized


def generate_meteo_visualizations(df_cleaned, df_normalized):
    """
    Génère les visualisations pour les données météorologiques.
    """
    plotly_boxplots = visualize_normalized_boxplots(df_normalized)
    plotly_T = plot_moving_averages(df_cleaned)
    plotly_TmM = plot_temperature_extremes(df_cleaned)
    plotly_Hum = plot_humidity(df_cleaned)
    plotly_precip = plot_precipitation(df_cleaned)
    plotly_WindSpeed, plotly_WindDir = plot_wind_speed_direction(df_cleaned)
    plotly_LightUV = plot_light_uv(df_cleaned)

    variable_base_names = [
        "Indoor Tem(°C)",
        "Indoor Tem.Max(°C)",
        "Indoor Tem.Min(°C)",
        "Outdoor Tem(°C)",
        "Outdoor Tem.Max(°C)",
        "Outdoor Tem.Min(°C)",
        "Indoor Hum(%)",
        "Outdoor Hum(%)",
    ]
    plotly_weekly_list = plot_weekly_statistics(df_cleaned, variable_base_names)

    return {
        "boxplots": plotly_boxplots,
        "temperature": plotly_T,
        "temp_minmax": plotly_TmM,
        "humidity": plotly_Hum,
        "precipitation": plotly_precip,
        "wind_speed": plotly_WindSpeed,
        "wind_dir": plotly_WindDir,
        "light_uv": plotly_LightUV,
        "weekly_stats": plotly_weekly_list,
    }, plotly_weekly_list


def prepare_fissure_data():
    """
    Charge les données de fissures, effectue les tests statistiques et génère les visualisations associées.
    """
    fissures_path = "data/Fissures/"
    # Chargement des données de fissures et des données anciennes
    df_fissures, df_fissures_old = chargement_donnees(fissures_path)

    # Effectuer les tests statistiques sur les données récentes
    tests_statistiques(df_fissures)

    # Générer les visualisations
    plotly_fissures = dataviz_evolution(df_fissures, df_fissures_old)
    plotly_fissures_old_new = dataviz_old_new(df_fissures, df_fissures_old)
    (
        second_phase_data,
        third_phase_data,
        fourth_phase_data,
        fifth_phase_data,
        sixth_phase_data,
        plotly_loess,
    ) = loess_regression(df_fissures)
    regression_results_df, plotly_RLevol = linear_regression(df_fissures)
    plotly_LRFissure = regression_comparison(df_fissures)
    plotly_fissure_forecast = dataviz_forecast(df_fissures, df_fissures_old)

    return (
        df_fissures,
        df_fissures_old,
        {
            "fissures": plotly_fissures,
            "fissures_old_new": plotly_fissures_old_new,
            "loess": plotly_loess,
            "trend": plotly_RLevol,
            "reglin": plotly_LRFissure,
            "fissure_forecast": plotly_fissure_forecast,
        },
    )


def generate_modeling_results(df_cleaned, df_fissures, df_fissures_old):
    """
    Prépare les données pour la modélisation, entraîne les modèles (météo et structure),
    et retourne les visualisations des résultats.
    """
    # === Modélisation Météo ===

    # Préparation des données météo
    df_weekly = select_weekly_variables(df_cleaned)
    df_joined = prepare_data(df_weekly, df_fissures)

    # Extraction des caractéristiques et de la cible (pour la modélisation météo)
    X = df_joined.drop(["Variation Bureau", "Date", "Bureau", "Mur extérieur"], axis=1)
    y = df_joined["Variation Bureau"]
    delta_y = y.max() - y.min()

    # Entraînement des modèles météo
    fglm_pipeline, rf_pipeline, gb_pipeline, X_test, y_test = train_models(X, y)
    logging.info("Modèles entraînés avec succès pour la météo")

    # Visualisation des résultats des modèles Météo
    plotly_FIfglm, plotly_FIrf, plotly_FIgb = visualize_model_results(
        fglm_pipeline, rf_pipeline, gb_pipeline, X_test, y_test, delta_y
    )

    # === Modélisation Structure ===

    # Prétraitement des données de fissures pour la modélisation structurelle
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        paliers_old,
        _,
        paliers_new,
        df_paliers_old,
        df_paliers_new,
        _,
    ) = preprocessing_old_new(df_fissures, df_fissures_old)

    # Modélisation des fissures (structure)
    model_results_structure = model_fissures_with_explanatory_vars(
        df_paliers_old, df_paliers_new
    )

    # Retourner les résultats des modélisations Météo et Structure
    return {
        # Modélisation Météo
        "feat_import_rf": plotly_FIrf,
        "feat_import_gb": plotly_FIgb,
        "feat_import_lm": plotly_FIfglm,
        # Modélisation Structure
        "feat_import_structure_rf": model_results_structure["Random Forest"]["plot"],
        "feat_import_structure_gb": model_results_structure["Gradient Boosting"][
            "plot"
        ],
        "feat_import_structure_lm": model_results_structure["Ridge"]["plot"],
    }


def create_dashboard(
    app,
    meteo_figures,
    fissure_figures,
    modeling_figures,
    plotly_weekly_list,
    structure_figures,
    df_paliers_combined,
):
    """
    Crée le tableau de bord Dash avec les visualisations fournies.
    """
    app.layout = html.Div(
        [
            html.Div(
                [
                    # Titre et boutons radio pour la sélection des sections principales
                    html.Div(
                        [
                            html.H1(
                                "Analyse et Modélisation des Causes d'une Fissure",
                                style={
                                    "textAlign": "left",
                                    "color": "#2c3e50",
                                    "font-family": "Arial, sans-serif",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                    "paddingLeft": "40px",
                                },
                            ),
                            dcc.RadioItems(
                                id="main-tabs",
                                options=[
                                    {"label": "Fissure", "value": "fissure"},
                                    {
                                        "label": "Météo",
                                        "value": "meteo",
                                    },
                                    {
                                        "label": "Modélisation Météo",
                                        "value": "modeling",
                                    },
                                    {"label": "Structure", "value": "structure_data"},
                                    {
                                        "label": "Modélisation Structure",
                                        "value": "structure",
                                    },
                                ],
                                value="fissure",
                                labelStyle={
                                    "display": "inline-block",
                                    "margin": "0 80px",
                                    "font-family": "Arial, sans-serif",
                                    "fontSize": "20px",
                                },
                                style={
                                    "float": "right",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                },
                            ),
                        ],
                        style={
                            "width": "100%",
                            "padding": "0 10px",
                            "backgroundColor": "#ecf0f1",
                            "height": "80px",
                            "display": "flex",
                            "alignItems": "center",
                        },
                    )
                ]
            ),
            html.Div(id="tab-content", style={"padding": "0", "margin": "0"}),
        ],
        style={
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "backgroundColor": "#f2f2f2",
        },
    )

    # Légende plan du bâtiment
    legend_items = [
        html.Span(
            "Acrotère : ",
            style={
                "color": "black",
                "font-family": "DejaVu Sans",  # Default Matplotlib font
                "font-size": "20px",
                "font-weight": "normal",
                "vertical-align": "middle",
            },
        ),
        html.Span(
            "en noir, date de construction du bâtiment (1959)",
            style={"color": "black", "font-family": "DejaVu Sans", "font-size": "16px"},
        ),
        html.Br(),
        html.Span(
            "IPN : ",
            style={
                "color": "blue",
                "font-family": "DejaVu Sans",  # Default Matplotlib font
                "font-size": "20px",
                "font-weight": "normal",
                "vertical-align": "middle",
            },
        ),
        html.Span(
            "en bleu, mise en place à la date des travaux (2016)",
            style={"color": "black", "font-family": "DejaVu Sans", "font-size": "16px"},
        ),
    ]

    @app.callback(
        dash.dependencies.Output("tab-content", "children"),
        [dash.dependencies.Input("main-tabs", "value")],
    )
    def render_content(tab):
        if tab == "meteo":
            return dcc.Tabs(
                [
                    dcc.Tab(
                        label="Températures",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["temperature"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Temp. MinMax",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["temp_minmax"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Humidité",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["humidity"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Précipitations",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["precipitation"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Vent Vitesse",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["wind_speed"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Vent Direction",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["wind_dir"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Éclairement",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["light_uv"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Statistiques",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(
                                        figure=fig,
                                        style={
                                            "width": "100%",
                                            "height": "85vh",
                                            "margin-top": "20px",
                                        },
                                        config={"responsive": True},
                                    )
                                    for fig in plotly_weekly_list
                                ],
                                style={"height": "85vh", "overflowY": "auto"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Pairplots",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["pairplots"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Boxplots\n(traitement long)",
                        children=[
                            dcc.Graph(
                                figure=meteo_figures["boxplots"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                ],
                style={"backgroundColor": "#ecf0f1", "borderRadius": "5px"},
            )
        elif tab == "fissure":
            return dcc.Tabs(
                [
                    dcc.Tab(
                        label="Fissures : suivi",
                        children=[
                            dcc.Graph(
                                figure=fissure_figures["fissures"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Fissure Bureau : analyse",
                        children=[
                            dcc.Graph(
                                figure=fissure_figures["fissures_old_new"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Fissure Bureau : prévisions",
                        children=[
                            dcc.Graph(
                                figure=fissure_figures["fissure_forecast"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Tendance",
                        children=[
                            dcc.Graph(
                                figure=fissure_figures["trend"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="LOESS",
                        children=[
                            dcc.Graph(
                                figure=fissure_figures["loess"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    # dcc.Tab(
                    #     label="RegLin Fissure",
                    #     children=[
                    #         dcc.Graph(
                    #             figure=fissure_figures["reglin"],
                    #             style={
                    #                 "width": "100%",
                    #                 "height": "85vh",
                    #                 "margin-top": "20px",
                    #             },
                    #             config={"responsive": True},
                    #         )
                    #     ],
                    # ),
                ],
                style={"backgroundColor": "#ecf0f1", "borderRadius": "5px"},
            )
        elif tab == "modeling":
            return dcc.Tabs(
                [
                    dcc.Tab(
                        label="FeatImport RF",
                        children=[
                            dcc.Graph(
                                figure=modeling_figures["feat_import_rf"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="FeatImport GB",
                        children=[
                            dcc.Graph(
                                figure=modeling_figures["feat_import_gb"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="FeatImport LM",
                        children=[
                            dcc.Graph(
                                figure=modeling_figures["feat_import_lm"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                ],
                style={
                    "backgroundColor": "#ecf0f1",
                    "borderRadius": "5px",
                    "margin-top": "20px",
                },
            )
        elif tab == "structure_data":
            return dcc.Tabs(
                [
                    dcc.Tab(
                        label="Plan du bâtiment",
                        children=[
                            html.Div(
                                children=[
                                    html.Img(
                                        src=generate_building_plan(),
                                        style={
                                            "max-width": "100%",
                                            "height": "auto",
                                            "display": "block",
                                            "margin-left": "auto",
                                            "margin-right": "auto",
                                        },
                                    ),
                                    html.Div(
                                        children=legend_items,
                                        style={
                                            "textAlign": "center",
                                            "marginTop": "20px",
                                            "fontSize": "16px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flex-direction": "column",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    "height": "90vh",
                                },
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Boxplots et Scatterplot",
                        children=[
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        figure=structure_figures["boxplot_scatter"],
                                        style={"width": "100%", "height": "45vh"},
                                        config={"responsive": True},
                                    ),
                                    dcc.Graph(
                                        figure=structure_figures["dual_axis_scatter"],
                                        style={"width": "100%", "height": "45vh"},
                                        config={"responsive": True},
                                    ),
                                ]
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Grille de Scatterplots",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        "Réglage du seuil du coefficient de Pearson",
                                        style={
                                            "fontSize": 16,
                                            "font-family": "DejaVu Sans",
                                            "margin-left": "20px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            dcc.Slider(
                                                id="correlation-threshold-slider",
                                                min=0.5,
                                                max=1.0,
                                                step=0.01,
                                                value=0.80,
                                                marks={
                                                    i: str(i)
                                                    for i in [
                                                        0.5,
                                                        0.6,
                                                        0.7,
                                                        0.8,
                                                        0.9,
                                                        1.0,
                                                    ]
                                                },
                                                tooltip={"placement": "bottom"},
                                            )
                                        ],
                                        style={
                                            "margin-top": "30px",
                                            "margin-bottom": "20px",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="scatterplot-grid",
                                        config={"responsive": True},
                                    ),
                                ],
                                style={"width": "100%", "margin-top": "20px"},
                            )
                        ],
                    ),
                ]
            )
        elif tab == "structure":
            return dcc.Tabs(
                [
                    dcc.Tab(
                        label="FeatImport RF",
                        children=[
                            dcc.Graph(
                                figure=modeling_figures["feat_import_structure_rf"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="FeatImport GB",
                        children=[
                            dcc.Graph(
                                figure=modeling_figures["feat_import_structure_gb"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="FeatImport LM",
                        children=[
                            dcc.Graph(
                                figure=modeling_figures["feat_import_structure_lm"],
                                style={
                                    "width": "100%",
                                    "height": "85vh",
                                    "margin-top": "20px",
                                },
                                config={"responsive": True},
                            )
                        ],
                    ),
                ]
            )

    @app.callback(
        dash.dependencies.Output("scatterplot-grid", "figure"),
        [dash.dependencies.Input("correlation-threshold-slider", "value")],
    )
    def update_scatterplot_grid(threshold):
        return generate_scatterplot_grid(df_paliers_combined, threshold)

    return app


def setup_logging():
    """
    Configure le logging de l'application pour s'assurer que les messages
    de log sont formatés et enregistrés correctement.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def save_dashboard_as_html(
    meteo_figures, fissure_figures, modeling_figures, structure_figures, filename
):
    """
    Sauvegarde toutes les figures du dashboard dans un fichier HTML unique.
    :param meteo_figures: Figures de la section météo.
    :param fissure_figures: Figures de la section fissures.
    :param modeling_figures: Figures de la section modélisation.
    :param structure_figures: Figures de la section structure.
    :param filename: Nom du fichier de sortie HTML.
    """
    html_content = ""

    # Fonction interne pour gérer les figures ou listes de figures
    def add_figures_to_html(figures_dict):
        nonlocal html_content
        for fig_name, fig in figures_dict.items():
            if isinstance(fig, list):  # Si la figure est une liste
                for subfig in fig:
                    html_content += pio.to_html(
                        subfig, full_html=False
                    )  # Ajouter chaque sous-figure
            else:
                html_content += pio.to_html(
                    fig, full_html=False
                )  # Ajouter la figure unique

    # Sauvegarde des figures météo
    add_figures_to_html(meteo_figures)

    # Sauvegarde des figures fissures
    add_figures_to_html(fissure_figures)

    # Sauvegarde des figures modélisation
    add_figures_to_html(modeling_figures)

    # Sauvegarde des figures structure
    add_figures_to_html(structure_figures)

    # Sauvegarder tout dans un fichier HTML
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Le dashboard complet a été sauvegardé dans {filename}")


def main():
    """
    Point d'entrée principal du script. Prépare les données, génère les visualisations,
    et lance l'application Dash.
    """

    # Préparation des données météo
    df_cleaned, df_normalized = prepare_meteo_data()

    # Génération des visualisations météo
    meteo_figures, plotly_weekly_list = generate_meteo_visualizations(
        df_cleaned, df_normalized
    )

    # Sélection des colonnes pour le pairplot
    selected_columns = [
        "Outdoor Tem(°C)",  # 'Indoor Tem(°C)',
        "Outdoor Hum(%)",  # 'Indoor Hum(%)',
        "Wind speed(km/h)",  # 'Wind direction',
        "Rainfull(Day)(mm)",  # 'Rainfull(Hour)(mm)',
        "Pressure(hpa)",  # 'Indoor Water Content (g/m³)',
        "Outdoor Water Content (g/m³)",
    ]
    df_filtered = df_cleaned[selected_columns]
    meteo_figures["pairplots"] = plot_pairplot(df_filtered)

    # Préparation des données fissures et génération des visualisations
    df_fissures, df_fissures_old, fissure_figures = prepare_fissure_data()

    # Modélisation et génération des visualisations
    modeling_figures = generate_modeling_results(
        df_cleaned, df_fissures, df_fissures_old
    )

    # Import des données de structure
    df_paliers_combined = return_df_paliers_combined()

    # Générer les visualisations structurelles à partir de df_paliers_combined
    structure_figures = {
        "boxplot_scatter": generate_boxplot_figure(df_paliers_combined),
        "dual_axis_scatter": generate_dual_axis_figure(df_paliers_combined),
        "scatterplot_grid": generate_scatterplot_grid(df_paliers_combined),
    }

    # Sauvegarde des figures dans des fichiers HTML individuels
    def save_figures(figures, prefix):
        for name, figure in figures.items():
            if isinstance(figure, list):
                for i, fig in enumerate(figure):
                    pio.write_html(
                        fig, file=f"results/{prefix}_{name}_{i}.html", auto_open=False
                    )
            else:
                pio.write_html(
                    figure, file=f"results/{prefix}_{name}.html", auto_open=False
                )

    save_figures(meteo_figures, "meteo")
    save_figures(fissure_figures, "fissure")
    save_figures(modeling_figures, "modeling")
    save_figures(structure_figures, "structure")

    # Sauvegarde des figures de statistiques hebdomadaires
    for i, fig in enumerate(plotly_weekly_list):
        pio.write_html(fig, file=f"results/weekly_stats_{i}.html", auto_open=False)

    # Sauvegarde du dashboard complet dans un fichier HTML
    save_dashboard_as_html(
        meteo_figures,
        fissure_figures,
        modeling_figures,
        structure_figures,
        "dashboard_complet.html",
    )

    logging.info("Toutes les figures ont été sauvegardées dans le dossier results/")

    # Créer l'application Dash
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Créer et configurer le tableau de bord
    app = create_dashboard(
        app,
        meteo_figures,
        fissure_figures,
        modeling_figures,
        plotly_weekly_list,
        structure_figures,
        df_paliers_combined,
    )

    return app


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting main function")
    app = main()
    app.run_server(debug=True, port=8050)
