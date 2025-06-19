import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from prophet import Prophet
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS, add_constant
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis.models import model_fissures_with_explanatory_vars


def ajouter_troisieme_subplot(fig, df, df_old):
    """Ajoute un troisième subplot comparant Bureau_old et Bureau avec ajustement par régression linéaire."""

    # Tracer les données
    fig.add_trace(
        go.Scatter(
            x=df_old["Date"],
            y=(df_old["Bureau_old"] - df_old["Bureau_old"].mean())
            / df_old["Bureau_old"].std(),
            mode="lines",
            name="Bureau Old",
            line=dict(color="gray"),
        ),
        row=3,
        col=1,
    )

    # Mise à jour des axes
    fig.update_xaxes(title_text="Date", row=3, col=1, type="date")
    fig.update_yaxes(title_text="Historique (Z)", row=3, col=1)

    return fig


def dataviz_evolution(df, df_old):
    # Vérification et conversion des dates en type datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    if not pd.api.types.is_datetime64_any_dtype(df_old["Date"]):
        df_old["Date"] = pd.to_datetime(df_old["Date"])

    # Création de la figure avec 3 subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False)

    # Premier graphique
    bureau_norm = (df["Bureau"] - df["Bureau"].mean()) / df["Bureau"].std()
    mur_exterieur_norm = (df["Mur extérieur"] - df["Mur extérieur"].mean()) / df[
        "Mur extérieur"
    ].std()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=bureau_norm,
            mode="lines",
            name="Bureau",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=mur_exterieur_norm,
            mode="lines",
            name="Mur Extérieur",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    corr_pearson = pearsonr(df["Bureau"], df["Mur extérieur"])
    corr_spearman = spearmanr(df["Bureau"], df["Mur extérieur"])

    fig.add_annotation(
        x=df["Date"][2],
        y=0,
        text=f"Pearson: {corr_pearson[0]:.2f}<br>Spearman: {corr_spearman[0]:.2f}",
        showarrow=False,
        font=dict(size=16),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Écart (Z)", row=1, col=1)

    # Deuxième graphique
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Variation Bureau"],
            mode="lines",
            name="Variations Bureau",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Variation Mur"],
            mode="lines",
            name="Variations Mur",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.5), row=2, col=1)
    corr_pearson2 = pearsonr(
        df["Variation Bureau"].dropna(), df["Variation Mur"].dropna()
    )
    corr_spearman2 = spearmanr(
        df["Variation Bureau"].dropna(), df["Variation Mur"].dropna()
    )
    fig.add_annotation(
        x=df["Date"][2],
        y=0.5,
        text=f"Pearson: {corr_pearson2[0]:.2f}<br>Spearman: {corr_spearman2[0]:.2f}",
        showarrow=False,
        font=dict(size=16),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Variation (en mm)", row=2, col=1)

    # Appel de la fonction pour ajouter le troisième subplot
    fig = ajouter_troisieme_subplot(fig, df, df_old)

    # Mise à jour globale de la mise en page
    fig.update_layout(
        title="Evolution des écarts, de leurs variations et des mesures historiques",
        autosize=True,
        height=1100,
        width=None,
        legend=dict(orientation="h", x=0, y=-0.2),
        font=dict(size=20),
    )

    return fig


def calculate_durations(df_paliers):
    durations = []
    for i in range(len(df_paliers)):
        start_date = df_paliers.iloc[i]["Début"]
        end_date = df_paliers.iloc[i]["Fin"]
        duration = (end_date - start_date).days
        durations.append(duration)
    return durations


def calculate_heights(df_paliers):
    heights = []
    for i in range(len(df_paliers) - 1):
        height = (
            df_paliers.iloc[i + 1]["Valeur moyenne"]
            - df_paliers.iloc[i]["Valeur moyenne"]
        )
        heights.append(height)
    return heights


# Fonction pour convertir les hauteurs de mm à µm
def convert_mm_to_um(height_mm):
    return int(round(height_mm * 1000))


# Fonction d'analyse statistique des données (test de validité pour IC)
def test_stat(X_log, y_log, y_log_pred):
    # Calcul des résidus à partir de l'ensemble d'entraînement
    residuals = (
        y_log - y_log_pred
    )  # Différence entre les vraies valeurs et les valeurs prédites
    s = np.std(residuals, ddof=1)  # Écart-type des résidus
    n = len(X_log)  # Taille de l'échantillon d'entraînement

    print(f"Nombre d'observations dans l'ensemble d'entraînement (min=30) : {n}")

    # Tracer l'histogramme des résidus
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.title("Histogramme des résidus")
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.show()

    # QQ-plot pour visualiser la normalité
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ-plot des résidus")
    plt.show()

    # Test de Shapiro-Wilk
    shapiro_test = stats.shapiro(residuals)
    print(
        f"Test de Shapiro-Wilk: Statistique = {shapiro_test.statistic}, p-value = {shapiro_test.pvalue}"
    )

    # Tracer les résidus en fonction des valeurs prédites (homoscédasticité)
    plt.scatter(y_log_pred, residuals)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("Résidus vs Valeurs prédites")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus")
    plt.show()

    # Tracer l'ACF des résidus (indépendance des erreurs)
    plot_acf(residuals)
    plt.title("Fonction d'autocorrélation des résidus")
    plt.show()

    return None


def add_prophet_forecast(df_combined_old, df_combined_internew):
    # Préparer les données d'entraînement pour Prophet
    df_prophet_train = df_combined_old.reset_index()[["Date", "Bureau"]]
    df_prophet_train.columns = [
        "ds",
        "y",
    ]  # Prophet nécessite des colonnes nommées 'ds' (dates) et 'y' (valeurs)

    # Initialiser et entraîner le modèle Prophet
    model_prophet = Prophet(interval_width=0.95, daily_seasonality=True)
    model_prophet.fit(df_prophet_train)

    # Préparer les futures dates pour la période de prévision couvrant "internew"
    future_dates = pd.DataFrame(df_combined_internew.index).reset_index(drop=True)
    future_dates.columns = ["ds"]

    # Effectuer les prédictions sur toute la période "internew"
    forecast = model_prophet.predict(future_dates)

    # S'assurer que les prédictions couvrent la période "internew"
    forecast.set_index("ds", inplace=True)
    forecast = forecast[["yhat", "yhat_lower", "yhat_upper"]]

    # Transformer les prédictions en logarithme pour rester cohérent avec le modèle précédent
    df_combined_internew["prophet_pred"] = forecast["yhat"]
    df_combined_internew["prophet_pred_upper"] = forecast["yhat_upper"]
    df_combined_internew["prophet_pred_lower"] = forecast["yhat_lower"]

    return df_combined_internew


def add_catboost_forecast(df_combined_old, df_combined_internew):
    # Préparation des données d'entraînement CatBoost
    df_catboost_train = df_combined_old.reset_index()[["Date", "Bureau"]].dropna()

    # Ajouter les caractéristiques temporelles à df_catboost_train
    df_catboost_train["Days"] = (
        df_catboost_train["Date"] - df_catboost_train["Date"].min()
    ).dt.days
    df_catboost_train["year"] = df_catboost_train["Date"].dt.year
    df_catboost_train["month"] = df_catboost_train["Date"].dt.month
    df_catboost_train["day_of_year"] = df_catboost_train["Date"].dt.dayofyear
    df_catboost_train["day_of_week"] = df_catboost_train["Date"].dt.dayofweek
    df_catboost_train["week_of_year"] = (
        df_catboost_train["Date"].dt.isocalendar().week.astype(int)
    )

    # Ajouter des décalages (lags) uniquement à partir des données disponibles
    for lag in [1, 7, 30]:
        df_catboost_train[f"lag_{lag}"] = df_catboost_train["Bureau"].shift(lag)

    # Supprimer les lignes avec des NaN causés par les décalages
    df_catboost_train.dropna(inplace=True)

    # Cible dans l'échelle logarithmique
    y = np.log(df_catboost_train["Bureau"])

    # Caractéristiques d'entraînement
    features = [
        "Days",
        "year",
        "month",
        "day_of_year",
        "day_of_week",
        "week_of_year",
        "lag_1",
        "lag_7",
        "lag_30",
    ]

    # Diviser en ensemble d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        df_catboost_train[features], y, test_size=0.2, random_state=42
    )

    # Entraînement de CatBoost
    model_catboost = CatBoostRegressor(
        iterations=5000,  # Augmentation des itérations
        learning_rate=0.005,  # Réduction du taux d'apprentissage
        depth=10,
        silent=True,
        random_seed=42,
    )
    model_catboost.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=200,
        verbose=False,
    )

    # Ajout des caractéristiques à df_combined_internew
    df_combined_internew["Days"] = (
        df_combined_internew.index - df_catboost_train["Date"].min()
    ).days
    df_combined_internew["year"] = df_combined_internew.index.year
    df_combined_internew["month"] = df_combined_internew.index.month
    df_combined_internew["day_of_year"] = df_combined_internew.index.dayofyear
    df_combined_internew["day_of_week"] = df_combined_internew.index.dayofweek
    df_combined_internew["week_of_year"] = (
        df_combined_internew.index.isocalendar().week.astype(int)
    )

    # Remplir les lags à partir des valeurs récentes de l'entraînement
    df_combined_internew["lag_1"] = df_combined_old["Bureau"].iloc[-1]
    df_combined_internew["lag_7"] = (
        df_combined_old["Bureau"].iloc[-7]
        if len(df_combined_old) >= 7
        else df_combined_old["Bureau"].iloc[0]
    )
    df_combined_internew["lag_30"] = (
        df_combined_old["Bureau"].iloc[-30]
        if len(df_combined_old) >= 30
        else df_combined_old["Bureau"].iloc[0]
    )

    # Prédictions avec CatBoost
    catboost_pred_log = model_catboost.predict(df_combined_internew[features])

    # Convertir les prédictions dans l'échelle exponentielle
    catboost_pred = np.exp(catboost_pred_log)

    # **Ajustement progressif avec maintien du trend**
    initial_value = df_combined_old["Bureau"].iloc[-1]
    start_pred_value = catboost_pred[0]
    adjustment_factor = initial_value / start_pred_value

    # Appliquer un ajustement progressif sur les prédictions
    adjusted_catboost_pred = catboost_pred * np.linspace(
        adjustment_factor, 1, len(catboost_pred)
    )

    # Calcul de l'intervalle de prédiction à 95%
    residuals = np.exp(y_train) - np.exp(model_catboost.predict(X_train))
    residual_std = residuals.std()
    confidence_interval = 1.96 * residual_std

    # Insérer les valeurs ajustées
    df_combined_internew["catboost_pred"] = adjusted_catboost_pred
    df_combined_internew["catboost_pred_lower"] = (
        adjusted_catboost_pred - confidence_interval
    )
    df_combined_internew["catboost_pred_upper"] = (
        adjusted_catboost_pred + confidence_interval
    )

    # Supprimer les colonnes temporaires
    df_combined_internew.drop(
        columns=[
            "Days",
            "year",
            "month",
            "day_of_year",
            "day_of_week",
            "week_of_year",
            "lag_1",
            "lag_7",
            "lag_30",
        ],
        inplace=True,
    )

    return df_combined_internew


def preprocessing_old_new(df_fissures, df_fissures_old):
    print("\n\nFonction 'preprocessing_old_new'\n\n")

    # Lecture des données
    df_old = df_fissures_old
    df_new = df_fissures

    # Modèle exponentiel de la première date au 13/04/2016
    df_exp = df_old[df_old["Date"] <= "2016-04-13"]
    X_exp = (df_exp["Date"] - df_exp["Date"].min()).dt.days.values.reshape(-1, 1)
    y_exp = df_exp["Bureau_old"].values
    model_exp = LinearRegression().fit(
        X_exp, np.log(y_exp)
    )  # Régression sur les valeurs log
    y_exp_pred = np.exp(
        model_exp.predict(X_exp)
    )  # Appliquer l'exponentielle à la prédiction

    # Calcul du coefficient de corrélation pour le modèle exponentiel
    correlation_exp = np.corrcoef(y_exp, y_exp_pred)[0, 1]
    print(
        f"Coefficient de corrélation pour le modèle exponentiel: {correlation_exp:.4f}"
    )

    # Modèle logarithmique du 13/04/2016 à la dernière date des mesures 'old'
    df_log = df_old[df_old["Date"] >= "2016-04-13"]
    X_log = (df_log["Date"] - df_log["Date"].min()).dt.days.values.reshape(-1, 1)
    y_log = df_log["Bureau_old"].values
    model_log = LinearRegression().fit(
        X_log, np.exp(y_log)
    )  # Régression sur les valeurs exp
    y_log_pred = np.log(model_log.predict(X_log))

    # Calcul du coefficient de corrélation pour le modèle logarithmique
    correlation_log = np.corrcoef(y_log, np.exp(y_log_pred))[0, 1]
    print(
        f"Coefficient de corrélation pour le modèle logarithmique: {correlation_log:.4f}"
    )

    # Prédiction pour la première date de la seconde phase
    X_new_start = np.array(
        [(df_new["Date"].iloc[0] - df_log["Date"].min()).days]
    ).reshape(-1, 1)
    predicted_start = (
        np.log(model_log.predict(X_new_start)[0]) - 0.103
    )  # Offset dû à la fluctuation de la première mesure

    # Ajustement proportionnel
    scaling_factor = 1.12  # Estimation due au changement de hauteur de prise de mesures
    df_new["Bureau_new_adjusted"] = predicted_start + scaling_factor * (
        df_new["Bureau"] - df_new["Bureau"].iloc[0]
    )

    # Concaténer les deux séries
    df_combined = pd.concat(
        [
            df_old[["Date", "Bureau_old"]].rename(columns={"Bureau_old": "Bureau"}),
            df_new[["Date", "Bureau_new_adjusted"]].rename(
                columns={"Bureau_new_adjusted": "Bureau"}
            ),
        ]
    )

    # Configurer l'index sur la colonne 'Date'
    df_combined = df_combined.set_index("Date").sort_index()

    # Supprimer les doublons éventuels dans l'index (en gardant la première occurrence)
    df_combined = df_combined[~df_combined.index.duplicated(keep="first")]

    # Diviser les données pour les tracer séparément
    df_combined_old = df_combined.loc[df_old["Date"].min() : df_old["Date"].max()]

    # Ici, on veut exclure la dernière date de df_combined_old de df_combined_inter
    df_combined_inter = df_combined.loc[
        df_combined_old.index.max() + pd.Timedelta(days=1) : df_new["Date"].min()
    ]

    # Assurez-vous que df_combined_new commence après la dernière date de df_combined_old
    df_combined_new = df_combined.loc[df_new["Date"].min() : df_combined.index.max()]

    # Nouveau dataframe réindexé 'internew'
    # Générer une série de dates quotidiennes de start_date à end_date
    date_range = pd.date_range(
        start=df_old["Date"].max(), end=df_new["Date"].max(), freq="D"
    )

    # Créer le DataFrame avec la colonne 'Date'
    df_combined_internew = pd.DataFrame(date_range, columns=["Date"]).set_index("Date")

    # Prolongation logarithmique tendancielle sur 'inter'
    X_log_internew = (
        df_combined_internew.index - df_log["Date"].min()
    ).days.values.reshape(-1, 1)
    y_log_pred_internew = np.log(model_log.predict(X_log_internew))

    # Calculer X et y combinés pour les deux phases
    X_old_combined = (df_combined_old.index - df_combined_old.index.min()).days.values
    y_old_combined = df_combined_old["Bureau"].values
    X_new_combined = (df_combined_new.index - df_combined_new.index.min()).days.values
    y_new_combined = df_combined_new["Bureau"].values

    # Paliers et segments

    # Points de rupture pour la première phase avec 16 segments
    manual_breaks_old_dates = [
        "2010-12-01",
        "2012-06-01",
        "2013-04-01",
        "2013-06-01",
        "2014-04-01",
        "2014-06-01",
        "2015-03-01",
        "2015-10-01",
        "2016-03-01",
        "2016-09-01",
        "2016-12-10",
        "2017-07-01",
        "2018-03-15",
        "2018-04-15",
        "2019-02-01",
        "2019-07-01",
        "2020-09-01",
    ]
    manual_breaks_old = [
        df_combined_old.index.get_loc(
            df_combined_old.index[
                df_combined_old.index.get_indexer(
                    [pd.to_datetime(date)], method="nearest"
                )[0]
            ]
        )
        for date in manual_breaks_old_dates
    ]

    # Points de rupture pour la deuxième phase avec 5 segments
    manual_breaks_new_dates = [
        "2023-12-01",
        "2024-01-15",
        "2024-02-01",
        "2024-06-01",
        "2024-07-05",
        "2024-08-18",
        "2024-09-29",
        "2025-02-16",
        "2025-03-02",
        "2025-05-04",
        "2025-05-18",
        "2025-06-15"
    ]
    manual_breaks_new = [
        df_combined_new.index.get_loc(
            df_combined_new.index[
                df_combined_new.index.get_indexer(
                    [pd.to_datetime(date)], method="nearest"
                )[0]
            ]
        )
        for date in manual_breaks_new_dates
    ]

    # Générer les segments pour la première phase
    segments_old = []
    paliers_old = []  # Liste pour stocker les paliers

    # Traitement des paliers
    for i in range(len(manual_breaks_old) - 1):
        start = manual_breaks_old[i]
        end = manual_breaks_old[i + 1]
        if i == 0:
            pass  # déporté sur une fonction séparée
        elif i % 2 == 0:
            pass  # déporté sur une fonction séparée
        else:  # Segments plats (pente nulle)
            avg_value = np.mean(y_old_combined[start : end + 1])
            y_pred = np.full(end + 1 - start, avg_value)
            segments_old.append((df_combined_old.index[start : end + 1], y_pred))
            paliers_old.append(
                [df_combined_old.index[start], df_combined_old.index[end], avg_value]
            )

    # Générer les segments pour la deuxième phase
    segments_new = []
    paliers_new = []  # Liste pour stocker les paliers

    for i in range(len(manual_breaks_new) - 1):
        start = manual_breaks_new[i]
        end = manual_breaks_new[i + 1]
        if i % 2 == 0:  # Segments plats (pente nulle)
            avg_value = np.mean(y_new_combined[start : end + 1])
            y_pred = np.full(end + 1 - start, avg_value)
            segments_new.append((df_combined_new.index[start : end + 1], y_pred))
            paliers_new.append(
                [df_combined_new.index[start], df_combined_new.index[end], avg_value]
            )

    # Création des DataFrames pour les paliers
    df_paliers_old = pd.DataFrame(
        paliers_old, columns=["Début", "Fin", "Valeur moyenne"]
    )
    df_paliers_new = pd.DataFrame(
        paliers_new, columns=["Début", "Fin", "Valeur moyenne"]
    )

    # Modélisation exponentielle et logarithmique

    # Ajouter les prédictions du modèle exponentiel à df_combined_old
    df_combined_old["model_exp"] = np.nan
    df_combined_old.loc[df_exp["Date"], "model_exp"] = y_exp_pred

    # Ajouter les prédictions du modèle logarithmique à df_combined_old
    df_combined_old["model_log"] = np.nan
    df_combined_old.loc[df_log["Date"], "model_log"] = y_log_pred

    # Ajouter les prédictions du modèle logarithmique à df_combined_new
    df_combined_internew["model_log_internew"] = np.nan
    df_combined_internew.loc[df_combined_internew.index, "model_log_internew"] = (
        y_log_pred_internew
    )

    # Intervalle de confiance (IC) sur les données d'entraînement de 'old'

    # Eléments statistiques dans l'espace linéaire (exp)
    n = len(y_log)
    s = np.std(np.exp(y_log))
    IC_Student = 2 * 1.96 * s / np.sqrt(n)

    # Prédictions exponentielles du modèle logarithmique
    y_exp_regression = model_log.predict(X_log)

    # IC de Student dans l'espace réel
    y_log_upper = np.log(y_exp_regression + IC_Student / 2)
    y_log_lower = np.log(y_exp_regression - IC_Student / 2)

    # Calcul des distances orthogonales avec les signes (positive au-dessus, négative en dessous)
    distances_orthogonales = (np.exp(y_log) - y_exp_regression) / np.sqrt(
        1 + model_log.coef_[0] ** 2
    )

    # Tri des distances en valeur absolue
    sorted_indices = np.argsort(np.abs(distances_orthogonales))

    # Détermination du nombre de points à garder : 95% de l'ensemble des points d'entraînement
    num_points_to_keep = int(0.95 * len(distances_orthogonales))

    # Identification des points à exclure : les 5% les plus éloignés
    indices_to_keep = sorted_indices[:num_points_to_keep]

    # Calcul des distances maximales positives et négatives parmi les points restants
    distance_orthogonale_pos = np.max(distances_orthogonales[indices_to_keep])
    distance_orthogonale_neg = np.min(distances_orthogonales[indices_to_keep])

    # Construction des droites parallèles pour l'IC dans l'espace exponentiel
    model_log_upper_exp = y_exp_regression + distance_orthogonale_pos * np.sqrt(
        1 + model_log.coef_[0] ** 2
    )
    model_log_lower_exp = y_exp_regression + distance_orthogonale_neg * np.sqrt(
        1 + model_log.coef_[0] ** 2
    )

    # Transformation de ces droites en courbes logarithmiques
    model_log_upper = np.log(model_log_upper_exp)
    model_log_lower = np.log(model_log_lower_exp)

    # Ajouter les IC au DataFrame d'entraînement
    df_combined_old.loc[df_log["Date"], "model_log_upper"] = model_log_upper
    df_combined_old.loc[df_log["Date"], "model_log_lower"] = model_log_lower
    df_combined_old.loc[df_log["Date"], "Student_log_upper"] = y_log_upper
    df_combined_old.loc[df_log["Date"], "Student_log_lower"] = y_log_lower

    # Intervalle de prédiction (IP)

    X_internew = df_combined_internew.index

    # Convertir la date de référence et X_internew en 'datetime64[D]'
    date_ref = np.datetime64(
        df_log["Date"].min(), "D"
    )  # Début des données d'entraînement
    X_internew_days = (
        pd.to_datetime(X_internew).to_numpy().astype("datetime64[D]") - date_ref
    ).astype(int)

    # Effectuer les prédictions avec model_log dans l'espace réel (log, non linéarisé)
    pred_log = np.log(model_log.predict(X_internew_days.reshape(-1, 1)))

    # Éléments statistiques
    n = len(X_log)  # Taille de l'échantillon
    y_pred = np.log(
        model_log.predict(X_log)
    )  # Prédictions du modèle sur les données d'entraînement (espace réel)
    # Calcul de la variance de l'erreur résiduelle
    var_res_log = np.sum((y_pred - y_log) ** 2) / (n - 2)
    # Calcul de la variance de la prédiction
    X_log_days = (X_log.astype("datetime64[D]") - date_ref).astype(int)
    x_mean = np.mean(X_log_days)
    var_pred = var_res_log * (
        1
        + 1 / n
        + ((X_internew_days - x_mean) ** 2) / np.sum((X_log_days - x_mean) ** 2)
    )

    # Intervalle de prédiction dans l'espace réel
    pred_log_upper = pred_log + 1.96 * np.sqrt(var_res_log + var_pred)
    pred_log_lower = pred_log - 1.96 * np.sqrt(var_res_log + var_pred)

    # Raccordement IC et IP - Correction avec ré-échantillonnage

    # Passer dans l'espace linéaire pour tout traiter d'un seul tenant
    pred_lin = np.exp(pred_log)
    pred_lin_upper = np.exp(pred_log_upper)
    pred_lin_lower = np.exp(pred_log_lower)

    # Récupération des offsets à la fin de l'IC (Intervalle de Confiance) dans l'espace linéaire
    final_IC_upper_lin = np.exp(df_combined_old["model_log_upper"].iloc[-1])
    final_IC_lower_lin = np.exp(df_combined_old["model_log_lower"].iloc[-1])
    final_prediction_lin = np.exp(df_combined_old["model_log"].iloc[-1])

    upper_offset_start = final_IC_upper_lin - final_prediction_lin
    lower_offset_start = final_prediction_lin - final_IC_lower_lin

    # Récupération des offsets à la fin de l'IP non corrigé dans l'espace linéaire
    final_pred_upper_lin = pred_lin_upper[-1]
    final_pred_lower_lin = pred_lin_lower[-1]
    final_pred_lin = pred_lin[-1]

    upper_offset_end = final_pred_upper_lin - final_pred_lin
    lower_offset_end = final_pred_lin - final_pred_lower_lin

    # Ajustement progressif sur l'ensemble de l'IP traité comme un tout
    adjusted_upper = np.zeros_like(pred_lin)
    adjusted_lower = np.zeros_like(pred_lin)

    def find_optimal_k(
        df_combined_new,
        pred_lin,
        upper_offset_start,
        upper_offset_end,
        lower_offset_start,
        lower_offset_end,
        X_internew,
        tol=0.01,
    ):
        # Définir les bornes inférieure et supérieure de la recherche de k
        k_min = 0.1 / len(X_internew)
        k_max = 5 / len(X_internew)

        def compute_coverage(k):
            # Initialiser les vecteurs ajustés
            adjusted_upper = np.zeros(len(X_internew))
            adjusted_lower = np.zeros(len(X_internew))

            # Calcul des offsets ajustés
            for i in range(len(X_internew)):
                factor = (np.exp(k * i) - 1) / (np.exp(k * len(X_internew)) - 1)
                interpolated_upper_offset = (
                    1 - factor
                ) * upper_offset_start + factor * upper_offset_end
                interpolated_lower_offset = (
                    1 - factor
                ) * lower_offset_start + factor * lower_offset_end
                adjusted_upper[i] = pred_lin[i] + interpolated_upper_offset
                adjusted_lower[i] = pred_lin[i] - interpolated_lower_offset

            # Mettre à jour les colonnes dans df_combined_new pour vérifier la couverture
            df_combined_new["pred_log_upper"] = np.log(adjusted_upper)
            df_combined_new["pred_log_lower"] = np.log(adjusted_lower)

            # Calculer le taux de couverture
            return IPnew_ratio(df_combined_new)

        # Recherche par dichotomie pour trouver la meilleure valeur de k
        while k_max - k_min > tol:
            k_mid = (k_min + k_max) / 2
            coverage = compute_coverage(k_mid)

            print(f"Essai avec k={k_mid:.6f}, taux de couverture={coverage * 100:.2f}%")

            if coverage > 0.95:
                k_max = k_mid  # Réduire k car la couverture est trop élevée
            else:
                k_min = k_mid  # Augmenter k car la couverture est trop faible

        optimal_k = (k_min + k_max) / 2
        print(f"Valeur optimale de k trouvée : {optimal_k:.6f}")
        return optimal_k

    # Trouver la valeur optimale de k
    optimal_k = find_optimal_k(
        df_combined_new,
        pred_lin,
        upper_offset_start,
        upper_offset_end,
        lower_offset_start,
        lower_offset_end,
        X_internew,
    )

    # Utiliser la valeur optimale de k trouvée pour le calcul final
    k = optimal_k

    # Application de l'ajustement exponentiel avec k optimal
    for i in range(len(X_internew)):
        factor = (np.exp(k * i) - 1) / (np.exp(k * len(X_internew)) - 1)
        interpolated_upper_offset = (
            1 - factor
        ) * upper_offset_start + factor * upper_offset_end
        interpolated_lower_offset = (
            1 - factor
        ) * lower_offset_start + factor * lower_offset_end
        adjusted_upper[i] = pred_lin[i] + interpolated_upper_offset
        adjusted_lower[i] = pred_lin[i] - interpolated_lower_offset

    # Transformation logarithmique finale après l'ajustement sur l'ensemble de l'IP ré-échantillonné
    pred_log_upper = np.log(adjusted_upper)
    pred_log_lower = np.log(adjusted_lower)

    # Insérer les valeurs de pred_log, pred_log_upper et pred_log_lower dans les DataFrames
    df_combined_internew["pred_log"] = pred_log
    df_combined_internew["pred_log_upper"] = pred_log_upper
    df_combined_internew["pred_log_lower"] = pred_log_lower

    # Répartition des données 'internew' dans 'inter' et 'new'

    # 1. Créer une plage complète de dates couvrant la période de df_combined_inter et df_combined_new
    date_range_inter = pd.date_range(
        start=df_combined_inter.index.min(), end=df_combined_inter.index.max(), freq="D"
    )

    # 2. Réindexer les DataFrames en utilisant ces plages de dates complètes
    df_combined_inter = df_combined_inter.reindex(date_range_inter)

    # 3. Conserver les valeurs préexistantes et combiner avec df_combined_internew
    # Utiliser `combine_first` pour ne pas écraser les valeurs préexistantes dans df_combined_inter
    df_combined_inter.update(
        df_combined_internew[["pred_log", "pred_log_upper", "pred_log_lower"]]
    )

    # 4. Assurer que les valeurs des colonnes préexistantes ne sont pas affectées
    # Remplir les valeurs NaN pour les dates qui n'étaient pas présentes initialement
    df_combined_inter.update(df_combined_inter.reindex(df_combined_inter.index))

    # Garder les dates pour lesquelles des valeurs existaient déjà intactes
    df_combined_inter = df_combined_inter.sort_index()

    # Si des doublons ou incohérences apparaissent dans les prédictions, remplacez les valeurs incorrectes par les valeurs d'origine
    df_combined_inter.loc[
        df_combined_inter.index.difference(df_combined_internew.index),
        ["pred_log", "pred_log_upper", "pred_log_lower"],
    ] = np.nan

    # 1. Créer une plage complète de dates couvrant df_combined_new et df_combined_internew
    full_date_range_new = pd.date_range(
        start=min(df_combined_new.index.min(), df_combined_internew.index.min()),
        end=max(df_combined_new.index.max(), df_combined_internew.index.max()),
        freq="D",
    )

    # 2. Réindexer df_combined_new sur cette plage complète sans perdre les valeurs existantes
    df_combined_new = df_combined_new.reindex(full_date_range_new)

    # 3. Ajouter explicitement les colonnes de prédiction si elles n'existent pas
    for col in ["pred_log", "pred_log_upper", "pred_log_lower"]:
        if col not in df_combined_new.columns:
            df_combined_new[col] = np.nan

    # 4. Mettre à jour df_combined_new avec les valeurs de df_combined_internew
    df_combined_new.update(
        df_combined_internew[["pred_log", "pred_log_upper", "pred_log_lower"]]
    )

    # 5. Réintégrer la colonne 'Bureau' en utilisant les données originales pour respecter les dates
    df_combined_new["Bureau"] = df_combined["Bureau"].reindex(df_combined_new.index)

    # 6. Vérifier la présence des colonnes
    print("Colonnes actuelles de df_combined_new:", df_combined_new.columns)

    # 7. Supprimer les lignes qui sont en dehors de la plage de dates d'origine de df_combined_new
    df_combined_new = df_combined_new.loc[
        df_combined.index.min() : df_combined.index.max()
    ]

    # 8. Trier par index
    df_combined_new = df_combined_new.sort_index()

    def IPnew_ratio(df_combined_new):
        # Exclure les lignes où les prédictions sont NaN
        valid_points = df_combined_new[
            ["Bureau", "pred_log_lower", "pred_log_upper"]
        ].dropna()

        # Calculer si les valeurs sont à l'intérieur de l'intervalle
        inside_interval_new = (
            valid_points["Bureau"] >= valid_points["pred_log_lower"]
        ) & (valid_points["Bureau"] <= valid_points["pred_log_upper"])

        # Calculer le taux de couverture
        coverage_new = inside_interval_new.mean()
        return coverage_new

    # Validation empirique pour la période "new"
    print(
        f"Taux de couverture empirique pour la période 'new': {IPnew_ratio(df_combined_new) * 100:.2f}%"
    )

    # Prophet

    # Ajouter les prédictions Prophet à df_combined_internew
    df_combined_internew = add_prophet_forecast(
        df_combined.loc[df_log["Date"].min() : df_old["Date"].max()],
        df_combined_internew,
    )

    # Répartition des prédictions Prophet de 'df_combined_internew' dans 'df_combined_inter' et 'df_combined_new'

    # Réindexer df_combined_inter pour inclure toutes les dates quotidiennes manquantes
    date_range_inter = pd.date_range(
        start=df_combined_inter.index.min(), end=df_combined_inter.index.max(), freq="D"
    )
    df_combined_inter = df_combined_inter.reindex(date_range_inter)

    # Mettre à jour df_combined_inter avec les prévisions Prophet de 'df_combined_internew'
    df_combined_inter[["prophet_pred", "prophet_pred_upper", "prophet_pred_lower"]] = (
        df_combined_internew[
            ["prophet_pred", "prophet_pred_upper", "prophet_pred_lower"]
        ]
    )

    # Réindexer df_combined_new pour inclure toutes les dates quotidiennes manquantes
    full_date_range_new = pd.date_range(
        start=df_combined_new.index.min(), end=df_combined_new.index.max(), freq="D"
    )
    df_combined_new = df_combined_new.reindex(full_date_range_new)

    # Mettre à jour df_combined_new avec les prévisions Prophet de 'df_combined_internew'
    df_combined_new[["prophet_pred", "prophet_pred_upper", "prophet_pred_lower"]] = (
        df_combined_internew[
            ["prophet_pred", "prophet_pred_upper", "prophet_pred_lower"]
        ]
    )

    def Prophet_IPnew_ratio(df_combined_new):
        # Exclure les lignes où les prédictions Prophet sont NaN
        valid_points = df_combined_new[
            ["Bureau", "prophet_pred_lower", "prophet_pred_upper"]
        ].dropna()

        # Calculer si les valeurs de 'Bureau' sont à l'intérieur de l'intervalle de prédiction de Prophet
        inside_interval_new = (
            valid_points["Bureau"] >= valid_points["prophet_pred_lower"]
        ) & (valid_points["Bureau"] <= valid_points["prophet_pred_upper"])

        # Calculer le taux de couverture
        coverage_new = inside_interval_new.mean()
        return coverage_new

    # Affichage du taux de couverture empirique pour la période 'new'
    print(
        f"Taux de couverture empirique pour Prophet sur la période 'new': {Prophet_IPnew_ratio(df_combined_new) * 100:.2f}%"
    )

    # # CatBoost
    #
    # # Appel à CatBoost
    # df_combined_internew = add_catboost_forecast(df_combined.loc[df_log["Date"].min():df_old["Date"].max()],
    #                                              df_combined_internew)
    #
    # # Fusionner les prédictions de CatBoost avec les DataFrames inter et new
    # df_combined_inter[['catboost_pred', 'catboost_pred_lower', 'catboost_pred_upper']] = df_combined_internew[
    #     ['catboost_pred', 'catboost_pred_lower', 'catboost_pred_upper']]
    # df_combined_new[['catboost_pred', 'catboost_pred_lower', 'catboost_pred_upper']] = df_combined_internew[
    #     ['catboost_pred', 'catboost_pred_lower', 'catboost_pred_upper']]

    print("\n\nFin de la fonction 'preprocessing_old_new'\n\n")

    return (
        df_combined_old,
        df_combined_new,
        X_old_combined,
        y_old_combined,
        X_new_combined,
        y_new_combined,
        manual_breaks_old,
        manual_breaks_new,
        segments_old,
        paliers_old,
        segments_new,
        paliers_new,
        df_paliers_old,
        df_paliers_new,
        df_combined_inter,
    )


def plot_scatter_plotly(
    df_combined_old, y_old_combined, df_combined_new, y_new_combined, df_combined_inter
):
    fig = go.Figure()

    # Ajouter les plages d'incertitude 95% effectifs pour la période "old"
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=df_combined_old["model_log_upper"],
            mode="lines",
            name="Plage supérieure (old)",
            line=dict(dash="dash", color="#00CC96"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=df_combined_old["model_log_lower"],
            # fill='tonexty',
            mode="lines",
            name="Intervalle à 95 % effectifs",
            line=dict(dash="dash", color="#00CC96"),
            # fillcolor='rgba(0, 204, 150, 0.15)',
            showlegend=True,
        )
    )

    # Ajouter les plages d'incertitude Student pour la période "old"
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=df_combined_old["Student_log_upper"],
            mode="lines",
            name="Plage supérieure (old)",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=df_combined_old["Student_log_lower"],
            fill="tonexty",
            mode="lines",
            name="Intervalle de confiance de Student à 95 %",
            line=dict(width=0),
            fillcolor="rgba(255, 0, 0, 0.15)",
            showlegend=True,
        )
    )

    # Modèle exponentiel (première partie 'old')
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=df_combined_old["model_exp"],
            mode="lines",
            line=dict(color="#EF553B"),
            name="Modèle exponentiel",
            opacity=0.8,
        )
    )

    # Modèle logarithmique (seconde partie 'old')
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=df_combined_old["model_log"],
            mode="lines",
            line=dict(color="#00CC96"),
            name="Modèle logarithmique",
            opacity=0.8,
        )
    )

    # Ajouter les plages d'incertitude pour la période "inter"
    fig.add_trace(
        go.Scatter(
            x=df_combined_inter.index,
            y=df_combined_inter["pred_log_upper"],
            mode="lines",
            name="Plage supérieure (inter)",
            line=dict(dash="dash", color="rgba(128, 128, 128, 0.3)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_inter.index,
            y=df_combined_inter["pred_log_lower"],
            fill="tonexty",
            mode="lines",
            name="Intervalle de prévisions à 95 % effectifs",
            line=dict(dash="dash", color="rgba(128, 128, 128, 0.3)"),
            fillcolor="rgba(128, 128, 128, 0.15)",
            showlegend=True,
        )
    )

    # Ajouter les plages d'incertitude pour la période "new"
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["pred_log_upper"],
            mode="lines",
            name="Plage supérieure (new)",
            line=dict(dash="dash", color="rgba(128, 128, 128, 0.3)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["pred_log_lower"],
            fill="tonexty",
            mode="lines",
            name="Plage inférieure (new)",
            line=dict(dash="dash", color="rgba(128, 128, 128, 0.3)"),
            fillcolor="rgba(128, 128, 128, 0.15)",
            showlegend=False,
        )
    )

    # Modèle logarithmique tendanciel "inter"
    fig.add_trace(
        go.Scatter(
            x=df_combined_inter.index,
            y=df_combined_inter["pred_log"],
            mode="lines",
            line=dict(dash="dash", color="#EF553B"),
            name="Prévisions logarithmiques",
            showlegend=True,
        )
    )

    # Modèle logarithmique tendanciel "new"
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["pred_log"],
            mode="lines",
            line=dict(dash="dash", color="#EF553B"),
            name="Dérive logarithmique",
            showlegend=False,
        )
    )

    # Assurer que 'df_combined_new' commence après la dernière date de 'df_combined_old'
    df_combined_new = df_combined_new[
        df_combined_new.index > df_combined_old.index.max()
    ]

    # Scatter plot pour la période ancienne
    fig.add_trace(
        go.Scatter(
            x=df_combined_old.index,
            y=y_old_combined,
            mode="markers",
            marker=dict(color="gray", size=5),
            name="Période Ancienne",
        )
    )

    # Modèle PELT : Structural Break Analysis (Ruptures)
    fig.add_trace(
        go.Scatter(
            x=[df_combined_old.index[70], df_combined_old.index[70]],
            y=[
                df_combined_old["model_exp"].max(),
                df_combined_old["model_log_upper"].min(),
            ],
            mode="lines",
            line=dict(color="lightgreen", width=5, dash="solid"),
            name="PELT : Structural Break Analysis",
        )
    )

    # HMM : 2 états
    fig.add_trace(
        go.Scatter(
            x=[df_combined_old.index[62], df_combined_old.index[62]],
            y=[
                df_combined_old["model_exp"].max(),
                df_combined_old["model_log_upper"].min(),
            ],
            mode="lines",
            line=dict(color="lightpink", width=5, dash="solid"),
            name="HMM à 2 états",
        )
    )

    # Travaux IPN
    fig.add_trace(
        go.Scatter(
            x=[df_combined_old.index[64], df_combined_old.index[64]],
            y=[
                df_combined_old["model_exp"].max(),
                df_combined_old["model_log_upper"].min(),
            ],
            mode="lines",
            line=dict(color="darkblue", width=5, dash="solid"),
            name="Installation IPN",
        )
    )

    # Scatter plot pour la période récente
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["Bureau"],
            mode="markers",
            marker=dict(color="blue", size=5),
            name="Période Récente (Ajustée)",
        )
    )

    # Ajouter les prédictions de Prophet pour la période "inter"
    fig.add_trace(
        go.Scatter(
            x=df_combined_inter.index,
            y=df_combined_inter["prophet_pred_upper"],
            mode="lines",
            name="Plage supérieure (Prophet inter)",
            line=dict(
                dash="dash", color="rgba(0, 102, 204, 0.3)"
            ),  # Couleur bleu semi-transparente
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_inter.index,
            y=df_combined_inter["prophet_pred_lower"],
            fill="tonexty",
            mode="lines",
            name="Intervalle de prévision Prophet à 95 %",
            line=dict(dash="dash", color="rgba(0, 102, 204, 0.3)"),
            fillcolor="rgba(0, 102, 204, 0.05)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_inter.index,
            y=df_combined_inter["prophet_pred"],
            mode="lines",
            line=dict(dash="dash", color="#0066CC"),  # Couleur bleu foncé
            name="Prévision Prophet",
            showlegend=True,
        )
    )

    # Ajouter les prédictions de Prophet pour la période "new"
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["prophet_pred_upper"],
            mode="lines",
            name="Plage supérieure (Prophet new)",
            line=dict(
                dash="dash", color="rgba(0, 102, 204, 0.3)"
            ),  # Couleur bleu semi-transparente
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["prophet_pred_lower"],
            fill="tonexty",
            mode="lines",
            name="Intervalle de prévision Prophet à 95 %",
            line=dict(dash="dash", color="rgba(0, 102, 204, 0.3)"),
            fillcolor="rgba(0, 102, 204, 0.05)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_combined_new.index,
            y=df_combined_new["prophet_pred"],
            mode="lines",
            line=dict(dash="dash", color="#0066CC"),  # Couleur bleu foncé
            name="Prévision Prophet",
            showlegend=False,
        )
    )

    # # Ajouter les plages d'incertitude CatBoost pour la période "new"
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_combined_new.index,
    #         y=df_combined_new['catboost_pred_upper'],
    #         mode='lines',
    #         name='Plage supérieure (CatBoost - new)',
    #         line=dict(dash='dash', color='rgba(255, 127, 80, 0.4)'),
    #         showlegend=False
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_combined_new.index,
    #         y=df_combined_new['catboost_pred_lower'],
    #         fill='tonexty',
    #         mode='lines',
    #         name='Plage inférieure CatBoost (new)',
    #         line=dict(dash='dash', color='rgba(255, 127, 80, 0.4)'),
    #         fillcolor='rgba(255, 127, 80, 0.15)',
    #         showlegend=False
    #     )
    # )
    #
    # # Prévision CatBoost tendancielle "inter"
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_combined_inter.index,
    #         y=df_combined_inter["catboost_pred"],
    #         mode='lines',
    #         line=dict(dash='dash', color='#FF7F50'),
    #         name='Prévision CatBoost (inter)',
    #         showlegend=True
    #     ))
    #
    # # Prévision CatBoost tendancielle "new"
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_combined_new.index,
    #         y=df_combined_new["catboost_pred"],
    #         mode='lines',
    #         line=dict(dash='dash', color='#FF7F50'),
    #         name='Prévision CatBoost (new)',
    #         showlegend=False
    #     ))

    # Layout
    fig.update_layout(
        width=None,
        height=None,
        xaxis_title="Date",
        yaxis_title="Écartement de la Fissure (mm)",
        title="Évolution de l'Écartement de la Fissure",
        showlegend=True,
        legend=dict(font=dict(size=12)),
    )

    fig_plot_scatter_plotly = fig

    return fig_plot_scatter_plotly


def plot_segments_plotly(fig_plot_scatter_plotly, segments_old, segments_new):
    # Tracer les segments pour la première phase (en bleu)
    for segment in segments_old:
        if (
            len(segment[0]) > 1 and len(segment[1]) > 1
        ):  # S'assurer qu'il y a bien des points à tracer
            fig_plot_scatter_plotly.add_trace(
                go.Scatter(
                    x=segment[0],
                    y=segment[1],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    showlegend=False,
                )
            )

    # Tracer les segments pour la deuxième phase (en rouge)
    for segment in segments_new:
        if len(segment[0]) > 1 and len(segment[1]) > 1:
            fig_plot_scatter_plotly.add_trace(
                go.Scatter(
                    x=segment[0],
                    y=segment[1],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    showlegend=False,
                )
            )

    fig_plot_segments_plotly = fig_plot_scatter_plotly
    return fig_plot_segments_plotly


def plot_additional_segments_plotly(
    fig_plot_segments_plotly,
    df_combined_old,
    y_old_combined,
    df_paliers_old,
    df_paliers_new,
):
    # Ajouter le segment du premier point au début du premier palier
    premier_point_x = df_combined_old.index[0]
    premier_point_y = y_old_combined[0]
    debut_premier_palier_x = df_paliers_old.iloc[0]["Début"]
    debut_premier_palier_y = df_paliers_old.iloc[0]["Valeur moyenne"]

    fig_plot_segments_plotly.add_trace(
        go.Scatter(
            x=[premier_point_x, debut_premier_palier_x],
            y=[premier_point_y, debut_premier_palier_y],
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            showlegend=False,
        )
    )

    # Ajouter les segments entre paliers pour la première phase (en bleu)
    for i in range(len(df_paliers_old) - 1):
        fig_plot_segments_plotly.add_trace(
            go.Scatter(
                x=[df_paliers_old.iloc[i]["Fin"], df_paliers_old.iloc[i + 1]["Début"]],
                y=[
                    df_paliers_old.iloc[i]["Valeur moyenne"],
                    df_paliers_old.iloc[i + 1]["Valeur moyenne"],
                ],
                mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                showlegend=False,
            )
        )

    # Ajouter les segments entre paliers pour la deuxième phase (en rouge)
    for i in range(len(df_paliers_new) - 1):
        fig_plot_segments_plotly.add_trace(
            go.Scatter(
                x=[df_paliers_new.iloc[i]["Fin"], df_paliers_new.iloc[i + 1]["Début"]],
                y=[
                    df_paliers_new.iloc[i]["Valeur moyenne"],
                    df_paliers_new.iloc[i + 1]["Valeur moyenne"],
                ],
                mode="lines",
                line=dict(color="blue", width=2, dash="dash"),
                showlegend=False,
            )
        )

    fig_plot_additional_segments_plotly = fig_plot_segments_plotly

    return fig_plot_additional_segments_plotly


def annotate_durations_plotly(
    fig_plot_additional_segments_plotly,
    df_paliers_old,
    df_paliers_new,
    horizontal_durations_old,
    horizontal_durations_new,
):
    # Ajouter les annotations pour les paliers (durée) pour la première phase
    for i in range(len(df_paliers_old)):
        duration_days = horizontal_durations_old[i]
        mid_point = (
            df_paliers_old.iloc[i]["Début"]
            + (df_paliers_old.iloc[i]["Fin"] - df_paliers_old.iloc[i]["Début"]) / 2
        )
        fig_plot_additional_segments_plotly.add_annotation(
            x=mid_point,
            y=df_paliers_old.iloc[i]["Valeur moyenne"],
            text=f"{duration_days} jours",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=30,
            font=dict(color="blue", size=10),
            bgcolor="cornsilk",
            # opacity=0.6
        )

    # Ajouter les annotations pour les paliers (durée) pour la deuxième phase
    for i in range(len(df_paliers_new)):
        duration_days = horizontal_durations_new[i]
        mid_point = (
            df_paliers_new.iloc[i]["Début"]
            + (df_paliers_new.iloc[i]["Fin"] - df_paliers_new.iloc[i]["Début"]) / 2
        )
        fig_plot_additional_segments_plotly.add_annotation(
            x=mid_point,
            y=df_paliers_new.iloc[i]["Valeur moyenne"],
            text=f"{duration_days} jours",
            showarrow=True,
            arrowhead=2,
            ax=15,
            ay=30,
            font=dict(color="blue", size=10),
            bgcolor="cornsilk",
        )

    fig_annotate_durations_plotly = fig_plot_additional_segments_plotly

    return fig_annotate_durations_plotly


def add_vertical_segments_and_heights_plotly(
    fig_annotate_durations_plotly,
    df_paliers_old,
    df_paliers_new,
    horizontal_heights_old,
    horizontal_heights_new,
):
    # Ajouter les segments verticaux et les annotations pour les hauteurs pour la première phase
    for i in range(len(df_paliers_old) - 1):
        height_um = convert_mm_to_um(horizontal_heights_old[i])
        fig_annotate_durations_plotly.add_trace(
            go.Scatter(
                x=[df_paliers_old.iloc[i]["Fin"], df_paliers_old.iloc[i]["Fin"]],
                y=[
                    df_paliers_old.iloc[i]["Valeur moyenne"],
                    df_paliers_old.iloc[i + 1]["Valeur moyenne"],
                ],
                mode="lines",
                line=dict(color="gray", width=2, dash="solid"),
                opacity=0.3,
                showlegend=False,
            )
        )
        fig_annotate_durations_plotly.add_annotation(
            x=df_paliers_old.iloc[i]["Fin"],
            y=(
                df_paliers_old.iloc[i]["Valeur moyenne"]
                + df_paliers_old.iloc[i + 1]["Valeur moyenne"]
            )
            / 2,
            text=f"{height_um} µm",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-10,
            font=dict(color="blue", size=10),
            bgcolor="lightgray",
            # opacity=0.6
        )

    # Ajouter les segments verticaux et les annotations pour les hauteurs pour la deuxième phase
    for i in range(len(df_paliers_new) - 1):
        height_um = convert_mm_to_um(horizontal_heights_new[i])
        fig_annotate_durations_plotly.add_trace(
            go.Scatter(
                x=[df_paliers_new.iloc[i]["Fin"], df_paliers_new.iloc[i]["Fin"]],
                y=[
                    df_paliers_new.iloc[i]["Valeur moyenne"],
                    df_paliers_new.iloc[i + 1]["Valeur moyenne"],
                ],
                mode="lines",
                line=dict(color="blue", width=2, dash="solid"),
                opacity=0.3,
                showlegend=False,
            )
        )
        fig_annotate_durations_plotly.add_annotation(
            x=df_paliers_new.iloc[i]["Fin"],
            y=(
                df_paliers_new.iloc[i]["Valeur moyenne"]
                + df_paliers_new.iloc[i + 1]["Valeur moyenne"]
            )
            / 2,
            text=f"{height_um} µm",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-10,
            font=dict(color="blue", size=10),
            bgcolor="lightgray",
            # opacity=0.6
        )

    fig_add_vertical_segments_and_heights_plotly = fig_annotate_durations_plotly
    return fig_add_vertical_segments_and_heights_plotly


def dataviz_old_new(df_fissures, df_fissures_old):
    # Prétraitement des données
    (
        df_combined_old,
        df_combined_new,
        X_old_combined,
        y_old_combined,
        X_new_combined,
        y_new_combined,
        manual_breaks_old,
        manual_breaks_new,
        segments_old,
        paliers_old,
        segments_new,
        paliers_new,
        df_paliers_old,
        df_paliers_new,
        df_combined_inter,
    ) = preprocessing_old_new(df_fissures, df_fissures_old)

    # Appel à la fonction de modélisation avec les paliers
    model_results = model_fissures_with_explanatory_vars(df_paliers_old, df_paliers_new)

    # Afficher les RMSE pour chaque modèle
    for model_name, result in model_results.items():
        print(f"{model_name} - RMSE: {result['rmse']:.4f}")

    # Vérifier l'absence de doublons
    if df_combined_old.index.duplicated().any():
        print("Doublons dans df_combined_old")
        print(df_combined_old[df_combined_old.index.duplicated(keep=False)])

    if df_combined_new.index.duplicated().any():
        print("Doublons dans df_combined_new")
        print(df_combined_new[df_combined_new.index.duplicated(keep=False)])

    # Calculer les durées et les hauteurs pour les annotations
    horizontal_durations_old = calculate_durations(df_paliers_old)
    horizontal_durations_new = calculate_durations(df_paliers_new)
    horizontal_heights_old = calculate_heights(df_paliers_old)
    horizontal_heights_new = calculate_heights(df_paliers_new)

    # Créer la figure initiale avec les scatter plots
    fig = plot_scatter_plotly(
        df_combined_old,
        y_old_combined,
        df_combined_new,
        y_new_combined,
        df_combined_inter,
    )

    # Ajouter les segments de paliers
    fig = plot_segments_plotly(fig, segments_old, segments_new)

    # Ajouter le segment du premier point au début du premier palier et les segments entre paliers
    fig = plot_additional_segments_plotly(
        fig, df_combined_old, y_old_combined, df_paliers_old, df_paliers_new
    )

    # Ajouter les annotations pour les durées des paliers
    fig = annotate_durations_plotly(
        fig,
        df_paliers_old,
        df_paliers_new,
        horizontal_durations_old,
        horizontal_durations_new,
    )

    # Ajouter les segments verticaux et les annotations pour les hauteurs
    fig = add_vertical_segments_and_heights_plotly(
        fig,
        df_paliers_old,
        df_paliers_new,
        horizontal_heights_old,
        horizontal_heights_new,
    )

    # Configuration finale du graphique avec les détails sur les dates et l'axe des x
    fig.update_layout(
        title="Évolution de l'Écartement de la Fissure avec Modélisation des Paliers",
        xaxis_title="Date",
        yaxis_title="Écartement de la Fissure (mm)",
        showlegend=True,
        xaxis=dict(showgrid=True, tickformat="%Y-%m"),
        yaxis=dict(showgrid=True),
        font=dict(color="black", size=20),
        autosize=True,
        width=None,
        height=None,
    )

    return fig


def adjust_new_series(df_dates_old, df_dates_new):
    """
    Ajuste les valeurs de la période 'new' en utilisant un modèle logarithmique basé sur les données 'old'.

    Arguments:
    df_dates_old : DataFrame contenant les dates et les valeurs de la période 'old'.
    df_dates_new : DataFrame contenant les dates et les valeurs de la période 'new'.

    Retourne:
    df_old : DataFrame avec les données de la période 'old'.
    df_new_adjusted : DataFrame avec les données ajustées de la période 'new'.
    """

    # Préparation des données pour l'ajustement
    df_old = df_dates_old.copy()
    df_new = df_dates_new.copy()

    # Modèle logarithmique du 13/04/2016 à la dernière date des mesures 'old'
    df_log = df_old[df_old.index >= "2016-04-13"]
    X_log = (df_log.index - df_log.index.min()).days.values.reshape(-1, 1)
    y_log = df_log["Bureau_old"].values
    model_log = LinearRegression().fit(
        X_log, np.exp(y_log)
    )  # Régression sur les valeurs exp
    y_log_pred = np.log(model_log.predict(X_log))

    # Prédiction pour la première date de la seconde phase
    X_new_start = np.array([(df_new.index[0] - df_log.index.min()).days]).reshape(-1, 1)
    predicted_start = (
        np.log(model_log.predict(X_new_start)[0]) - 0.103
    )  # Offset dû à la fluctuation de la première mesure

    # Ajustement proportionnel
    scaling_factor = 1.12  # Estimation due au changement de hauteur de prise de mesures
    df_new["Bureau_adjusted"] = predicted_start + scaling_factor * (
        df_new["Bureau"] - df_new["Bureau"].iloc[0]
    )

    return df_old, df_new


def dataviz_forecast(
    df_fissures,
    df_fissures_old,
    path_old="data/Fissures/Fissure_old.xlsx",
    path_new="data/Fissures/Fissure_2.xlsx",
):
    """
    Crée une figure avec les points des périodes 'old' et 'new' en utilisant les dates correctes
    provenant des fichiers .xlsx et les valeurs ajustées en utilisant la fonction adjust_new_series.
    Ajoute les prévisions avec le modèle linéaire, exponentiel et Prophet.
    Ajoute un point respectant toutes les contraintes d'intervalles (IP uniquement).
    """

    # Chargement des fichiers .xlsx pour récupérer les dates correctes
    global last_upper, ecart_last_upper, upper_max
    df_dates_old = pd.read_excel(
        path_old, sheet_name="Feuil3", usecols=["date", "bureau_old"]
    )
    df_dates_new = pd.read_excel(path_new, usecols=["Date", "Bureau\n(mm)"])

    # Mise en forme des DataFrames
    df_dates_old.rename(
        columns={"date": "Date", "bureau_old": "Bureau_old"}, inplace=True
    )
    df_dates_new.rename(
        columns={"Date": "Date", "Bureau\n(mm)": "Bureau"}, inplace=True
    )
    df_dates_old.set_index("Date", inplace=True)
    df_dates_new.set_index("Date", inplace=True)

    # Appel de la fonction d'ajustement
    df_old, df_new_adjusted = adjust_new_series(df_dates_old, df_dates_new)

    # Ajout de la colonne 'Date_ordinal' pour la régression linéaire
    df_new_adjusted["Date_ordinal"] = df_new_adjusted.index.map(pd.Timestamp.toordinal)

    # Création de la figure avec les points ajustés
    fig = go.Figure()

    # Ajout des points pour la période 'old'
    fig.add_trace(
        go.Scatter(
            x=df_old.index,
            y=df_old["Bureau_old"],
            mode="markers",
            marker=dict(color="gray", size=5),
            name="Période ancienne",
        )
    )

    # Ajout des points ajustés pour la période 'new'
    fig.add_trace(
        go.Scatter(
            x=df_new_adjusted.index,
            y=df_new_adjusted["Bureau_adjusted"],
            mode="markers",
            marker=dict(color="blue", size=5),
            name="Période récente ajustée",
        )
    )

    # Appel des modèles et récupération des IP
    fig, y_pred_linear, linear_intervals = linear_model_forecast(df_new_adjusted, fig)
    fig, y_pred_exponential, exp_intervals = exponential_model_forecast(
        df_new_adjusted, fig
    )

    # Combinaison des anciennes et nouvelles données pour Prophet
    df_combined = pd.concat(
        [
            df_old[["Bureau_old"]].rename(columns={"Bureau_old": "Bureau"}),
            df_new_adjusted[["Bureau_adjusted"]].rename(
                columns={"Bureau_adjusted": "Bureau"}
            ),
        ]
    )

    fig, yhat_prophet, prophet_intervals = prophet_forecast(df_combined, fig)

    # RMSE

    # Assurez-vous que yhat_prophet est un DataFrame ou une Série avec les dates en index
    yhat_prophet_df = pd.DataFrame({"yhat": yhat_prophet})
    yhat_prophet_df.index = pd.to_datetime(prophet_intervals["future_dates"])

    # Filtrer les valeurs de Prophet en utilisant uniquement les dates de df_new_adjusted, sans altérer les valeurs
    yhat_prophet_filtered = yhat_prophet_df.loc[df_new_adjusted.index]["yhat"]

    # Calcul des RMSE en utilisant les valeurs Prophet alignées sur les bonnes dates
    rmse_linear = np.sqrt(
        mean_squared_error(df_new_adjusted["Bureau_adjusted"], y_pred_linear)
    )
    rmse_exponential = np.sqrt(
        mean_squared_error(df_new_adjusted["Bureau_adjusted"], y_pred_exponential)
    )
    rmse_prophet = np.sqrt(
        mean_squared_error(df_new_adjusted["Bureau_adjusted"], yhat_prophet_filtered)
    )

    # Affichage des RMSE
    print(f"RMSE Linéaire: {rmse_linear:.4f}")
    print(f"RMSE Exponentiel: {rmse_exponential:.4f}")
    print(f"RMSE Prophet: {rmse_prophet:.4f}")

    # Localisation du point le plus tardif respectant toutes les conditions d'intervalles (IP)
    point_to_add = find_latest_intersection_direct(
        linear_intervals, exp_intervals, prophet_intervals
    )

    # Ajout du point à la figure si trouvé
    if point_to_add:
        fig.add_trace(
            go.Scatter(
                x=[point_to_add["date"]],
                y=[point_to_add["value"]],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Prévision la plus lointaine",
            )
        )

        # Calcul de l'écart avec le premier point de df_dates_new pour le point principal
        ecart = point_to_add["value"] - df_new_adjusted["Bureau_adjusted"].iloc[0]

        # Ajout de l'annotation pour le point principal
        fig.add_annotation(
            x=point_to_add["date"],
            y=point_to_add["value"],
            text=f"Date: {point_to_add['date'].strftime('%Y-%m-%d')}<br>Écart: {ecart:.2f} mm",
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=-40,
            font=dict(size=12, color="red"),
            align="left",
            bordercolor="red",
            borderwidth=1,
            bgcolor="cornsilk",
        )

        # Convertir les dates de prophet_intervals en format Timestamp si ce n'est pas déjà le cas
        prophet_dates = pd.to_datetime(prophet_intervals["future_dates"])

        # Trouver l'indice correspondant à la date du point déterminé pour chacun des modèles
        lin_idx = linear_intervals["future_dates"].index(point_to_add["date"])
        exp_idx = exp_intervals["future_dates"].index(point_to_add["date"])
        prophet_idx = prophet_dates.tolist().index(point_to_add["date"])

        # Déterminer les ordonnées maximale et minimale pour la date du point trouvé
        upper_max = max(
            linear_intervals["ip_upper"][lin_idx],
            exp_intervals["ip_upper"][exp_idx],
            prophet_intervals["ip_upper"][prophet_idx],
        )

        lower_min = min(
            linear_intervals["ip_lower"][lin_idx],
            exp_intervals["ip_lower"][exp_idx],
            prophet_intervals["ip_lower"][prophet_idx],
        )

        # Déterminer la couleur du point en fonction de l'intervalle auquel appartient lower_min
        if lower_min == linear_intervals["ip_lower"][lin_idx]:
            lower_color = "green"  # Couleur pour le modèle linéaire
        elif lower_min == exp_intervals["ip_lower"][exp_idx]:
            lower_color = "purple"  # Couleur pour le modèle exponentiel
        else:
            lower_color = "blue"  # Couleur pour le modèle Prophet

        # Déterminer la couleur du point en fonction de l'intervalle auquel appartient upper_max
        if upper_max == linear_intervals["ip_upper"][lin_idx]:
            upper_color = "green"  # Couleur pour le modèle linéaire
        elif upper_max == exp_intervals["ip_upper"][exp_idx]:
            upper_color = "purple"  # Couleur pour le modèle exponentiel
        else:
            upper_color = "blue"  # Couleur pour le modèle Prophet

        # Ajout des points aux ordonnées maximale et minimale avec les couleurs spécifiques
        fig.add_trace(
            go.Scatter(
                x=[point_to_add["date"]],
                y=[upper_max],
                mode="markers",
                marker=dict(color=upper_color, size=10, symbol="triangle-up"),
                name="Écart maximal",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[point_to_add["date"]],
                y=[lower_min],
                mode="markers",
                marker=dict(color=lower_color, size=10, symbol="triangle-down"),
                name="Écart minimal",
            )
        )

        # Calcul des écarts avec le premier point de df_dates_new
        ecart_upper = upper_max - df_new_adjusted["Bureau_adjusted"].iloc[0]
        ecart_lower = lower_min - df_new_adjusted["Bureau_adjusted"].iloc[0]

        # Ajout des annotations pour ces points
        fig.add_annotation(
            x=point_to_add["date"],
            y=upper_max,
            text=f"Écart: {ecart_upper:.2f} mm",
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=-40,
            font=dict(size=12, color=upper_color),
            align="left",
            bordercolor=upper_color,
            borderwidth=1,
            bgcolor="cornsilk",
        )

        fig.add_annotation(
            x=point_to_add["date"],
            y=lower_min,
            text=f"Écart: {ecart_lower:.2f} mm",
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=40,
            font=dict(size=12, color=lower_color),
            align="left",
            bordercolor=lower_color,
            borderwidth=1,
            bgcolor="cornsilk",
        )

        # Détermination de la dernière date disponible dans les intervalles des modèles
        last_date_lin = linear_intervals["future_dates"][-1]
        last_date_exp = exp_intervals["future_dates"][-1]
        last_date_prophet = prophet_dates[-1]

        # Prendre la date la plus ancienne parmi les dernières dates communes
        last_date = min(last_date_lin, last_date_exp, last_date_prophet)

        # Trouver les indices correspondants pour cette dernière date
        last_lin_idx = linear_intervals["future_dates"].index(last_date)
        last_exp_idx = exp_intervals["future_dates"].index(last_date)
        last_prophet_idx = prophet_dates.tolist().index(last_date)

        # Déterminer les ordonnées upper et lower à la dernière date
        last_upper = max(
            linear_intervals["ip_upper"][last_lin_idx],
            exp_intervals["ip_upper"][last_exp_idx],
            prophet_intervals["ip_upper"][last_prophet_idx],
        )

        last_lower = min(
            linear_intervals["ip_lower"][last_lin_idx],
            exp_intervals["ip_lower"][last_exp_idx],
            prophet_intervals["ip_lower"][last_prophet_idx],
        )

        # Calcul des écarts avec le premier point de df_new_adjusted pour la dernière date
        ecart_last_upper = last_upper - df_new_adjusted["Bureau_adjusted"].iloc[0]
        ecart_last_lower = last_lower - df_new_adjusted["Bureau_adjusted"].iloc[0]

        # Ajout des annotations pour les points à la dernière date
        fig.add_annotation(
            x=last_date,
            y=last_upper,
            text=f"Écart: {ecart_last_upper:.2f} mm",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=12, color="black"),
            align="left",
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
        )

        fig.add_annotation(
            x=last_date,
            y=last_lower,
            text=f"Écart: {ecart_last_lower:.2f} mm",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=40,
            font=dict(size=12, color="black"),
            align="left",
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
        )

        # Mise à jour de la plage de l'axe x (un mois après la date du point déterminé)
        xaxis_range_end = (point_to_add["date"] + pd.DateOffset(months=1)).strftime(
            "%Y-%m-%d"
        )
    else:
        xaxis_range_end = "2028-01-01"  # Valeur par défaut si aucun point n'est trouvé

    # Configuration de la figure avec RMSE comme sous-titre
    # Configuration de la figure avec les RMSE en sous-titre
    fig.update_layout(
        title="Modèles de prévision de l'écartement<br><sup>RMSE Modèle Linéaire: {:.4f}, RMSE Modèle Exponentiel: {:.4f}, "
        "RMSE Modèle Prophet: {:.4f}</sup>".format(
            rmse_linear, rmse_exponential, rmse_prophet
        ),
        font_size=20,
        xaxis_title="Date",
        xaxis=dict(
            range=["2023-12-03", xaxis_range_end],
            showgrid=True,
            tickformat="%Y-%m",
        ),
        yaxis_title="Écartement (mm)",
        yaxis=dict(
            range=[3.5, upper_max + 0.25],
            showgrid=True,
        ),
        legend=dict(orientation="h", x=0, y=-0.2),
        showlegend=True,
    )

    return fig


def find_latest_intersection_direct(linear_intervals, exp_intervals, prophet_intervals):
    """
    Trouve le point le plus tardif où il existe un y commun aux intervalles de prédiction des modèles linéaire,
    exponentiel et Prophet.

    Parameters:
        linear_intervals (dict): Dictionnaire contenant les IP du modèle linéaire.
        exp_intervals (dict): Dictionnaire contenant les IP du modèle exponentiel.
        prophet_intervals (dict): Dictionnaire contenant les IP du modèle Prophet.

    Returns:
        dict: Un dictionnaire contenant la date et la valeur du point trouvée ou None si aucun point ne correspond.
    """

    # Convertir les dates de Prophet en Timestamp pour assurer la compatibilité
    prophet_dates = pd.to_datetime(prophet_intervals["future_dates"]).to_pydatetime()
    prophet_dates = np.array([pd.Timestamp(date) for date in prophet_dates])

    # Convertir les dates des modèles linéaire et exponentiel en numpy arrays
    lin_dates = np.array(linear_intervals["future_dates"])
    exp_dates = np.array(exp_intervals["future_dates"])

    # Récupérer les valeurs des intervalles
    lin_low = np.array(linear_intervals["ip_lower"])
    lin_up = np.array(linear_intervals["ip_upper"])
    exp_low = np.array(exp_intervals["ip_lower"])
    exp_up = np.array(exp_intervals["ip_upper"])
    prophet_low = np.array(prophet_intervals["ip_lower"])
    prophet_up = np.array(prophet_intervals["ip_upper"])

    # Filtrer les dates de Prophet pour ne garder que celles qui sont dans lin_dates et exp_dates
    mask = np.isin(prophet_dates, lin_dates) & np.isin(prophet_dates, exp_dates)
    prophet_dates = prophet_dates[mask]
    prophet_low = prophet_low[mask]
    prophet_up = prophet_up[mask]

    # Vérifier les nouvelles dates communes
    common_indices = np.intersect1d(
        np.intersect1d(
            np.where(np.in1d(lin_dates, exp_dates))[0],
            np.where(np.in1d(lin_dates, prophet_dates))[0],
        ),
        np.where(np.in1d(exp_dates, prophet_dates))[0],
    )

    if len(common_indices) == 0:
        print("Aucune date commune trouvée entre les modèles après filtrage.")
        return None

    print(f"Nombre de dates communes trouvées : {len(common_indices)}")

    # Parcourir les indices des dates communes en commençant par la plus tardive
    for idx in reversed(common_indices):
        # Calculer les limites inférieures et supérieures à cette date
        lower_bound = max(lin_low[idx], exp_low[idx], prophet_low[idx])
        upper_bound = min(lin_up[idx], exp_up[idx], prophet_up[idx])

        # Afficher les détails de chaque étape pour vérifier l'intersection
        # print(f"Date: {lin_dates[idx]}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

        # Vérifier s'il existe une intersection (si le lower_bound est inférieur ou égal à l'upper_bound)
        if lower_bound <= upper_bound:
            # Retourner le point trouvé avec la valeur centrale de l'intersection
            date = lin_dates[idx]
            value = (
                lower_bound + upper_bound
            ) / 2  # Valeur au centre de l'intersection
            return {"date": date, "value": value}

    # Aucun point trouvé qui respecte toutes les conditions
    print("Aucune intersection trouvée sur les dates communes.")
    return None


def linear_model_forecast(df_new_adjusted, fig):
    """
    Ajoute le modèle linéaire avec les IC et IP à la figure.
    """
    X = sm.add_constant(df_new_adjusted["Date_ordinal"])
    y = df_new_adjusted["Bureau_adjusted"]
    linear_model = sm.OLS(y, X).fit()
    y_pred_linear = linear_model.predict(X)

    # Prévisions à 20 ans
    last_date_ordinal = df_new_adjusted["Date_ordinal"].max()
    future_dates_ordinal = np.arange(
        last_date_ordinal + 1, last_date_ordinal + 1 + 20 * 365
    )
    X_future = sm.add_constant(future_dates_ordinal)
    y_future_pred_linear = linear_model.predict(X_future)
    future_dates = [pd.Timestamp.fromordinal(date) for date in future_dates_ordinal]

    # Calcul de l'IC de Student
    predictions_linear = linear_model.get_prediction(X)
    prediction_summary_linear = predictions_linear.summary_frame(alpha=0.05)
    ci_lower_student_linear = prediction_summary_linear["mean_ci_lower"]
    ci_upper_student_linear = prediction_summary_linear["mean_ci_upper"]

    # Ajouter l'IC de Student en remplissant l'espace entre les courbes inférieure et supérieure
    fig.add_trace(
        go.Scatter(
            x=list(df_new_adjusted.index) + list(df_new_adjusted.index[::-1]),
            y=list(ci_lower_student_linear) + list(ci_upper_student_linear[::-1]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            mode="lines",
            line=dict(color="red", dash="dot"),
            name="IC Student Linéaire",
            showlegend=True,
        )
    )

    # Calcul des distances orthogonales pour l'IC effectif
    slope = linear_model.params[1]
    distances_orthogonales = (y - y_pred_linear) / np.sqrt(1 + slope**2)
    sorted_indices = np.argsort(np.abs(distances_orthogonales))
    num_points_to_keep = int(0.95 * len(distances_orthogonales))
    indices_to_keep = sorted_indices[:num_points_to_keep]
    distance_orthogonale_pos = np.max(distances_orthogonales[indices_to_keep])
    distance_orthogonale_neg = np.min(distances_orthogonales[indices_to_keep])

    # Ajouter l'IC effectif supérieur et inférieur
    fig.add_trace(
        go.Scatter(
            x=df_new_adjusted.index,
            y=y_pred_linear + distance_orthogonale_pos * np.sqrt(1 + slope**2),
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="IC Effectif à 95 % (Linéaire)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_new_adjusted.index,
            y=y_pred_linear + distance_orthogonale_neg * np.sqrt(1 + slope**2),
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="IC Effectif Inférieur (Linéaire)",
            showlegend=False,
        )
    )

    # Intervalles de prédiction (IP) pour la période de prévision
    n = len(X)
    x_mean = np.mean(df_new_adjusted["Date_ordinal"])
    var_res_linear = np.sum((y - y_pred_linear) ** 2) / (n - 2)
    var_pred_future_linear = var_res_linear * (
        1
        + 1 / n
        + ((future_dates_ordinal - x_mean) ** 2)
        / np.sum((df_new_adjusted["Date_ordinal"] - x_mean) ** 2)
    )
    ip_upper_future_linear = y_future_pred_linear + 1.96 * np.sqrt(
        var_pred_future_linear
    )
    ip_lower_future_linear = y_future_pred_linear - 1.96 * np.sqrt(
        var_pred_future_linear
    )

    # Ajouter l'IP linéaire avec remplissage
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(ip_upper_future_linear) + list(ip_lower_future_linear[::-1]),
            fill="toself",
            fillcolor="rgba(0, 255, 0, 0.2)",
            line=dict(color="rgba(0, 255, 0, 0)"),
            name="IP Linéaire",
            showlegend=True,
        )
    )

    # Ajout de la régression linéaire et des prévisions
    fig.add_trace(
        go.Scatter(
            x=df_new_adjusted.index,
            y=y_pred_linear,
            mode="lines",
            name="Régression Linéaire (OLS)",
            line=dict(color="green"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=y_future_pred_linear,
            mode="lines",
            name="Prévision linéaire",
            line=dict(color="green", dash="dot"),
            showlegend=False,
        )
    )

    return (
        fig,
        y_pred_linear,
        {
            "ip_lower": ip_lower_future_linear,
            "ip_upper": ip_upper_future_linear,
            "future_dates": future_dates,
        },
    )


def exponential_model_forecast(df_new_adjusted, fig):
    """
    Ajoute le modèle exponentiel avec les IC et IP à la figure.
    """
    X = sm.add_constant(df_new_adjusted["Date_ordinal"])
    y = df_new_adjusted["Bureau_adjusted"]
    y_log = np.log(y)

    # Ajustement du modèle exponentiel (régression linéaire sur les log)
    exponential_model = sm.OLS(y_log, X).fit()
    y_pred_exponential = np.exp(exponential_model.predict(X))

    # Prévisions exponentielles à 20 ans
    last_date_ordinal = df_new_adjusted["Date_ordinal"].max()
    future_dates_ordinal = np.arange(
        last_date_ordinal + 1, last_date_ordinal + 1 + 20 * 365
    )
    X_future = sm.add_constant(future_dates_ordinal)
    y_future_pred_exponential = np.exp(exponential_model.predict(X_future))
    future_dates = [
        pd.Timestamp.fromordinal(int(date)) for date in future_dates_ordinal
    ]

    # Calcul de l'IC de Student
    predictions_exponential = exponential_model.get_prediction(X)
    prediction_summary_exponential = predictions_exponential.summary_frame(alpha=0.05)
    ci_lower_student_exp = np.exp(prediction_summary_exponential["mean_ci_lower"])
    ci_upper_student_exp = np.exp(prediction_summary_exponential["mean_ci_upper"])

    # Ajouter l'IC de Student en remplissant l'espace entre les courbes inférieure et supérieure
    fig.add_trace(
        go.Scatter(
            x=list(df_new_adjusted.index) + list(df_new_adjusted.index[::-1]),
            y=list(ci_lower_student_exp) + list(ci_upper_student_exp[::-1]),
            fill="toself",
            fillcolor="rgba(128, 0, 128, 0.2)",
            mode="lines",
            line=dict(color="purple", dash="dot"),
            name="IC Student Exponentiel",
            showlegend=True,
        )
    )

    # Intervalles de prédiction (IP) pour la période de prévision
    n = len(X)
    x_mean = np.mean(df_new_adjusted["Date_ordinal"])
    var_res_exp = np.sum((y_log - np.log(y_pred_exponential)) ** 2) / (n - 2)
    var_pred_future_exp = var_res_exp * (
        1
        + 1 / n
        + ((future_dates_ordinal - x_mean) ** 2)
        / np.sum((df_new_adjusted["Date_ordinal"] - x_mean) ** 2)
    )
    ip_upper_future_exp = np.exp(
        np.log(y_future_pred_exponential) + 1.96 * np.sqrt(var_pred_future_exp)
    )
    ip_lower_future_exp = np.exp(
        np.log(y_future_pred_exponential) - 1.96 * np.sqrt(var_pred_future_exp)
    )

    # Ajouter l'IP exponentiel avec remplissage
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(ip_upper_future_exp) + list(ip_lower_future_exp[::-1]),
            fill="toself",
            fillcolor="rgba(128, 0, 128, 0.2)",
            line=dict(color="rgba(128, 0, 128, 0)"),
            name="IP Exponentiel",
            showlegend=True,
        )
    )

    # Ajout de la régression exponentielle et des prévisions
    fig.add_trace(
        go.Scatter(
            x=df_new_adjusted.index,
            y=y_pred_exponential,
            mode="lines",
            name="Modèle Exponentiel",
            line=dict(color="purple"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=y_future_pred_exponential,
            mode="lines",
            name="Prévision exponentielle",
            line=dict(color="purple", dash="dot"),
            showlegend=True,
        )
    )

    return (
        fig,
        y_pred_exponential,
        {
            "ip_lower": ip_lower_future_exp,
            "ip_upper": ip_upper_future_exp,
            "future_dates": future_dates,
        },
    )


def prophet_forecast(df_combined, fig):
    """
    Applique un modèle Prophet pour prévoir les fissures en utilisant le DataFrame combiné 'df_combined'.
    """
    df_combined = df_combined[df_combined.index >= "13-04-2016"]
    df_combined = df_combined.reset_index().rename(
        columns={"Date": "ds", "Bureau": "y"}
    )

    # Initialisation et ajustement du modèle Prophet
    model = Prophet(
        interval_width=0.95
    )  # , seasonality_mode='multiplicative', n_changepoints=20)
    # model.add_seasonality(name='palier', period=200, fourier_order=5)
    model.fit(df_combined)

    # Création des futures dates (20 ans à l'avance)
    future_dates = model.make_future_dataframe(periods=20 * 365)
    forecast = model.predict(future_dates)

    # Ajout des prévisions de Prophet à la figure
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Prévision Prophet",
            line=dict(color="blue", dash="dot"),
            showlegend=True,
        )
    )

    # Ajout de l'intervalle de confiance de Prophet
    fig.add_trace(
        go.Scatter(
            x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
            y=list(forecast["yhat_lower"]) + list(forecast["yhat_upper"][::-1]),
            fill="toself",
            fillcolor="rgba(0, 100, 250, 0.1)",
            line=dict(color="rgba(0, 100, 250, 0)"),
            name="IC/IP Prophet",
            showlegend=True,
        )
    )

    return (
        fig,
        forecast["yhat"],
        {
            "ip_lower": forecast["yhat_lower"].values,
            "ip_upper": forecast["yhat_upper"].values,
            "future_dates": forecast["ds"].values,
        },
    )
