import matplotlib
matplotlib.use("Agg")  # pour ne pas invoquer de fenêtre graphique sous Windows
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR
from features.selection import select_features_combined, random_search_selection
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr
from rich.console import Console
from collections import Counter

console = Console()

def train_and_evaluate_model(name, model, X_sel, y_clean, cv, combo_label):
    """
    Entraîne 'model' sur (X_sel, y_clean) avec cross-validation,
    calcule diverses métriques, ET génère une figure :
      - Sous-plot 0 : scatter (Mesuré vs Prévu) + regression + IC 95%
      - Sous-plot 1 : bar chart des coefficients (ou importances)

    On ajoute dans la text-box du premier sous-plot :
        - RMSE
        - R²
        - (RMSE / moyenne mesurée)*100 (%)
    """
    try:
        # n_jobs=1 pour éviter les warnings qui rendent la console illisible sous Windows
        y_pred_cv = cross_val_predict(model, X_sel, y_clean, cv=cv, n_jobs=1)
        model.fit(X_sel, y_clean)
    except Exception as e:
        console.print(f"[bold red]Erreur lors de l'entraînement du modèle {name} ({combo_label}): {e}[/bold red]")
        return {
            "Model": name, "RMSE": float('nan'), "MAPE (%)": float('nan'), "R2": float('nan'),
            "Adjusted_R2": float('nan'), "AIC": float('nan'), "BIC": float('nan'),
            "Num_Params": float('nan'), "Pearson_r": float('nan'),
            "p_value": float('nan'), "Top3": []
        }

    # ------------------------------------------------------------------
    # Aller chercher l'estimateur final s'il s'agit d'un Pipeline
    # ------------------------------------------------------------------
    if isinstance(model, Pipeline):
        final_est = model.steps[-1][1]  # ex: (StandardScaler(), Lasso()) => Lasso() est final
    else:
        final_est = model

    # Détermination du vecteur de coefficients (ou importances) :
    if hasattr(final_est, "coef_"):
        num_params = len(final_est.coef_)
        coef_array = final_est.coef_
    elif hasattr(final_est, "feature_importances_"):
        num_params = len(final_est.feature_importances_)
        coef_array = final_est.feature_importances_
    else:
        num_params = X_sel.shape[1]
        coef_array = [float('nan')] * X_sel.shape[1]

    # Tri selon valeur absolue, ordre décroissant
    coef_series = pd.Series(coef_array, index=X_sel.columns)
    coef_series_sorted = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)

    # On ne garde que le top 10 pour l'affichage
    top_10_series = coef_series_sorted.head(10)
    # Les 3 premières dans ce top 10 deviendront "Top3" pour le reporting
    top3 = top_10_series.head(3).index.tolist()

    # Calcul des métriques
    rmse_val = np.sqrt(mean_squared_error(y_clean, y_pred_cv))
    mape_val = mean_absolute_percentage_error(y_clean, y_pred_cv) * 100
    r2_val = r2_score(y_clean, y_pred_cv)
    n = len(y_clean)
    rss = np.sum((y_clean - y_pred_cv) ** 2)

    # AIC / BIC
    if rss > 0:
        aic = n * np.log(rss / n) + 2 * num_params
        bic = n * np.log(rss / n) + num_params * np.log(n)
    else:
        aic = float('-inf')
        bic = float('-inf')

    # R² ajusté
    if (n - num_params - 1) > 0:
        adj_r2 = 1 - (1 - r2_val) * (n - 1) / (n - num_params - 1)
    else:
        adj_r2 = float('nan')

    # Corrélation de Pearson
    r_val, p_val = pearsonr(y_clean, y_pred_cv)

    # Génération de la figure (scatter + bar chart)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Sous-plot 0 : scatter (Mesuré vs Prévu) + régression
    df_plot = pd.DataFrame({"Measured": y_clean, "Predicted": y_pred_cv})
    sns.regplot(data=df_plot, x="Measured", y="Predicted",
                ci=95, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}, ax=axs[0])
    axs[0].set_xlabel("Mesuré")
    axs[0].set_ylabel("Prévu")
    axs[0].set_title(f"{name} ({combo_label})")

    # Ligne 1:1
    min_val = min(df_plot["Measured"].min(), df_plot["Predicted"].min())
    max_val = max(df_plot["Measured"].max(), df_plot["Predicted"].max())
    axs[0].plot([min_val, max_val], [min_val, max_val], '--', color='gray', label="1:1 line")

    mean_measured = y_clean.mean()
    pct_rmse = (rmse_val / mean_measured) * 100 if mean_measured != 0 else np.nan

    txt = (f"RMSE: {rmse_val:.4f}\n"
           f"R²: {r2_val:.3f}\n"
           f"RMSE%: {pct_rmse:.1f}%")
    axs[0].text(0.05, 0.95, txt,
                transform=axs[0].transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axs[0].legend(loc="lower right")

    # Sous-plot 1 : bar chart des top 10 features
    bars = axs[1].barh(top_10_series.index, top_10_series.values, color='green', alpha=0.8)
    axs[1].invert_yaxis()

    # Si tout est à zéro (ou quasi), on force un xlim pour éviter un tracé "vide"
    max_abs = top_10_series.abs().max()
    if np.isnan(max_abs) or max_abs < 1e-12:
        axs[1].set_xlim(-0.01, 0.01)

    # On adapte le label de l'axe X en fonction du type d'estimateur
    if hasattr(final_est, "coef_"):
        axs[1].set_xlabel("Coefficient")
    elif hasattr(final_est, "feature_importances_"):
        axs[1].set_xlabel("Importance")
    else:
        axs[1].set_xlabel("Feature Score (?)")
    axs[1].set_title("Top features (ordre importance)")

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"model_{combo_label}_{name}.png")
    plt.savefig(fig_path)
    plt.close()

    return {
        "Model": name,
        "RMSE": rmse_val,
        "MAPE (%)": mape_val,
        "R2": r2_val,
        "Adjusted_R2": adj_r2,
        "AIC": aic,
        "BIC": bic,
        "Num_Params": num_params,
        "Pearson_r": r_val,
        "p_value": p_val,
        "Top3": top3
    }

def run_modeling(X, y, combo_label, cv, selection_method="combined", **sel_params):
    """
    Lance la modélisation sur plusieurs modèles (OLS, LASSO, etc.),
    + fait la figure de comparaison globale => models_comparison_{combo_label}.png
    """
    data = X.copy()
    data[y.name] = y
    data = data.dropna()
    X_clean = data.drop(columns=[y.name])
    y_clean = data[y.name]

    # 1) Sélection de features (ou non) selon selection_method
    if selection_method == "combined":
        if sel_params.get("use_random_search", False):
            # random_search_selection fait une random search sur RFE + stability
            best_params = random_search_selection(X_clean, y_clean, n_search=10)
            selected = select_features_combined(X_clean, y_clean,
                                                n_features_to_select=best_params[0],
                                                n_iter=50,
                                                sample_fraction=best_params[1],
                                                stability_threshold=best_params[2])
        else:
            selected = select_features_combined(X_clean, y_clean,
                                                n_features_to_select=sel_params.get("n_features_to_select", 20),
                                                n_iter=sel_params.get("n_iter", 50),
                                                sample_fraction=sel_params.get("sample_fraction", 0.75),
                                                stability_threshold=sel_params.get("stability_threshold", 0.5))
        X_sel = X_clean[selected]
    elif selection_method == "top10":
        def safe_corr(s):
            return 0 if s.std() == 0 else np.abs(np.corrcoef(s, y_clean)[0,1])
        corr_series = X_clean.apply(safe_corr)
        selected = corr_series.sort_values(ascending=False).head(10).index.tolist()
        X_sel = X_clean[selected]
    else:
        # Pas de sélection : on garde toutes les colonnes
        X_sel = X_clean.copy()

    # 2) On définit nos modèles
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, ARDRegression, BayesianRidge
    from sklearn.ensemble import RandomForestRegressor
    from catboost import CatBoostRegressor

    models = {
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "LASSO": make_pipeline(StandardScaler(), Lasso(alpha=1e-1, max_iter=200000)),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, max_iter=200000)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=200000)),
        "ARD": make_pipeline(StandardScaler(), ARDRegression()),
        "BayesianRidge": make_pipeline(StandardScaler(), BayesianRidge()),
        "RandomForest": make_pipeline(StandardScaler(),
                                      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        "CatBoost": make_pipeline(StandardScaler(),
                                  CatBoostRegressor(verbose=0, random_state=42, thread_count=1))
    }

    # 3) Pour chaque modèle, on appelle train_and_evaluate_model
    results = []
    for name, model in models.items():
        r = train_and_evaluate_model(name, model, X_sel, y_clean, cv, combo_label)
        results.append(r)

    # 4) On compile un DataFrame de performances, on fait la figure de comparaison globale
    performance_df = pd.DataFrame(results).sort_values(by="RMSE")

    # On regarde aussi les top3 features de chaque modèle pour faire un histogramme
    all_top3 = []
    for r in results:
        all_top3.extend(r["Top3"])
    counter_top3 = Counter(all_top3)
    top_10 = counter_top3.most_common(10)
    labels = [t[0] for t in top_10]
    vals = [t[1] for t in top_10]

    fig_global, (ax1_global, ax2_global) = plt.subplots(1, 2, figsize=(16, 6))

    # Sous-plot 1 : bar chart des RMSE par modèle
    ax1_global.bar(performance_df["Model"], performance_df["RMSE"], color="skyblue")
    ax1_global.set_xlabel("Modèle")
    ax1_global.set_ylabel("RMSE")
    ax1_global.set_title(f"Comparaison des RMSE ({combo_label}, Target: {y_clean.name})")
    plt.setp(ax1_global.get_xticklabels(), rotation=45, ha="right")

    # Sous-plot 2 : bar chart des top3 features (compteur)
    ax2_global.bar(labels, vals, color="lightgreen")
    ax2_global.set_xlabel("Variable explicative")
    ax2_global.set_ylabel("Fréquence (Top3)")
    ax2_global.set_title("Répartition des top 3 features")
    plt.setp(ax2_global.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    global_fig_path = os.path.join(RESULTS_DIR, f"models_comparison_{combo_label}.png")
    plt.savefig(global_fig_path)
    plt.close()

    console.print(f"\n[bold magenta]----- Classement des modèles ({combo_label}, Target: {y_clean.name}) -----[/bold magenta]")
    console.print(performance_df[["Model", "RMSE", "MAPE (%)", "R2", "Adjusted_R2", "AIC", "BIC", "Num_Params", "Pearson_r", "p_value"]])

    return performance_df
