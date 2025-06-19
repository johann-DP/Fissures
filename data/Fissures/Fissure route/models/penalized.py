import os
import pandas as pd
from models.training import train_and_evaluate_model
from sklearn.linear_model import LassoCV, ElasticNetCV
from config import RESULTS_DIR
from rich.console import Console
import matplotlib.pyplot as plt
from collections import Counter

console = Console()

def run_modeling_all_features_penalized(X, y, combo_label, cv):
    """
    Conserver TOUTES les features et entraîner LassoCV et ElasticNetCV.
    Génère la figure "models_comparison_AllPenalized_{combo_label}.png"
    en affichant (à gauche) la RMSE par modèle, et (à droite) le top 10
    des features les plus fréquentes dans le Top3.
    """
    X_clean = X.dropna()
    y_clean = y.dropna()
    data = X_clean.copy()
    data[y.name] = y_clean
    data = data.dropna()
    y_clean = data[y.name]
    X_clean = data.drop(columns=[y.name])

    models = {
        "LassoCV": LassoCV(cv=5, random_state=42),
        "ElasticNetCV": ElasticNetCV(cv=5, random_state=42)
    }
    results = []
    for name, model in models.items():
        r = train_and_evaluate_model(name, model, X_clean, y_clean, cv, combo_label + "_AllPenalized")
        results.append(r)

    performance_df = pd.DataFrame(results).sort_values(by="RMSE")

    # Récupération des top3 de chaque modèle
    all_top3 = []
    for r in results:
        all_top3.extend(r["Top3"])
    counter_top3 = Counter(all_top3)
    top_10 = counter_top3.most_common(10)

    labels = [t[0] for t in top_10]
    vals = [t[1] for t in top_10]

    fig_global, (ax1_global, ax2_global) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart des RMSE
    ax1_global.bar(performance_df["Model"], performance_df["RMSE"], color="skyblue")
    ax1_global.set_xlabel("Modèle")
    ax1_global.set_ylabel("RMSE")
    ax1_global.set_title(f"Comparaison des RMSE (AllFeatures Penalisé, {combo_label})")
    # Faire pivoter le nom des modèles
    plt.setp(ax1_global.get_xticklabels(), rotation=45, ha="right")

    # Bar chart Top 10
    ax2_global.bar(labels, vals, color="lightgreen")
    ax2_global.set_xlabel("Variable explicative")
    ax2_global.set_ylabel("Fréquence (Top3)")
    ax2_global.set_title("Répartition des top 3 features (Top 10)")
    plt.setp(ax2_global.get_xticklabels(), rotation=45, ha="right")

    # Supprimer l'encadré "La variable la plus fréquente..." pour ne pas masquer
    # => on n'ajoute pas de text box ici

    plt.tight_layout()
    global_fig_path = os.path.join(RESULTS_DIR, f"models_comparison_AllPenalized_{combo_label}.png")
    plt.savefig(global_fig_path)
    plt.close()

    return performance_df

def run_modeling_all_features_simple(X, y, combo_label, cv):
    """
    Conserver TOUTES les features et entraîner Ridge et Lasso (classiques).
    Génère la figure "models_comparison_AllSimple_{combo_label}.png".
    """
    X_clean = X.dropna()
    y_clean = y.dropna()
    data = X_clean.copy()
    data[y.name] = y_clean
    data = data.dropna()
    y_clean = data[y.name]
    X_clean = data.drop(columns=[y.name])

    from sklearn.linear_model import Ridge, Lasso
    models = {
        "Ridge": Ridge(alpha=1.0, max_iter=10000),
        "Lasso": Lasso(alpha=1e-3, max_iter=10000)
    }

    results = []
    for name, model in models.items():
        r = train_and_evaluate_model(name, model, X_clean, y_clean, cv, combo_label + "_AllSimple")
        results.append(r)

    performance_df = pd.DataFrame(results).sort_values(by="RMSE")

    # Récupération des top3
    all_top3 = []
    for r in results:
        all_top3.extend(r["Top3"])
    counter_top3 = Counter(all_top3)
    top_10 = counter_top3.most_common(10)
    labels = [t[0] for t in top_10]
    vals = [t[1] for t in top_10]

    fig_global, (ax1_global, ax2_global) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart des RMSE
    ax1_global.bar(performance_df["Model"], performance_df["RMSE"], color="skyblue")
    ax1_global.set_xlabel("Modèle")
    ax1_global.set_ylabel("RMSE")
    ax1_global.set_title(f"Comparaison des RMSE (AllFeatures Simple, {combo_label})")
    plt.setp(ax1_global.get_xticklabels(), rotation=45, ha="right")

    # Bar chart Top 10
    ax2_global.bar(labels, vals, color="lightgreen")
    ax2_global.set_xlabel("Variable explicative")
    ax2_global.set_ylabel("Fréquence (Top3)")
    ax2_global.set_title("Répartition des top 3 features")
    plt.setp(ax2_global.get_xticklabels(), rotation=45, ha="right")

    # Pas de textbox "variable la plus fréquente"

    plt.tight_layout()
    global_fig_path = os.path.join(RESULTS_DIR, f"models_comparison_AllSimple_{combo_label}.png")
    plt.savefig(global_fig_path)
    plt.close()

    return performance_df
