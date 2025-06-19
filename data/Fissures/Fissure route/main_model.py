import os
import pickle
import time
from config import RESULTS_DIR
from data_route.preprocessing import (
    load_mur_data,
    load_weather_data,
    load_or_build_hourly_features,
    load_or_build_aggregated_features
)
from models import training, penalized
from visualization import plots, stats
from sklearn.model_selection import KFold
from rich.console import Console
from sklearn.exceptions import ConvergenceWarning
import warnings
from presentation import ppt_generator

console = Console()

# -------------------------------------------------------------------
# On ignore les ConvergenceWarning pour ne plus voir "Objective did not converge"
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------------------------------------------------
# Paramètres globaux
# -------------------------------------------------------------------
USE_PARALLEL = True

##################### A COMMENTER POUR DEMARRER A ZERO ! ############
# START = 12  # Étape à laquelle on démarre (8 => on refait de 8 à 12)
#####################################################################

# Nom du fichier où on enregistre le "checkpoint" (étape atteinte)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "checkpoint.pkl")

# Fichier où on stocke toutes les variables produites par chaque étape
PIPELINE_RESULTS_FILE = os.path.join(RESULTS_DIR, "pipeline_results.pkl")

# Dictionnaire global pour stocker/recharger tous les résultats (perf_xxx, etc.)
pipeline_results = {}


def load_checkpoint():
    """Charge le numéro d'étape actuel depuis un fichier 'checkpoint.pkl' (ou 0 s'il n'existe pas)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            chk = pickle.load(f)
        return chk
    else:
        return 0


def save_checkpoint(step):
    """Sauvegarde le numéro d'étape 'step' dans 'checkpoint.pkl'."""
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(step, f)


def load_pipeline_results():
    """Charge le dictionnaire pipeline_results depuis le fichier pickle, s'il existe."""
    global pipeline_results
    if os.path.exists(PIPELINE_RESULTS_FILE):
        with open(PIPELINE_RESULTS_FILE, "rb") as f:
            pipeline_results = pickle.load(f)
    else:
        pipeline_results = {}


def save_pipeline_results():
    """Sauvegarde le dictionnaire pipeline_results dans le fichier pickle."""
    with open(PIPELINE_RESULTS_FILE, "wb") as f:
        pickle.dump(pipeline_results, f)


def run_if_needed(current_step, checkpoint, label, func):
    """
    Exécute la fonction 'func' si 'checkpoint' < current_step ; sinon on passe.
    - current_step : numéro d'étape (int)
    - checkpoint : étape déjà validée
    - label : libellé pour la console
    - func() : fonction à exécuter si on n'a pas déjà validé l'étape

    Retourne le nouveau checkpoint (inchangé si on passe, = current_step si on exécute).
    """
    if checkpoint >= current_step:
        console.print(
            f"[bold blue]Étape {current_step} - {label} déjà validée (checkpoint={checkpoint}), on passe.[/bold blue]"
        )
        return checkpoint
    else:
        console.print(f"\n[bold underline]Étape {current_step} - {label}[/bold underline]")
        start_local = time.time()
        func()  # on exécute la fonction
        elapsed = time.time() - start_local
        console.print(f"[green]Étape {current_step} terminée en {elapsed:.2f} secondes.[/green]")
        save_checkpoint(current_step)
        return current_step


def partially_reset_checkpoint(target_step):
    """
    Permet de réinitialiser manuellement le checkpoint à 'target_step'.
    Exemple d'usage : on définit START=8, on appelle partiellement
    le script en redémarrant à l'étape 8.
    """
    global checkpoint
    if checkpoint > target_step:
        console.print(
            f"[bold yellow]Réinitialisation partielle du checkpoint : de {checkpoint} à {target_step}[/bold yellow]"
        )
        save_checkpoint(target_step)
        checkpoint = target_step
    else:
        console.print(
            f"[bold cyan]Checkpoint actuel ({checkpoint}) est déjà <= {target_step}. Pas de réinitialisation.[/bold cyan]"
        )


# -------------------------------------------------------------------
# Début du script principal
# -------------------------------------------------------------------
console.print("[bold underline blue]Début de la modélisation (main_model.py)[/bold underline blue]")

# 1) Chargement des données brutes (mur + météo)
console.print("[bold]Chargement des données mur et météo...[/bold]")

# ATTENTION : on suppose que load_weather_data renvoie (df_weather, vars_a_retenir, weather_file_name)
# => Vous devrez adapter la signature de load_weather_data dans preprocessing.py
df_mur = load_mur_data()
df_weather, vars_a_retenir, weather_file_name = load_weather_data(df_mur)
console.print("[bold green]Données brutes chargées.[/bold green]")

# 2) Gestion des warnings de convergence
original_showwarning = warnings.showwarning
convergence_messages = set()


def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, ConvergenceWarning):
        convergence_messages.add(str(message))
    else:
        original_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = custom_showwarning

# 3) Lecture du checkpoint et du dictionnaire global
checkpoint = load_checkpoint()
load_pipeline_results()

# 4) Vérif si le nom du fichier xls a changé OU si la dernière date dans mur_route.xlsx a changé
current_mur_last_date = df_mur["date"].max()  # Ou "measurement_time" selon vos colonnes
old_mur_last_date = pipeline_results.get("last_mur_date", None)

old_weather_file = pipeline_results.get("last_weather_file", None)

data_changed = False
if old_mur_last_date is not None and old_mur_last_date != current_mur_last_date:
    data_changed = True
if old_weather_file is not None and old_weather_file != weather_file_name:
    data_changed = True

if data_changed:
    console.print(
        "[bold yellow]Le fichier xls ou la date mur_route.xlsx a changé => reset checkpoint à 0[/bold yellow]"
    )
    checkpoint = 0
    save_checkpoint(0)

pipeline_results["last_mur_date"] = current_mur_last_date
pipeline_results["last_weather_file"] = weather_file_name
save_pipeline_results()

# 5) Possibilité de redémarrer partiellement depuis l'étape START (si vous l'avez décommenté)
if "START" in globals():
    partially_reset_checkpoint(START - 1)

# 6) Construction (ou chargement depuis cache) des features horaires
console.print("[bold]Construction/Chargement des features horaires...[/bold]")
df_features, hourly_updated = load_or_build_hourly_features(
    df_mur, df_weather, vars_a_retenir, RESULTS_DIR
)
X_hourly = df_features.drop(columns=['date', 'inch', 'delta_inch'])
y_abs = df_features['inch']
y_delta = df_features['delta_inch']

# 7) Construction (ou chargement) des features agrégées
console.print("[bold]Construction/Chargement des features agrégées...[/bold]")
df_agg_features, agg_updated = load_or_build_aggregated_features(
    df_mur, df_weather, vars_a_retenir, RESULTS_DIR
)
X_agg = df_agg_features.drop(columns=['date', 'inch', 'delta_inch'])
y_abs_agg = df_agg_features['inch']
y_delta_agg = df_agg_features['delta_inch']

# 8) Définir la validation croisée
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# -------------------------------------------------------------------
# Étape 1 - Modélisation: Hourly_Absolute
# -------------------------------------------------------------------
def step1():
    result = training.run_modeling(
        X_hourly, y_abs, "Hourly_Absolute", cv,
        selection_method="combined",
        use_random_search=True
    )
    pipeline_results["perf_hourly_abs"] = result
    save_pipeline_results()


checkpoint = run_if_needed(1, checkpoint, "Modélisation: Hourly_Absolute", step1)


# -------------------------------------------------------------------
# Étape 2 - Modélisation: Hourly_Delta
# -------------------------------------------------------------------
def step2():
    result = training.run_modeling(
        X_hourly, y_delta, "Hourly_Delta", cv,
        selection_method="combined",
        use_random_search=True
    )
    pipeline_results["perf_hourly_delta"] = result
    save_pipeline_results()


checkpoint = run_if_needed(2, checkpoint, "Modélisation: Hourly_Delta", step2)


# -------------------------------------------------------------------
# Étape 3 - Modélisation: Aggregated_Absolute
# -------------------------------------------------------------------
def step3():
    result = training.run_modeling(
        X_agg, y_abs_agg, "Aggregated_Absolute", cv,
        selection_method="combined",
        use_random_search=True
    )
    pipeline_results["perf_agg_abs"] = result
    save_pipeline_results()


checkpoint = run_if_needed(3, checkpoint, "Modélisation: Aggregated_Absolute", step3)


# -------------------------------------------------------------------
# Étape 4 - Modélisation: Aggregated_Delta
# -------------------------------------------------------------------
def step4():
    result = training.run_modeling(
        X_agg, y_delta_agg, "Aggregated_Delta", cv,
        selection_method="combined",
        use_random_search=True
    )
    pipeline_results["perf_agg_delta"] = result
    save_pipeline_results()


checkpoint = run_if_needed(4, checkpoint, "Modélisation: Aggregated_Delta", step4)


# -------------------------------------------------------------------
# Étape 5 - Hourly Absolute : ALL + Penalized
# -------------------------------------------------------------------
def step5():
    result = penalized.run_modeling_all_features_penalized(X_hourly, y_abs, "Hourly_Absolute", cv)
    pipeline_results["perf_hourly_abs_penalized"] = result
    save_pipeline_results()
    # On fait un petit récap, comme demandé
    console.print(f"[bold magenta]----- Classement (Hourly_Absolute, ALL + Penalized) -----[/bold magenta]")
    console.print(result[["Model", "RMSE", "MAPE (%)", "R2", "Adjusted_R2", "Pearson_r", "p_value"]])


checkpoint = run_if_needed(5, checkpoint, "Hourly Absolute : ALL + Penalized", step5)


# -------------------------------------------------------------------
# Étape 6 - Hourly Absolute : ALL + Ridge/Lasso
# -------------------------------------------------------------------
def step6():
    result = penalized.run_modeling_all_features_simple(X_hourly, y_abs, "Hourly_Absolute", cv)
    pipeline_results["perf_hourly_abs_simple"] = result
    save_pipeline_results()
    console.print(f"[bold magenta]----- Classement (Hourly_Absolute, ALL + Ridge/Lasso) -----[/bold magenta]")
    console.print(result[["Model", "RMSE", "MAPE (%)", "R2", "Adjusted_R2", "Pearson_r", "p_value"]])


checkpoint = run_if_needed(6, checkpoint, "Hourly Absolute : ALL + Ridge/Lasso", step6)


# -------------------------------------------------------------------
# Étape 7 - Hourly Delta : ALL + Penalized
# -------------------------------------------------------------------
def step7():
    result = penalized.run_modeling_all_features_penalized(X_hourly, y_delta, "Hourly_Delta", cv)
    pipeline_results["perf_hourly_delta_penalized"] = result
    save_pipeline_results()
    console.print(f"[bold magenta]----- Classement (Hourly_Delta, ALL + Penalized) -----[/bold magenta]")
    console.print(result[["Model", "RMSE", "MAPE (%)", "R2", "Adjusted_R2", "Pearson_r", "p_value"]])


checkpoint = run_if_needed(7, checkpoint, "Hourly Delta : ALL + Penalized", step7)


# -------------------------------------------------------------------
# Étape 8 - Hourly Delta : ALL + Ridge/Lasso
# -------------------------------------------------------------------
def step8():
    result = penalized.run_modeling_all_features_simple(X_hourly, y_delta, "Hourly_Delta", cv)
    pipeline_results["perf_hourly_delta_simple"] = result
    save_pipeline_results()
    console.print(f"[bold magenta]----- Classement (Hourly_Delta, ALL + Ridge/Lasso) -----[/bold magenta]")
    console.print(result[["Model", "RMSE", "MAPE (%)", "R2", "Adjusted_R2", "Pearson_r", "p_value"]])


checkpoint = run_if_needed(8, checkpoint, "Hourly Delta : ALL + Ridge/Lasso", step8)


# -------------------------------------------------------------------
# Étape 9 - Figure d'évolution de l'inclinaison
# -------------------------------------------------------------------
def step9():
    console.print("\n[bold underline]Génération de la figure d'évolution de l'inclinaison[/bold underline]")
    df_date = df_mur[['measurement_time', 'inch']].dropna().copy()
    df_date['time_numeric'] = df_date['measurement_time'].apply(lambda x: x.timestamp())
    plots.plot_evolution_inclinaison(df_date, os.path.join(RESULTS_DIR, "evolution_inclinaison.png"))


checkpoint = run_if_needed(9, checkpoint, "Figure d'évolution de l'inclinaison", step9)


# -------------------------------------------------------------------
# Étape 10 - Tests statistiques
# -------------------------------------------------------------------
def step10():
    console.print("\n[bold underline]Tests statistiques sur l'inclinaison[/bold underline]")
    df_date = df_mur[['measurement_time', 'inch']].dropna().copy()
    df_date['time_numeric'] = df_date['measurement_time'].apply(lambda x: x.timestamp())
    lr_res = stats.global_regression_stats(df_date)
    mk_res = stats.mann_kendall_test(df_date)
    stats.local_regression_analysis(df_date, window_size=5)
    # Comparatif résumé (Mann-Kendall / Spearman / LR)
    stats.compare_tests(df_date, lr_res, mk_res)


checkpoint = run_if_needed(10, checkpoint, "Tests statistiques", step10)


# -------------------------------------------------------------------
# Étape 11 - Dataviz Aggregated_Absolute
# -------------------------------------------------------------------
def step11():
    console.print("\n[bold underline]Dataviz features explicatives (Aggregated_Absolute)[/bold underline]")
    if "perf_agg_abs" not in pipeline_results:
        console.print("[bold red]Impossible de faire la Dataviz : perf_agg_abs indisponible.[/bold red]")
        console.print("[bold red]Soit l'étape 3 n'a pas été exécutée, soit pipeline_results est incomplet.[/bold red]")
        return

    perf_agg_abs = pipeline_results["perf_agg_abs"]
    best_agg_abs_model = perf_agg_abs.iloc[0]
    best_features = best_agg_abs_model["Top3"]

    if best_features:
        available_features = [f for f in best_features if f in df_agg_features.columns]
        if available_features:
            plots.plot_explanatory_evolution(
                df_agg_features,
                df_agg_features['date'],
                available_features,
                "Aggregated_Absolute",
                RESULTS_DIR
            )
            console.print("[bold green]Dataviz (Aggregated_Absolute) générée.[/bold green]")
        else:
            console.print("[bold red]Aucune feature Top3 trouvée dans df_agg_features.[/bold red]")
    else:
        console.print("[bold red]Aucune feature Top3 enregistrée pour Aggregated_Absolute.[/bold red]")


checkpoint = run_if_needed(11, checkpoint, "Dataviz Aggregated_Absolute", step11)


# -------------------------------------------------------------------
# Étape 12 - Génération du PPT final
# -------------------------------------------------------------------
def step12():
    console.print("\n[bold underline]Génération de la présentation PowerPoint[/bold underline]")
    ppt_path = os.path.join(os.path.dirname(RESULTS_DIR), "Synthese_Modelisation.pptx")
    ppt_generator.create_ppt_from_results(pipeline_results, RESULTS_DIR, ppt_path)


checkpoint = run_if_needed(12, checkpoint, "Génération du PPT final", step12)

# -------------------------------------------------------------------
# Fin de script : avertissements de convergence, etc.
# -------------------------------------------------------------------
if checkpoint >= 12:
    if convergence_messages:
        console.print("[bold red]Attention :[/bold red] Certains modèles n'ont pas convergé. Messages :")
        for msg in convergence_messages:
            console.print(f"- {msg}")
    console.print("[bold green]Script terminé. Résultats et présentation générés.[/bold green]")
else:
    console.print(
        "[bold cyan]Script interrompu avant la fin ? Re-lancez pour continuer les étapes suivantes.[/bold cyan]"
    )
