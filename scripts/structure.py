from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def chargement_donnees(chemin):
    """Charge les données depuis deux fichiers Excel."""
    df = pd.read_excel(f"{chemin}Fissure_2.xlsx")
    df.columns = [
        "Date",
        "Bureau",
        "Variation Bureau",
        "Mur extérieur",
        "Variation Mur",
    ]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    df_old = pd.read_excel(f"{chemin}Fissure_old.xlsx", sheet_name="Feuil3")
    df_old.columns = ["date", "bureau_old"]
    df_old["Date"] = pd.to_datetime(df_old["date"])
    df_old["Bureau_old"] = df_old["bureau_old"].astype(float)

    return df, df_old


def preprocessing_old_new(df_fissures, df_fissures_old):
    df_old = df_fissures_old
    df_new = df_fissures

    # Filtrer et appliquer la régression linéaire sur les données de la première phase
    df_old_filtered = df_old[df_old["Date"] >= "2015-01-01"]
    X_old_filtered = (
        df_old_filtered["Date"] - df_old_filtered["Date"].min()
    ).dt.days.values.reshape(-1, 1)
    y_old_filtered = df_old_filtered["Bureau_old"].values
    model = LinearRegression().fit(X_old_filtered, y_old_filtered)

    # Prédiction pour le début de la nouvelle phase
    X_new_start = np.array(
        [(df_new["Date"].iloc[0] - df_old_filtered["Date"].min()).days]
    ).reshape(-1, 1)
    predicted_start = model.predict(X_new_start)[0] - 0.103

    # Ajustement des nouvelles données
    scaling_factor = 1.16
    df_new["Bureau_new_adjusted"] = predicted_start + scaling_factor * (
        df_new["Bureau"] - df_new["Bureau"].iloc[0]
    )

    # Concaténer les séries de données
    df_combined = pd.concat(
        [
            df_old[["Date", "Bureau_old"]].rename(columns={"Bureau_old": "Bureau"}),
            df_new[["Date", "Bureau_new_adjusted"]].rename(
                columns={"Bureau_new_adjusted": "Bureau"}
            ),
        ]
    )
    df_combined = (
        df_combined.drop_duplicates(subset="Date").set_index("Date").sort_index()
    )

    # Division des données en deux phases
    df_combined_old = df_combined.loc[df_old["Date"].min() : df_old["Date"].max()]
    df_combined_new = df_combined.loc[df_new["Date"].min() : df_new["Date"].max()]

    # Calcul des points de rupture manuels
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
    manual_breaks_new_dates = [
        "2023-12-01",
        "2024-01-15",
        "2024-02-01",
        "2024-06-01",
        "2024-07-05",
        "2024-09-01",
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

    # Générer les paliers pour la première phase
    paliers_old = []
    for i in range(len(manual_breaks_old) - 1):
        start = manual_breaks_old[i]
        end = manual_breaks_old[i + 1]
        if i % 2 != 0:  # Segments plats (pente nulle)
            avg_value = np.mean(df_combined_old["Bureau"][start : end + 1])
            paliers_old.append(
                [df_combined_old.index[start], df_combined_old.index[end], avg_value]
            )

    # Générer les paliers pour la deuxième phase
    paliers_new = []
    for i in range(len(manual_breaks_new) - 1):
        start = manual_breaks_new[i]
        end = manual_breaks_new[i + 1]
        if i % 2 == 0:  # Segments plats (pente nulle)
            avg_value = np.mean(df_combined_new["Bureau"][start : end + 1])
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

    return df_paliers_old, df_paliers_new


def structure_dataviz(df_paliers_old, df_paliers_new):
    df_paliers_combined = pd.concat([df_paliers_old, df_paliers_new])

    # Ajout des colonnes supplémentaires pour l'analyse structurelle
    construction_year = 1959
    df_paliers_combined["Building_Age"] = (
        df_paliers_combined["Début"].dt.year - construction_year
    )
    df_paliers_combined["Building_Age"] = df_paliers_combined["Building_Age"].apply(
        lambda x: max(x, 1)
    )

    # Les propriétés IPN sont constantes, nous les stockons dans des variables séparées
    b_aile, h_aile, d_aile = 0.15, 0.01, 0.15
    I_aile = 2 * ((b_aile * h_aile**3) / 12 + (b_aile * h_aile) * d_aile**2)
    b_central, h_central = 0.30, 0.015
    I_central = (b_central * h_central**3) / 12
    IPN_Moment_Inertia = I_aile + I_central

    E_acier = 210 * 10**9
    IPN_Rigidite_Flexion = E_acier * IPN_Moment_Inertia
    IPN_Section = b_aile * h_aile + b_central * h_central
    IPN_Stress_Factor = 1 / IPN_Section

    # Les autres colonnes variables
    df_paliers_combined["IPN_Age"] = df_paliers_combined["Début"].dt.year - 2016
    df_paliers_combined["IPN_Age"] = df_paliers_combined["IPN_Age"].apply(
        lambda x: max(x, 0)
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

    # Afficher les valeurs constantes une seule fois
    print(f"IPN_Moment_Inertia: {IPN_Moment_Inertia}")
    print(f"IPN_Rigidite_Flexion: {IPN_Rigidite_Flexion}")
    print(f"IPN_Section: {IPN_Section}")
    print(f"IPN_Stress_Factor: {IPN_Stress_Factor}")

    return df_paliers_combined


# Chargement et préparation des données
chemin = "data/Fissures/"
df_fissures, df_fissures_old = chargement_donnees(chemin)
df_paliers_old, df_paliers_new = preprocessing_old_new(df_fissures, df_fissures_old)
df_paliers_combined = structure_dataviz(df_paliers_old, df_paliers_new)

# Affichage du DataFrame final
print(df_paliers_combined)


def plot_paliers_with_dual_axis(df_paliers_combined):
    """Trace l'évolution des paliers avec deux échelles : valeurs moyennes des paliers et durée des paliers."""

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Subplot 1 : Évolution des paliers (axe des valeurs moyennes)
    for _, row in df_paliers_combined.iterrows():
        ax1.hlines(
            y=row["Valeur moyenne"],
            xmin=row["Début"],
            xmax=row["Fin"],
            color="blue",
            linewidth=2,
        )

    # Labels et titre pour le premier axe (valeurs moyennes)
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Valeur moyenne", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("Évolution des paliers et de la durée des paliers")
    ax1.grid(True)

    # Création du deuxième axe pour la durée des paliers
    ax2 = ax1.twinx()  # Partager le même axe des X

    # Calcul de la date centrale entre 'Début' et 'Fin'
    df_paliers_combined["Date_Centrale"] = (
        df_paliers_combined["Début"]
        + (df_paliers_combined["Fin"] - df_paliers_combined["Début"]) / 2
    )

    # Tracé de l'évolution de la durée des paliers (sans relier les points)
    ax2.scatter(
        df_paliers_combined["Date_Centrale"],
        df_paliers_combined["Palier_Duration"],
        color="green",
        marker="o",
        label="Durée des paliers",
    )

    # Labels et titre pour le deuxième axe (durée des paliers)
    ax2.set_ylabel("Durée des paliers (en jours)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Formatage des dates sur l'axe des X
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def plot_standardized_boxplots(df_paliers_combined):
    """Trace les boxplots des colonnes standardisées et un pairplot des variables sélectionnées."""

    # Sélectionner les colonnes à standardiser
    columns_to_standardize = [
        "Tassement_Differentiel_IPN",
        "Tassement_Mur",
        "Tassement_Colline",
        "Corrosion_Index",
        "Fatigue_Factor",
        "Degradation_Factor",
    ]

    # Standardiser les colonnes à l'aide de StandardScaler
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df_paliers_combined[columns_to_standardize]),
        columns=columns_to_standardize,
    )

    # Création de la figure pour les boxplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Subplot 1 : Boxplots des colonnes standardisées (utilisation de seaborn)
    sns.violinplot(data=df_standardized[columns_to_standardize], ax=ax1)
    ax1.set_title("Boxplots des colonnes standardisées")
    ax1.set_ylabel("Valeurs standardisées (Z-score)")
    ax1.tick_params(axis="x", rotation=45)

    # Ajustement de la mise en page du premier plot
    plt.tight_layout()

    # Affichage du premier subplot
    plt.show()


def plot_filtered_scatterplot_grid(df_paliers_combined, threshold=0.99):
    """Trace une grille de scatterplots pour les paires de colonnes dont la corrélation de Pearson est inférieure au
    seuil spécifié, sans doublons."""

    # Sélectionner les colonnes à analyser
    columns = [
        "Tassement_Differentiel_IPN",
        "Tassement_Mur",
        "Tassement_Colline",
        "Corrosion_Index",
        "Fatigue_Factor",
        "Degradation_Factor",
        "Valeur moyenne",
        "Building_Age",
        "IPN_Age",
        "Palier_Duration",
    ]

    # Standardiser les colonnes avec StandardScaler
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df_paliers_combined[columns]), columns=columns
    )

    # Calculer la matrice de corrélation de Pearson
    corr_matrix = df_standardized.corr()

    # Créer une liste de tuples (colonne1, colonne2, coef_corr) en évitant les doublons (A,B) et (B,A)
    correlations = []
    seen_pairs = set()  # Pour éviter les doublons (A, B) == (B, A)

    for col1 in columns:
        for col2 in columns:
            if col1 != col2 and (col2, col1) not in seen_pairs:
                corr_coef = abs(corr_matrix.loc[col1, col2])
                if corr_coef < threshold:
                    correlations.append((col1, col2))
                    seen_pairs.add((col1, col2))  # Marquer la paire comme vue

    # Filtrer les paires de colonnes en fonction du seuil
    filtered_pairs = correlations

    # Calculer le nombre de subplots nécessaire (carré ou rectangle)
    num_plots = len(filtered_pairs)
    if num_plots == 0:
        print(
            f"Aucune paire de colonnes n'a une corrélation inférieure au seuil de {threshold}."
        )
        return

    num_cols = int(sqrt(num_plots))
    num_rows = ceil(num_plots / num_cols)

    # Créer une figure avec la grille de subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    # Si on a un seul subplot, le convertir en une liste pour une itération facile
    if num_plots == 1:
        axes = [axes]
    elif num_rows == 1:  # Si on a une seule ligne, axes sera un tableau 1D
        axes = axes.flatten()
    else:
        axes = axes.ravel()  # Sinon, aplatissez la grille 2D en 1D

    # Tracer les scatterplots dans chaque case
    for i, (col1, col2) in enumerate(filtered_pairs):
        ax = axes[i]
        sns.scatterplot(x=df_standardized[col1], y=df_standardized[col2], ax=ax)
        ax.set_title(f"{col1} vs {col2}")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)

    # Supprimer les subplots vides s'il y en a
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Appel de la fonction avec le DataFrame df_paliers_combined
plot_paliers_with_dual_axis(df_paliers_combined)
plot_standardized_boxplots(df_paliers_combined)
plot_filtered_scatterplot_grid(df_paliers_combined)
