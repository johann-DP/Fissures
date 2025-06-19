import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Charger les données depuis les fichiers Excel
file_old = "data/Fissures/Fissure_old.xlsx"
file_new = "data/Fissures/Fissure_2.xlsx"

# Lecture des données
df_old = pd.read_excel(file_old, sheet_name="Feuil3", parse_dates=["date"])
df_new = pd.read_excel(file_new, parse_dates=["Date"])

# Renommer les colonnes pour faciliter la fusion des données
df_old.rename(columns={"date": "Date", "bureau_old": "Bureau_old"}, inplace=True)
df_new.rename(columns={"Bureau\n(mm)": "Bureau_new"}, inplace=True)

# Filtrer les données de la première phase après le 1er janvier 2015 pour la régression linéaire
df_old_filtered = df_old[df_old["Date"] >= "2015-01-01"]

# Régression linéaire sur les données filtrées de la première phase
X_old_filtered = (
    df_old_filtered["Date"] - df_old_filtered["Date"].min()
).dt.days.values.reshape(-1, 1)
y_old_filtered = df_old_filtered["Bureau_old"].values
model = LinearRegression().fit(X_old_filtered, y_old_filtered)

# Prédiction pour la première date de la seconde phase
X_new_start = np.array(
    [(df_new["Date"].iloc[0] - df_old_filtered["Date"].min()).days]
).reshape(-1, 1)
predicted_start = (
    model.predict(X_new_start)[0] - 0.103
)  # Offset dû à la fluctuation de la première mesure

# Calcul de la pente moyenne par jour de la période postérieure à 2015
slope_old_filtered = model.coef_[0]

# Calculer la pente linéaire de la deuxième phase basée sur les premier et dernier points
days_new = (df_new["Date"] - df_new["Date"].min()).dt.days.values
predicted_end = predicted_start + slope_old_filtered * days_new[-1]
current_end = df_new["Bureau_new"].iloc[-1]

# Ajustement proportionnel
scaling_factor = (
    1.16  # Moyenne des sauts de paliers old = Moyenne des sauts de paliers new
)
df_new["Bureau_new_adjusted"] = predicted_start + scaling_factor * (
    df_new["Bureau_new"] - df_new["Bureau_new"].iloc[0]
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

# Vérifier et supprimer les doublons dans l'index
df_combined = df_combined.drop_duplicates(subset="Date").set_index("Date").sort_index()

# Diviser les données pour les tracer séparément
df_combined_old = df_combined.loc[df_old["Date"].min() : df_old["Date"].max()]
df_combined_new = df_combined.loc[df_new["Date"].min() : df_new["Date"].max()]

# Calculer X et y combinés pour les deux phases
X_old_combined = (df_combined_old.index - df_combined_old.index.min()).days.values
y_old_combined = df_combined_old["Bureau"].values
X_new_combined = (df_combined_new.index - df_combined_new.index.min()).days.values
y_new_combined = df_combined_new["Bureau"].values

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
            df_combined_old.index.get_indexer([pd.to_datetime(date)], method="nearest")[
                0
            ]
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
    "2024-09-01",
]
manual_breaks_new = [
    df_combined_new.index.get_loc(
        df_combined_new.index[
            df_combined_new.index.get_indexer([pd.to_datetime(date)], method="nearest")[
                0
            ]
        ]
    )
    for date in manual_breaks_new_dates
]


# Calcul des durées des paliers
def calculate_durations(df_paliers):
    durations = []
    for i in range(len(df_paliers)):
        start_date = df_paliers.iloc[i]["Début"]
        end_date = df_paliers.iloc[i]["Fin"]
        duration = (end_date - start_date).days
        durations.append(duration)
    return durations


# Calcul des hauteurs des segments verticaux
def calculate_heights(df_paliers):
    heights = []
    for i in range(len(df_paliers) - 1):
        height = (
            df_paliers.iloc[i + 1]["Valeur moyenne"]
            - df_paliers.iloc[i]["Valeur moyenne"]
        )
        heights.append(height)
    return heights


# Générer les segments pour la première phase
segments_old = []
paliers_old = []  # Liste pour stocker les paliers

for i in range(len(manual_breaks_old) - 1):
    start = manual_breaks_old[i]
    end = manual_breaks_old[i + 1]
    if i == 0:  # Premier segment partant du premier point de mesure
        x_segment = [df_combined_old.index[start], df_combined_old.index[end]]
        y_segment = [y_old_combined[start], y_old_combined[end]]
        # segments_old.append((x_segment, y_segment))  # Commenté pour ne pas afficher ce segment
    elif i % 2 == 0:  # Segments de pente positive reliant les paliers
        x_segment = [df_combined_old.index[start], df_combined_old.index[end]]
        y_segment = [y_old_combined[start], y_old_combined[end]]
        # segments_old.append((x_segment, y_segment))  # Commenté pour ne pas afficher ces segments
    else:  # Segments plats (pente nulle)
        avg_value = np.mean(y_old_combined[start : end + 1])
        y_pred = np.full(end + 1 - start, avg_value)
        segments_old.append((df_combined_old.index[start : end + 1], y_pred))
        # Ajouter les paliers au DataFrame
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
        # Ajouter les paliers au DataFrame
        paliers_new.append(
            [df_combined_new.index[start], df_combined_new.index[end], avg_value]
        )
    else:  # Segments de pente positive reliant les paliers
        x_segment = [df_combined_new.index[start], df_combined_new.index[end]]
        y_segment = [y_new_combined[start], y_new_combined[end]]
        # segments_new.append((x_segment, y_segment))  # Commenté pour ne pas afficher ces segments

# Création des DataFrames pour les paliers
df_paliers_old = pd.DataFrame(paliers_old, columns=["Début", "Fin", "Valeur moyenne"])
df_paliers_new = pd.DataFrame(paliers_new, columns=["Début", "Fin", "Valeur moyenne"])

# Recalculer les durées et hauteurs
horizontal_durations_old = calculate_durations(df_paliers_old)
horizontal_durations_new = calculate_durations(df_paliers_new)

horizontal_heights_old = calculate_heights(df_paliers_old)
horizontal_heights_new = calculate_heights(df_paliers_new)


# Fonction pour convertir les hauteurs de mm à µm
def convert_mm_to_um(height_mm):
    return int(
        round(height_mm * 1000)
    )  # Convertir en µm et arrondir à l'entier le plus proche


# Visualisation
plt.figure(figsize=(12, 6))

# Conserver les scatterplots existants
plt.scatter(
    df_combined_old.index,
    y_old_combined,
    label="Période Ancienne",
    color="blue",
    marker=".",
    alpha=0.5,
)
plt.scatter(
    df_combined_new.index,
    y_new_combined,
    label="Période Récente (Ajustée)",
    color="red",
    marker=".",
    alpha=0.5,
)

# Tracer les segments pour la première phase (conserver les segments horizontaux)
for segment in segments_old:
    plt.plot(
        segment[0], segment[1], color="blue", linestyle="-"
    )  # Tous les segments sont en bleu

# Tracer les segments pour la deuxième phase (conserver les segments horizontaux)
for segment in segments_new:
    plt.plot(
        segment[0], segment[1], color="red", linestyle="-"
    )  # Tous les segments sont en rouge

# Ajouter le segment du premier point au début du premier palier
premier_point_x = df_combined_old.index[0]
premier_point_y = y_old_combined[0]
debut_premier_palier_x = df_paliers_old.iloc[0]["Début"]
debut_premier_palier_y = df_paliers_old.iloc[0]["Valeur moyenne"]

plt.plot(
    [premier_point_x, debut_premier_palier_x],
    [premier_point_y, debut_premier_palier_y],
    color="blue",
    linestyle="--",
)

# Ajouter les segments entre paliers (de la fin d'un palier au début du suivant)
for i in range(len(df_paliers_old) - 1):
    plt.plot(
        [df_paliers_old.iloc[i]["Fin"], df_paliers_old.iloc[i + 1]["Début"]],
        [
            df_paliers_old.iloc[i]["Valeur moyenne"],
            df_paliers_old.iloc[i + 1]["Valeur moyenne"],
        ],
        color="blue",
        linestyle="--",
    )  # Tous les segments sont en bleu

for i in range(len(df_paliers_new) - 1):
    plt.plot(
        [df_paliers_new.iloc[i]["Fin"], df_paliers_new.iloc[i + 1]["Début"]],
        [
            df_paliers_new.iloc[i]["Valeur moyenne"],
            df_paliers_new.iloc[i + 1]["Valeur moyenne"],
        ],
        color="red",
        linestyle="--",
    )  # Tous les segments sont en rouge

# Ajouter les annotations pour les paliers (durée) au centre des paliers
for i in range(len(df_paliers_old)):
    duration_days = horizontal_durations_old[i]
    plt.annotate(
        f"{duration_days} jours",
        xy=(
            df_paliers_old.iloc[i]["Début"]
            + (df_paliers_old.iloc[i]["Fin"] - df_paliers_old.iloc[i]["Début"]) / 2,
            df_paliers_old.iloc[i]["Valeur moyenne"],
        ),
        xytext=(0, -30),
        textcoords="offset points",
        fontsize=7,
        color="blue",
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="blue", facecolor="lightblue", alpha=0.5
        ),
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="blue"),
    )

for i in range(len(df_paliers_new)):
    duration_days = horizontal_durations_new[i]
    plt.annotate(
        f"{duration_days} jours",
        xy=(
            df_paliers_new.iloc[i]["Début"]
            + (df_paliers_new.iloc[i]["Fin"] - df_paliers_new.iloc[i]["Début"]) / 2,
            df_paliers_new.iloc[i]["Valeur moyenne"],
        ),
        xytext=(15, -30),
        textcoords="offset points",
        fontsize=7,
        color="red",
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="red", facecolor="lightcoral", alpha=0.5
        ),
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="red"),
    )

# Ajouter les segments verticaux avec des doubles-flèches pour les hauteurs
for i in range(len(df_paliers_old) - 1):
    height_um = convert_mm_to_um(horizontal_heights_old[i])
    plt.plot(
        [df_paliers_old.iloc[i]["Fin"], df_paliers_old.iloc[i]["Fin"]],
        [
            df_paliers_old.iloc[i]["Valeur moyenne"],
            df_paliers_old.iloc[i + 1]["Valeur moyenne"],
        ],
        color="blue",
        linestyle="-",
        alpha=0.3,
    )
    plt.annotate(
        f"{height_um} µm",
        xy=(
            df_paliers_old.iloc[i]["Fin"],
            (
                df_paliers_old.iloc[i]["Valeur moyenne"]
                + df_paliers_old.iloc[i + 1]["Valeur moyenne"]
            )
            / 2,
        ),
        xytext=(-50, 10),
        textcoords="offset points",
        fontsize=7,
        color="blue",
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="blue", facecolor="lightblue", alpha=0.5
        ),
        arrowprops=dict(arrowstyle="-|>", color="blue"),
    )

for i in range(len(df_paliers_new) - 1):
    height_um = convert_mm_to_um(horizontal_heights_new[i])
    plt.plot(
        [df_paliers_new.iloc[i]["Fin"], df_paliers_new.iloc[i]["Fin"]],
        [
            df_paliers_new.iloc[i]["Valeur moyenne"],
            df_paliers_new.iloc[i + 1]["Valeur moyenne"],
        ],
        color="red",
        linestyle="-",
        alpha=0.3,
    )
    plt.annotate(
        f"{height_um} µm",
        xy=(
            df_paliers_new.iloc[i]["Fin"],
            (
                df_paliers_new.iloc[i]["Valeur moyenne"]
                + df_paliers_new.iloc[i + 1]["Valeur moyenne"]
            )
            / 2,
        ),
        xytext=(-50, 10),
        textcoords="offset points",
        fontsize=7,
        color="red",
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="red", facecolor="lightcoral", alpha=0.5
        ),
        arrowprops=dict(arrowstyle="-|>", color="red"),
    )

# Configuration du graphique pour afficher plus de détails sur les dates
plt.xlabel("Date")
plt.ylabel("Écartement de la Fissure (mm)")
plt.title("Évolution de l'Écartement de la Fissure avec Modélisation des Paliers")

# Ajustement des formats de date pour l'axe des x
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
# plt.gcf().autofmt_xdate()

plt.legend()
plt.grid(True)
plt.show()
