import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Charger les fichiers Excel et les concaténer
file1 = "20231201_20240301.xls"
file2 = "20240301_20240601.xls"
direct = "C:/Users/johan/Bureau/"
path1 = direct + file1
path2 = direct + file2

df1 = pd.read_excel(path1)
df2 = pd.read_excel(path2)

# Vérifier si les colonnes sont identiques dans les deux fichiers
assert (
    df1.columns.tolist() == df2.columns.tolist()
), "Les colonnes des deux fichiers ne correspondent pas"

# Concaténer les deux DataFrames
df_combined = pd.concat([df1, df2], ignore_index=True)
print("Jointure des fichiers effectuée")

# Suppression des colonnes vides
df_cleaned = df_combined.dropna(axis=1, how="all")

# Suppression des lignes contenant des valeurs interpolées non pertinentes
# interpolated_flag = -999  # Modifier cette variable selon votre dataset
# df_cleaned = df_cleaned[df_cleaned.apply(lambda row: all(row != interpolated_flag), axis=1)]

# Convertir la colonne 'Time' au format datetime et retirer les données manquantes
df_cleaned["Time"] = pd.to_datetime(df_cleaned["Time"])
df_cleaned = df_cleaned[df_cleaned["Time"] >= "2023-12-17 16:51:21+08:00"]

# Identifier et traiter l'outlier dans la température minimale extérieure
outlier = df_cleaned[df_cleaned["Outdoor Tem.Min(°C)"] > 1000]
print("Outlier identifié:", outlier[["Time", "Outdoor Tem.Min(°C)"]])
df_cleaned.loc[outlier.index, "Outdoor Tem.Min(°C)"] = np.nan
df_cleaned["Outdoor Tem.Min(°C)"].interpolate(method="linear", inplace=True)

# Transformation de la direction du vent
df_cleaned["Wind direction sin"] = np.sin(np.radians(df_cleaned["Wind direction"]))
df_cleaned["Wind direction cos"] = np.cos(np.radians(df_cleaned["Wind direction"]))

# Traitement des variables avec skewness
skewed_columns = [
    "Outdoor Hum.Max(%)",
    "Rainfull(Hour)(mm)",
    "Rainfull(Day)(mm)",
    "Wind speed(km/h)",
    "Light intensity",
    "UV rating",
]

for column in skewed_columns:
    df_cleaned[column + "_log"] = np.log1p(df_cleaned[column])


# Calcul des moyennes, médianes, écart-types, skewness et kurtosis par semaine pour les statistiques
def calculate_kurtosis(x):
    return x.kurtosis()


weekly_stats = df_cleaned.resample("W-SUN", on="Time").agg(
    ["mean", "median", "std", "skew", calculate_kurtosis]
)
weekly_stats.columns = ["_".join(x) for x in weekly_stats.columns.ravel()]

# Ajouter les moyennes mobiles (lissage pour les visualisations)
window_size = 21  # Par exemple, une fenêtre de 3 semaines
df_cleaned["Indoor Tem(°C) MA"] = (
    df_cleaned["Indoor Tem(°C)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Outdoor Tem(°C) MA"] = (
    df_cleaned["Outdoor Tem(°C)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Indoor Hum(%) MA"] = (
    df_cleaned["Indoor Hum(%)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Outdoor Hum(%) MA"] = (
    df_cleaned["Outdoor Hum(%)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Indoor Tem.Max(°C) MA"] = (
    df_cleaned["Indoor Tem.Max(°C)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Indoor Tem.Min(°C) MA"] = (
    df_cleaned["Indoor Tem.Min(°C)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Outdoor Tem.Max(°C) MA"] = (
    df_cleaned["Outdoor Tem.Max(°C)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Outdoor Tem.Min(°C) MA"] = (
    df_cleaned["Outdoor Tem.Min(°C)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Wind speed(km/h) MA"] = (
    df_cleaned["Wind speed(km/h)"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Wind direction sin MA"] = (
    df_cleaned["Wind direction sin"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Wind direction cos MA"] = (
    df_cleaned["Wind direction cos"].rolling(window=window_size, center=True).mean()
)
df_cleaned["Light intensity MA"] = (
    df_cleaned["Light intensity"].rolling(window=window_size, center=True).mean()
)
df_cleaned["UV rating MA"] = (
    df_cleaned["UV rating"].rolling(window=window_size, center=True).mean()
)

# Fusion des statistiques hebdomadaires avec le dataset principal
df_cleaned = pd.merge_asof(
    df_cleaned.sort_values("Time"),
    weekly_stats.reset_index().sort_values("Time"),
    on="Time",
    direction="backward",
)

# Sauvegarde du dataset pour les travaux futurs
df_cleaned.to_csv("df_cleaned_with_stats.csv", index=False)

# Normaliser les données (sauf la colonne 'Time')
scaler = StandardScaler()
data_to_normalize = df_cleaned.drop(columns=["Time"])
normalized_data = scaler.fit_transform(data_to_normalize)
df_normalized = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)

# Visualiser les boxplots des variables normalisées sur une seule figure
plt.figure(figsize=(45, 10), dpi=300)
df_normalized.boxplot(rot=90)
plt.title("Boxplots des variables météorologiques normalisées")
plt.show()

# Visualisation des données

# Humidité intérieure et extérieure
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Hum(%)"],
    label="Indoor Humidity",
    alpha=0.3,
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Hum(%) MA"],
    label="Indoor Humidity MA",
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Hum(%)"],
    label="Outdoor Humidity",
    alpha=0.3,
    c="green",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Hum(%) MA"],
    label="Outdoor Humidity MA",
    c="green",
    lw=1,
)
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.title("Évolution de l'humidité intérieure et extérieure")
plt.legend(fontsize=6)
plt.show()

# Températures maximales et minimales
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Tem.Max(°C)"],
    label="Indoor Max Temperature",
    alpha=0.3,
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Tem.Max(°C) MA"],
    label="Indoor Max Temperature MA",
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Tem.Min(°C)"],
    label="Indoor Min Temperature",
    alpha=0.3,
    c="green",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Tem.Min(°C) MA"],
    label="Indoor Min Temperature MA",
    c="green",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Tem.Max(°C)"],
    label="Outdoor Max Temperature",
    alpha=0.3,
    c="purple",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Tem.Max(°C) MA"],
    label="Outdoor Max Temperature MA",
    c="purple",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Tem.Min(°C)"],
    label="Outdoor Min Temperature",
    alpha=0.3,
    c="orange",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Tem.Min(°C) MA"],
    label="Outdoor Min Temperature MA",
    c="orange",
    lw=1,
)
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Évolution des températures maximales et minimales")
plt.legend(fontsize=6)
plt.show()

# Évolution des précipitations (log scale)
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Rainfull(Hour)(mm)"],
    label="Rainfall per Hour",
    lw=1,
)
plt.plot(
    df_cleaned["Time"], df_cleaned["Rainfull(Day)(mm)"], label="Rainfall per Day", lw=1
)
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Rainfall (mm)")
plt.title("Évolution des précipitations (échelle log)")
plt.legend(fontsize=6)
plt.show()

# Vitesse et direction du vent
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Wind speed(km/h)"],
    label="Wind Speed",
    alpha=0.3,
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Wind speed(km/h) MA"],
    label="Wind Speed MA",
    c="blue",
    lw=1,
)
plt.xlabel("Time")
plt.ylabel("Wind Speed (km/h)")
plt.title("Évolution de la vitesse du vent")
plt.legend(fontsize=6)
plt.show()

# Direction du vent (sin et cos)
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Wind direction sin"],
    label="Wind Direction Sin",
    alpha=0.3,
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Wind direction sin MA"],
    label="Wind Direction Sin MA",
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Wind direction cos"],
    label="Wind Direction Cos",
    alpha=0.3,
    c="green",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Wind direction cos MA"],
    label="Wind Direction Cos MA",
    c="green",
    lw=1,
)
plt.xlabel("Time")
plt.ylabel("Direction")
plt.title("Évolution de la direction du vent (sinus et cosinus)")
plt.legend(fontsize=6)
plt.show()

# Intensité lumineuse et indice UV
fig, ax1 = plt.subplots(figsize=(15, 5), dpi=300)

ax1.plot(
    df_cleaned["Time"],
    df_cleaned["Light intensity"],
    label="Light Intensity",
    color="tab:blue",
    alpha=0.3,
    lw=1,
)
ax1.plot(
    df_cleaned["Time"],
    df_cleaned["Light intensity MA"],
    label="Light Intensity MA",
    color="tab:blue",
    lw=1,
)
ax1.set_xlabel("Time")
ax1.set_ylabel("Light Intensity", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    df_cleaned["Time"],
    df_cleaned["UV rating"],
    label="UV Rating",
    color="tab:green",
    alpha=0.3,
    lw=1,
)
ax2.plot(
    df_cleaned["Time"],
    df_cleaned["UV rating MA"],
    label="UV Rating MA",
    color="tab:green",
    lw=1,
)
ax2.set_ylabel("UV Rating", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")

fig.tight_layout()
plt.title("Évolution de l'intensité lumineuse et de l'indice UV")
fig.legend(loc="upper right", fontsize=6)
plt.show()

# Moyennes mobiles pour les températures intérieures et extérieures
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Tem(°C)"],
    label="Indoor Temperature",
    alpha=0.3,
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Indoor Tem(°C) MA"],
    label="Indoor Temperature MA",
    c="blue",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Tem(°C)"],
    label="Outdoor Temperature",
    alpha=0.3,
    c="green",
    lw=1,
)
plt.plot(
    df_cleaned["Time"],
    df_cleaned["Outdoor Tem(°C) MA"],
    label="Outdoor Temperature MA",
    c="green",
    lw=1,
)
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Évolution de la température intérieure et extérieure avec moyennes mobiles")
plt.legend(fontsize=6)
plt.show()


# Fonction pour regrouper les mêmes grandeurs sur une même figure
def plot_weekly_statistics(df_cleaned, variable_base_names):
    for base_name in variable_base_names:
        # Filtrer les colonnes correspondant aux statistiques de base_name
        stats_columns = [
            col
            for col in df_cleaned.columns
            if base_name in col
            and any(
                stat in col
                for stat in ["mean", "median", "std", "skew", "calculate_kurtosis"]
            )
        ]

        # Subplots pour moyennes et médianes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, dpi=300)

        for col in stats_columns:
            if "mean" in col or "median" in col:
                ax1.plot(df_cleaned["Time"], df_cleaned[col], label=col.split("_")[1])
            if "std" in col:
                ax2.plot(
                    df_cleaned["Time"],
                    df_cleaned[col],
                    label=col.split("_")[1] + " (std)",
                    color="tab:blue",
                )
            if "skew" in col:
                ax3 = ax2.twinx()
                ax3.plot(
                    df_cleaned["Time"],
                    df_cleaned[col],
                    label=col.split("_")[1] + " (skew)",
                    linestyle="--",
                    color="tab:orange",
                )
                ax3.set_ylabel("Skewness", color="tab:orange")
                ax3.tick_params(axis="y", labelcolor="tab:orange")
            if "calculate_kurtosis" in col:
                ax4 = ax2.twinx()
                ax4.spines["right"].set_position(("axes", 1.2))
                ax4.plot(
                    df_cleaned["Time"],
                    df_cleaned[col],
                    label=col.split("_")[1] + " (kurtosis)",
                    linestyle=":",
                    color="tab:red",
                )
                ax4.set_ylabel("Kurtosis", color="tab:red")
                ax4.tick_params(axis="y", labelcolor="tab:red")

        ax1.set_title(f"Évolution des moyennes et médianes pour {base_name}")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title(
            f"Évolution des écart-types, skewness et kurtosis pour {base_name}"
        )
        ax2.set_ylabel("Standard Deviation", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        fig.tight_layout()
        plt.xlabel("Time")
        plt.show()


# Liste des bases des noms de variables à tracer
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

# Tracer des grandeurs statistiques hebdomadaires
plot_weekly_statistics(df_cleaned, variable_base_names)


# Fonction pour extraire les statistiques hebdomadaires à la date de la mesure
def extract_weekly_statistics(df_cleaned, variable_base_names):
    stats_columns = [
        col
        for base_name in variable_base_names
        for col in df_cleaned.columns
        if base_name in col
        and any(
            stat in col
            for stat in ["mean", "median", "std", "skew", "calculate_kurtosis"]
        )
    ]

    # Extraire uniquement les dates des mesures (dimanche à midi)
    measurement_dates = df_cleaned[df_cleaned["Time"].dt.weekday == 6]["Time"]
    df_stats = df_cleaned[df_cleaned["Time"].isin(measurement_dates)][stats_columns]
    df_stats["Time"] = measurement_dates.values
    return df_stats


df_stats = extract_weekly_statistics(df_cleaned, variable_base_names)

# Tracés la matrice de distributions
pairplot = sns.pairplot(df_cleaned[variable_base_names], diag_kind="kde")
pairplot.fig.set_dpi(300)
plt.show()

print("Nettoyage des données et visualisations terminés")
