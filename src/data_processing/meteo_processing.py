import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_concat_excel_files(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".xls")])
    dataframes = [pd.read_excel(os.path.join(data_dir, file)) for file in files]
    df_combined = pd.concat(dataframes, ignore_index=True)
    return df_combined


def clean_data(df_combined):
    df_cleaned = df_combined.dropna(axis=1, how="all")
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('CH')]
    df_cleaned["Time"] = pd.to_datetime(df_cleaned["Time"])
    df_cleaned = df_cleaned[df_cleaned["Time"] >= "2023-12-17 16:51:21+08:00"]

    outlier = df_cleaned[df_cleaned["Outdoor Tem.Min(°C)"] > 1000]
    df_cleaned.loc[outlier.index, "Outdoor Tem.Min(°C)"] = np.nan
    df_cleaned["Outdoor Tem.Min(°C)"].interpolate(method="linear", inplace=True)

    df_cleaned["Wind direction sin"] = np.sin(np.radians(df_cleaned["Wind direction"]))
    df_cleaned["Wind direction cos"] = np.cos(np.radians(df_cleaned["Wind direction"]))

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

    # Changement parvenu le 6/10/2024
    # Remplir les valeurs manquantes de 'Wind speed(km/h)' avec 0
    df_cleaned["Wind speed(km/h)"].fillna(
        df_cleaned["Wind speed(Hour)(km/h)"], inplace=True
    )

    # Supprimer les colonnes 'Wind speed(Hour)(km/h)' et 'Wind speed(Day)(km/h)'
    df_cleaned.drop(
        columns=["Wind speed(Hour)(km/h)", "Wind speed(Day)(km/h)"], inplace=True
    )

    return df_cleaned


def compute_weekly_stats(df_cleaned):
    def calculate_kurtosis(x):
        return x.kurtosis()

    weekly_stats = df_cleaned.resample("W-SUN", on="Time").agg(
        ["mean", "median", "std", "skew", calculate_kurtosis]
    )
    weekly_stats.columns = ["_".join(x) for x in weekly_stats.columns.ravel()]

    return weekly_stats


def add_moving_averages(df_cleaned, window_size=21):
    columns_to_average = [
        "Indoor Tem(°C)",
        "Outdoor Tem(°C)",
        "Indoor Hum(%)",
        "Outdoor Hum(%)",
        "Indoor Tem.Max(°C)",
        "Indoor Tem.Min(°C)",
        "Outdoor Tem.Max(°C)",
        "Outdoor Tem.Min(°C)",
        "Wind speed(km/h)",
        "Wind direction",
        "Wind direction sin",
        "Wind direction cos",
        "Light intensity",
        "UV rating",
        "Indoor Water Content (g/m³)",
        "Outdoor Water Content (g/m³)",
        "Indoor Water Content Max (g/m³)",
        "Indoor Water Content Min (g/m³)",
        "Outdoor Water Content Max (g/m³)",
        "Outdoor Water Content Min (g/m³)",
    ]

    for column in columns_to_average:
        df_cleaned[column + " MA"] = (
            df_cleaned[column].rolling(window=window_size, center=True).mean()
        )
        assert (
            column + " MA" in df_cleaned.columns
        ), f"Column {column + ' MA'} not found after moving average computation!"

    return df_cleaned


def save_cleaned_data(df_cleaned, output_path):
    df_cleaned.to_csv(output_path, index=False)


def normalize_data(df_cleaned):
    scaler = StandardScaler()
    data_to_normalize = df_cleaned.drop(columns=["Time"])
    normalized_data = scaler.fit_transform(data_to_normalize)
    df_normalized = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
    return df_normalized


def Ps(theta):
    return 610.5 * np.exp((17.27 * theta) / (theta + 237.3))


def Pi(theta, RH):
    return (RH / 100) * Ps(theta)


def rho_v(theta, RH):
    Pi_val = Pi(theta, RH)
    M_v = 18.016
    R = 8.314
    T = theta + 273.15
    return Pi_val * M_v / (R * T)


def calculate_water_content(df_cleaned):
    df_cleaned["Indoor Water Content (g/m³)"] = rho_v(
        df_cleaned["Indoor Tem(°C)"], df_cleaned["Indoor Hum(%)"]
    )
    df_cleaned["Indoor Water Content Max (g/m³)"] = rho_v(
        df_cleaned["Indoor Tem.Max(°C)"], df_cleaned["Indoor Hum(%)"]
    )
    df_cleaned["Indoor Water Content Min (g/m³)"] = rho_v(
        df_cleaned["Indoor Tem.Min(°C)"], df_cleaned["Indoor Hum(%)"]
    )

    df_cleaned["Outdoor Water Content (g/m³)"] = rho_v(
        df_cleaned["Outdoor Tem(°C)"], df_cleaned["Outdoor Hum(%)"]
    )
    df_cleaned["Outdoor Water Content Max (g/m³)"] = rho_v(
        df_cleaned["Outdoor Tem.Max(°C)"], df_cleaned["Outdoor Hum(%)"]
    )
    df_cleaned["Outdoor Water Content Min (g/m³)"] = rho_v(
        df_cleaned["Outdoor Tem.Min(°C)"], df_cleaned["Outdoor Hum(%)"]
    )

    return df_cleaned


def add_weekly_stats(df_cleaned):
    def calculate_kurtosis(x):
        return x.kurtosis()

    weekly_stats = df_cleaned.resample("W-SUN", on="Time").agg(
        ["mean", "median", "std", "skew", calculate_kurtosis]
    )
    weekly_stats.columns = ["_".join(x) for x in weekly_stats.columns.ravel()]

    df_cleaned = pd.merge_asof(
        df_cleaned.sort_values("Time"),
        weekly_stats.reset_index().sort_values("Time"),
        on="Time",
        direction="backward",
    )

    return df_cleaned


def add_temporal_derivatives(df):
    # Assurez-vous que 'Time' est en datetime si ce n'est pas déjà le cas
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"])

    # Convertir 'Time' en secondes depuis le début pour la simplicité des calculs
    time_in_seconds = (df["Time"] - df["Time"].min()).dt.total_seconds()

    # Calculer la dérivée pour chaque colonne autre que 'Time'
    for column in df.columns:
        if column != "Time":
            # Calcul de la différence des valeurs
            value_diff = df[column].diff()
            # Calcul de la différence de temps en secondes
            time_diff = time_in_seconds.diff()
            # Calcul de la dérivée temporelle
            df[f"{column}_dt"] = value_diff / time_diff

    return df
