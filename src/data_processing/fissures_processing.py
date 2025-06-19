import pandas as pd


def chargement_donnees(chemin):
    """Charge les données depuis deux fichiers Excel."""
    # Chargement du premier fichier Fissure_2.xlsx
    df = pd.read_excel(f"{chemin}Fissure_2.xlsx")
    df.columns = [
        "Date",
        "Bureau",
        "Mur extérieur",
        "Variation Bureau",
        "Variation Mur",
        "Mur route"
    ]
    df = df[df.columns[:-1]]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    # Chargement du deuxième fichier Fissure_old.xlsx depuis la feuille Feuil3
    df_old = pd.read_excel(f"{chemin}Fissure_old.xlsx", sheet_name="Feuil3")
    df_old.columns = ["date", "bureau_old"]
    df_old["Date"] = pd.to_datetime(df_old["date"])
    df_old["Bureau_old"] = df_old["bureau_old"].astype(float)

    return df, df_old
