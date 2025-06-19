import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_evolution_inclinaison(df_date, output_path):
    """
    df_date doit contenir 'measurement_time' et 'inch' + 'time_numeric'.
    On trace un regplot (OLS) avec IC à 95 %.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.regplot(x="time_numeric", y="inch", data=df_date,
                ci=95, scatter_kws={"s": 50, "alpha": 0.6},
                line_kws={"color": "red"})
    plt.xlabel("Date (timestamp)")
    plt.ylabel("Mesure (inch)")
    plt.title("Évolution de l'inclinaison du mur avec OLS (IC à 95%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_explanatory_evolution(df_features, date_series, top_features, combo_label, output_folder):
    """
    Trace l'évolution de quelques features explicatives dans le temps.
    """
    plt.figure(figsize=(16, 8))
    for feat in top_features:
        plt.plot(date_series, df_features[feat], label=feat, alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Valeur de la feature")
    plt.title(f"Évolution des features explicatives - {combo_label}")
    plt.legend(loc="best")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(output_folder, f"explanatory_evolution_{combo_label}.png")
    plt.savefig(fig_path)
    plt.close()
