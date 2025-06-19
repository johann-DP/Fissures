import base64
import io
import os
from math import ceil, sqrt

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.preprocessing import StandardScaler

from data_processing.fissures_processing import chargement_donnees
from visualization.fissures_visualization import preprocessing_old_new


def return_df_paliers_combined():
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
        df_paliers_combined["Fatigue_Factor"] = np.sqrt(
            df_paliers_combined["Building_Age"]
        )
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
    (_, _, _, _, _, _, _, _, _, _, _, _, df_paliers_old, df_paliers_new, _) = (
        preprocessing_old_new(df_fissures, df_fissures_old)
    )
    df_paliers_combined = structure_dataviz(df_paliers_old, df_paliers_new)

    return df_paliers_combined


def generate_building_plan():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))

    # Dimensions du bâtiment et éléments structurels
    longueur_batiment = 15
    largeur_batiment = 7.5
    hauteur_batiment = 2.7
    epaisseur_murs = 0.15

    # Toit terrasse
    epaisseur_dalle_toit = 0.15
    epaisseur_isolation_toit = 0.12
    epaisseur_gravier = 0.10
    hauteur_acrotere = 0.50
    epaisseur_acrotere = 0.20
    epaisseur_toit_total = (
        epaisseur_dalle_toit + epaisseur_isolation_toit + epaisseur_gravier
    )

    # Fondations et structure sous le bâtiment
    longueur_poutre = 7.5
    hauteur_poutre = 0.30
    largeur_poutre = 0.30
    porte_a_faux = 2.8
    appui_colline = 4.7

    # IPN
    hauteur_ipn = 2.4
    largeur_ipn = 0.20
    epaisseur_ipn_centrale = 0.30

    # Socles en béton
    taille_socle = 1

    # Mur de soutènement
    hauteur_mur_sout = 2.4
    epaisseur_mur_sout = 0.35

    # Création de la figure et des axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))

    # Fixer les limites des axes pour correspondre exactement
    ax1.set_xlim(-1, 16)  # Pour la vue de face
    ax2.set_xlim(-1, 16)  # Pour la vue de côté
    ax1.set_ylim(-1.5, 6)  # Pour la vue de face
    ax2.set_ylim(-1.5, 6)  # Pour la vue de côté

    # S'assurer que les deux axes ont le même aspect ratio
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # Vue de face
    ax1.set_title("Vue de face du bâtiment")

    # Colline (vue de côté)
    colline = patches.Polygon(
        [(-1, hauteur_mur_sout + 1.2), (16, hauteur_mur_sout + 1.2), (16, 0), (-1, 0)],
        closed=True,
        linewidth=1,
        edgecolor="black",
        facecolor="green",
    )
    ax1.add_patch(colline)

    # Espace
    espace = patches.Rectangle(
        (0, hauteur_mur_sout),
        longueur_batiment,
        hauteur_mur_sout + hauteur_poutre,
        linewidth=1,
        edgecolor=None,
        facecolor="white",
    )
    ax1.add_patch(espace)

    # Mur de soutènement
    mur_sout = patches.Rectangle(
        (0, 0),
        longueur_batiment,
        hauteur_mur_sout,
        linewidth=1,
        edgecolor="black",
        facecolor="gray",
    )
    ax1.add_patch(mur_sout)

    # Murs du bâtiment
    batiment = patches.Rectangle(
        (0, hauteur_mur_sout + hauteur_poutre),
        longueur_batiment,
        hauteur_batiment,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgray",
    )
    ax1.add_patch(batiment)

    # Dalle de sol (vue de face) - 15 cm d'épaisseur, placée au-dessus des poutres
    dalle_sol_face = patches.Rectangle(
        (0, hauteur_mur_sout + largeur_poutre),
        longueur_batiment,
        0.15,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax1.add_patch(dalle_sol_face)

    # Acrotère complet (vue de face)
    acrotere_face = patches.Rectangle(
        (
            -0.7,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit
            + epaisseur_gravier
            - 0.20,
        ),
        longueur_batiment + 1.4,
        0.50,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax1.add_patch(acrotere_face)

    # Couches du toit (vue de face) - correcte alignement des dimensions
    dalle_beton = patches.Rectangle(
        (0, hauteur_mur_sout + hauteur_batiment),
        longueur_batiment,
        epaisseur_dalle_toit,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax1.add_patch(dalle_beton)

    isolation = patches.Rectangle(
        (0, hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit),
        longueur_batiment,
        epaisseur_isolation_toit,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgray",
    )
    ax1.add_patch(isolation)

    gravier = patches.Rectangle(
        (
            0,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit,
        ),
        longueur_batiment,
        epaisseur_gravier,
        linewidth=1,
        edgecolor="black",
        facecolor="gray",
    )
    ax1.add_patch(gravier)

    # Remblai (vue de face) - sous le mur de soutènement et entourant les socles des IPN, sans les masquer
    remblai = patches.Polygon(
        [(-1, 0), (16, 0), (16, -1.5), (-1, -1.5)],
        closed=True,
        linewidth=1,
        edgecolor="black",
        facecolor="yellow",
    )
    ax1.add_patch(remblai)

    # IPN sous les poutres - 5 IPN espacés correctement
    espacement_ipn = 3.6  # Espacement des IPN
    for i in range(5):  # Cinq IPN, espacés de 3.6m
        x_ipn = 0.30 + i * espacement_ipn  # IPN commence à 30 cm du bord
        ipn = patches.Rectangle(
            (x_ipn - largeur_ipn / 2, 0),
            largeur_ipn,
            hauteur_ipn,
            linewidth=1,
            edgecolor="black",
            facecolor="blue",
        )
        ax1.add_patch(ipn)

        # Aligner les poutres sur les IPN
        poutre = patches.Rectangle(
            (x_ipn - largeur_poutre / 2, hauteur_mur_sout),
            largeur_poutre,
            hauteur_poutre,
            linewidth=1,
            edgecolor="black",
            facecolor="gray",
        )
        ax1.add_patch(poutre)

        # Ajouter le socle en béton sous l'IPN
        socle = patches.Rectangle(
            (x_ipn - taille_socle / 2, -taille_socle),
            taille_socle,
            taille_socle,
            linewidth=1,
            edgecolor="black",
            facecolor="darkgray",
        )
        ax1.add_patch(socle)

    # Vue de côté
    ax2.set_title("Vue de côté du bâtiment")

    # Mur de soutènement (côté) - Positionné à 4,7 m de l'autre extrémité des poutres
    mur_sout_cote = patches.Rectangle(
        (appui_colline, 0),
        epaisseur_mur_sout,
        hauteur_mur_sout,
        linewidth=1,
        edgecolor="black",
        facecolor="gray",
    )
    ax2.add_patch(mur_sout_cote)

    # Bâtiment (vue côté)
    batiment_cote = patches.Rectangle(
        (epaisseur_mur_sout, hauteur_mur_sout + largeur_poutre),
        largeur_batiment,
        hauteur_batiment,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgray",
    )
    ax2.add_patch(batiment_cote)

    # Acrotère complet (vue de côté)
    acrotere_cote = patches.Rectangle(
        (
            epaisseur_mur_sout - 0.7,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit
            + epaisseur_gravier
            - 0.20,
        ),
        largeur_batiment + 1.4,
        0.50,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax2.add_patch(acrotere_cote)

    # Couches du toit (vue de côté) - correct alignement des dimensions
    dalle_beton_cote = patches.Rectangle(
        (epaisseur_mur_sout, hauteur_mur_sout + hauteur_batiment),
        largeur_batiment,
        epaisseur_dalle_toit,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax2.add_patch(dalle_beton_cote)

    isolation_cote = patches.Rectangle(
        (
            epaisseur_mur_sout,
            hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit,
        ),
        largeur_batiment,
        epaisseur_isolation_toit,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgray",
    )
    ax2.add_patch(isolation_cote)

    gravier_cote = patches.Rectangle(
        (
            epaisseur_mur_sout,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit,
        ),
        largeur_batiment,
        epaisseur_gravier,
        linewidth=1,
        edgecolor="black",
        facecolor="gray",
    )
    ax2.add_patch(gravier_cote)

    # Colline (vue de côté)
    colline = patches.Polygon(
        [
            (-1, hauteur_mur_sout + 1.2),
            (7.0 - (taille_socle - largeur_ipn) / 2, -1.5),
            (-1, -1.5),
        ],
        closed=True,
        linewidth=1,
        edgecolor="black",
        facecolor="green",
    )
    ax2.add_patch(colline)

    # Dalle de sol (vue de côté) - 15 cm d'épaisseur, placée au-dessus des poutres
    dalle_sol_cote = patches.Rectangle(
        (epaisseur_mur_sout, hauteur_mur_sout + largeur_poutre),
        longueur_poutre,
        0.15,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax2.add_patch(dalle_sol_cote)

    # Poutre en béton (vue côté)
    poutre_cote = patches.Rectangle(
        (epaisseur_mur_sout, hauteur_mur_sout),
        longueur_poutre,
        hauteur_poutre,
        linewidth=1,
        edgecolor="black",
        facecolor="gray",
    )
    ax2.add_patch(poutre_cote)

    # IPN sous la poutre (vue côté) - Positionné à 20 cm de l'extrémité de la poutre
    ipn_cote = patches.Rectangle(
        (7.0, 0),
        largeur_ipn,
        hauteur_ipn,
        linewidth=1,
        edgecolor="black",
        facecolor="blue",
    )
    ax2.add_patch(ipn_cote)

    # Remblai (vue de côté) - sous le mur de soutènement et entourant les socles des IPN, sans les masquer
    remblai = patches.Polygon(
        [(3.2, 0), (10, 0), (10, -1.5), (7.0 - (taille_socle - largeur_ipn) / 2, -1.5)],
        closed=True,
        linewidth=1,
        edgecolor="black",
        facecolor="yellow",
    )
    ax2.add_patch(remblai)

    # Ajouter les socles sous les IPN dans la vue de côté
    socle_cote = patches.Rectangle(
        (7.0 - (taille_socle - largeur_ipn) / 2, -taille_socle),
        taille_socle,
        taille_socle,
        linewidth=1,
        edgecolor="black",
        facecolor="darkgray",
    )
    ax2.add_patch(socle_cote)

    # Fissure (vue de côté)

    # Coordonnées de départ et de fin pour la fissure
    x_start_fissure = 4.7 - 0.20  # 20 cm de l'axe du mur de soutènement
    x_end_fissure = 4.7 - 0.40  # 40 cm de l'axe du mur de soutènement
    y_start_fissure = hauteur_mur_sout + largeur_poutre  # Dalle de sol
    y_end_fissure = (
        hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit
    )  # Dalle de plafond

    # Nombre de segments pour la fissure
    n_segments = 50
    lengths = np.random.uniform(
        0.005, 0.02, n_segments
    )  # Longueur des segments entre 0.5 cm et 2 cm

    # Normalisation des longueurs pour que la somme couvre toute la hauteur du mur
    total_height = y_end_fissure - y_start_fissure
    lengths = (
        lengths / np.sum(lengths) * total_height
    )  # Les longueurs sont recalibrées pour couvrir toute la hauteur

    # Coordonnées de départ pour la fissure
    x_fissure = [x_start_fissure]
    y_fissure = [y_start_fissure]

    # Génération des segments de la fissure
    max_fluctuation = 0.02  # 2 cm de fluctuation possible

    for i in range(n_segments):
        # Incrément vertical (proportionnel à la longueur générée)
        y_increment = lengths[i]

        # Calcul du déplacement horizontal progressif
        progress_ratio = (
            y_fissure[-1] - y_start_fissure
        ) / total_height  # Progression verticale en pourcentage
        target_x = x_start_fissure + progress_ratio * (
            x_end_fissure - x_start_fissure
        )  # Déplacement progressif vers x_end_fissure

        # Générer une fluctuation horizontale plus importante (jusqu'à 10 cm)
        x_increment = np.random.uniform(-max_fluctuation, max_fluctuation)

        # Calculer les nouvelles coordonnées
        new_x = target_x + x_increment
        new_y = y_fissure[-1] + y_increment

        # Limiter la fissure à ne pas dépasser la dalle de plafond
        if new_y >= y_end_fissure:
            new_y = y_end_fissure
            new_x = x_end_fissure  # Terminer à x_end_fissure

        # Ajouter les nouvelles coordonnées
        x_fissure.append(new_x)
        y_fissure.append(new_y)

        # Si on atteint la dalle de plafond, arrêter la boucle
        if new_y >= y_end_fissure:
            break

    # Génération des épaisseurs variables
    line_widths = np.linspace(0.2, 1.0, len(x_fissure))  # Épaisseur croissante

    # Ajout de la fissure sur l'axe de la vue de côté
    for i in range(1, len(x_fissure)):
        ax2.plot(
            [x_fissure[i - 1], x_fissure[i]],
            [y_fissure[i - 1], y_fissure[i]],
            color="red",
            linewidth=line_widths[i - 1],
        )

    # Annotations pour les polygones sans flèche (colline et remblai)

    # Colline (milieu du polygone)
    ax2.text(
        (appui_colline / 2) - 1,
        hauteur_mur_sout / 2 - 1,
        "Colline",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
    )

    # Remblai (milieu du polygone)
    ax2.text(
        (appui_colline + 7.0) / 2 + 3,
        -taille_socle / 2 + 0.2,
        "Remblai",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )

    # Annotations avec flèches pour les autres éléments

    width = 0.5
    headwidth = 4
    headlength = 6

    # Acrotère
    ax2.annotate(
        "Acrotère",
        xy=(
            8.5,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_toit_total
            + hauteur_acrotere / 2,
        ),
        xytext=(
            13,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_toit_total
            + hauteur_acrotere / 2,
        ),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
    )

    # Gravier
    ax2.annotate(
        "Gravier",
        xy=(
            8,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit
            + epaisseur_gravier / 2,
        ),
        xytext=(13, hauteur_mur_sout + hauteur_batiment + epaisseur_gravier / 2 + 0.2),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
    )

    # Isolation
    ax2.annotate(
        "Isolation",
        xy=(
            8,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit / 2,
        ),
        xytext=(
            13,
            hauteur_mur_sout
            + hauteur_batiment
            + epaisseur_dalle_toit
            + epaisseur_isolation_toit / 2
            - 0.3,
        ),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
    )

    # Dalle plafond
    ax2.annotate(
        "Dalle plafond",
        xy=(8, hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit / 2),
        xytext=(
            13,
            hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit / 2 - 0.6,
        ),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="bottom",
    )

    # Mur du bâtiment
    ax2.annotate(
        "Mur bâtiment",
        xy=(8, hauteur_mur_sout + hauteur_batiment / 2),
        xytext=(13, hauteur_mur_sout + hauteur_batiment / 2),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
    )

    # Dalle de sol (avec flèche)
    ax2.annotate(
        "Dalle de sol",
        xy=(
            8,
            hauteur_mur_sout + largeur_poutre + 0.15 / 2,
        ),  # Milieu de la dalle de sol
        xytext=(
            13,
            hauteur_mur_sout + largeur_poutre + 0.15 / 2 + 0.3,
        ),  # Position de l'annotation
        arrowprops=dict(
            facecolor="black", shrink=0.05, width=0.5, headwidth=4, headlength=6
        ),
        # Propriétés de la flèche
        ha="center",
        va="center",
    )

    # Poutre béton
    ax2.annotate(
        "Poutre béton",
        xy=(8, hauteur_mur_sout + hauteur_poutre / 2),
        xytext=(13, hauteur_mur_sout + hauteur_poutre / 2),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
    )

    # Mur de soutènement
    ax2.annotate(
        "Mur de soutènement",
        xy=(appui_colline + epaisseur_mur_sout / 2, hauteur_mur_sout / 2 - 0.5),
        xytext=(13, hauteur_mur_sout / 2 - 0.5),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
    )

    # IPN
    ax2.annotate(
        "IPN",
        xy=(7.0 + largeur_ipn / 2, hauteur_ipn / 2 + 0.3),
        xytext=(13, hauteur_ipn / 2 + 0.3),
        arrowprops=dict(
            facecolor="blue",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
        color="blue",
    )

    # Socle béton
    ax2.annotate(
        "Socle béton",
        xy=(8 - (taille_socle - largeur_ipn) / 2, -taille_socle / 2 - 0.3),
        xytext=(13, -taille_socle / 2 - 0.3),
        arrowprops=dict(
            facecolor="blue",
            shrink=0.05,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
        ),
        ha="center",
        va="center",
        color="blue",
    )

    # Ajuster les axes et afficher les plans
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # Supprimer les axes pour la vue de face
    ax1.set_axis_off()

    # Supprimer les axes pour la vue de côté
    ax2.set_axis_off()

    plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0, wspace=0.05)
    # plt.show()

    # Sauvegarder la figure dans le répertoire 'docs/'
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    figure_path = os.path.join(docs_dir, "building_plan.png")
    plt.savefig(figure_path, format="png", dpi=300)
    print(f"Figure sauvegardée dans {figure_path}")

    # Convertir la figure en image base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=600)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{image_base64}"


def generate_boxplot_figure(df_paliers_combined):
    """Génère des boxplots normalisés avec Plotly."""

    # Colonnes à standardiser
    columns_to_standardize = [
        "Tassement_Differentiel_IPN",
        "Tassement_Mur",
        "Tassement_Colline",
        "Corrosion_Index",
        "Fatigue_Factor",
        "Degradation_Factor",
    ]

    # Standardisation des colonnes
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df_paliers_combined[columns_to_standardize]),
        columns=columns_to_standardize,
    )

    # Créer la figure de boxplot
    fig = go.Figure()

    for column in columns_to_standardize:
        fig.add_trace(go.Box(y=df_standardized[column], name=column))

    # Mise à jour des axes et du titre
    fig.update_layout(
        title="Boxplots des colonnes standardisées",
        yaxis_title="Valeurs standardisées (Z-score)",
        xaxis_title="Variables",
        height=500,
        font=dict(size=20),
    )

    return fig


def generate_dual_axis_figure(df_paliers_combined):
    """Génère un scatterplot avec deux axes : valeur moyenne des paliers et durée des paliers."""

    fig = go.Figure()

    # Préparation des données pour la valeur moyenne
    x_values = []
    y_values = []
    for _, row in df_paliers_combined.iterrows():
        x_values.extend(
            [row["Début"], row["Fin"], None]
        )  # None pour "casser" la ligne entre les paliers
        y_values.extend([row["Valeur moyenne"], row["Valeur moyenne"], None])

    # Ajout d'une seule trace pour la valeur moyenne
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Valeur moyenne",
        )
    )

    # Calcul de la date centrale pour la durée des paliers
    df_paliers_combined["Date_Centrale"] = (
        df_paliers_combined["Début"]
        + (df_paliers_combined["Fin"] - df_paliers_combined["Début"]) / 2
    )

    # Ajout des points pour la durée des paliers
    fig.add_trace(
        go.Scatter(
            x=df_paliers_combined["Date_Centrale"],
            y=df_paliers_combined["Palier_Duration"],
            mode="markers",
            marker=dict(color="green", size=8),
            name="Durée des paliers",
            yaxis="y2",
        )
    )

    # Mise à jour des axes et du titre
    fig.update_layout(
        title="Évolution des paliers et de la durée des paliers",
        xaxis_title="Temps",
        yaxis=dict(
            title="Valeur moyenne (en mm)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Durée (en jours)",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
        ),
        height=500,
        font=dict(size=20),
    )

    return fig


def generate_scatterplot_grid(df_paliers_combined, threshold=0.80):
    """Génère une grille de scatterplots basée sur le seuil de corrélation."""

    # Colonnes à utiliser pour les scatterplots
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

    # Standardisation des colonnes
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df_paliers_combined[columns]), columns=columns
    )

    # Calcul de la matrice de corrélation
    corr_matrix = df_standardized.corr().abs()

    # Filtrer les paires de colonnes dont la corrélation est inférieure au seuil
    filtered_pairs = [
        (col1, col2)
        for col1 in columns
        for col2 in columns
        if col1 != col2 and corr_matrix.loc[col1, col2] < threshold
    ]

    if not filtered_pairs:
        return go.Figure()

    # Calcul du nombre de lignes et colonnes nécessaires pour la grille
    num_plots = len(filtered_pairs)
    num_cols = int(sqrt(num_plots))
    num_rows = ceil(num_plots / num_cols)

    # Création de la figure avec plusieurs scatterplots
    fig = sp.make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f"{col1} vs {col2}" for col1, col2 in filtered_pairs],
    )

    for i, (col1, col2) in enumerate(filtered_pairs):
        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.add_trace(
            go.Scatter(
                x=df_standardized[col1],
                y=df_standardized[col2],
                mode="markers",
                name=f"{col1} vs {col2}",
            ),
            row=row,
            col=col,
        )

    # Mise en page de la figure
    fig.update_layout(
        title=f"Pairplot : seuil du coefficient de Pearson à {threshold} (abscisses vs ordonnées)",
        height=300 * num_rows,
        showlegend=False,
        font=dict(size=20),
    )

    return fig
