import matplotlib.patches as patches
import matplotlib.pyplot as plt

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

# Couches du toit (vue de côté) - correcte alignement des dimensions
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
    (epaisseur_mur_sout, hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit),
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

# Colline (vue de côté) - pente de la colline, partant du mur de gauche et atteignant 2,4 m sous le mur de soutènement
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
    (7.0, 0), largeur_ipn, hauteur_ipn, linewidth=1, edgecolor="black", facecolor="blue"
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
    va="bottom",
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
    xytext=(13, hauteur_mur_sout + hauteur_batiment + epaisseur_dalle_toit / 2 - 0.6),
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
    xy=(8, hauteur_mur_sout + largeur_poutre + 0.15 / 2),  # Milieu de la dalle de sol
    xytext=(
        13,
        hauteur_mur_sout + largeur_poutre + 0.15 / 2 + 0.3,
    ),  # Position de l'annotation
    arrowprops=dict(
        facecolor="black", shrink=0.05, width=0.5, headwidth=4, headlength=6
    ),  # Propriétés de la flèche
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
        facecolor="black",
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
        facecolor="black",
        shrink=0.05,
        width=width,
        headwidth=headwidth,
        headlength=headlength,
    ),
    ha="center",
    va="center",
)


# Ajuster les axes et afficher les plans
ax1.set_aspect("equal")
ax2.set_aspect("equal")

# Supprimer les axes pour la vue de face
ax1.set_axis_off()

# Supprimer les axes pour la vue de côté
ax2.set_axis_off()


plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0, wspace=0.05)
plt.show()
