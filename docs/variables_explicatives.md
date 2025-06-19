


Variables explicatives prises en compte pour la structure :

    Building_Age : L'âge du bâtiment, depuis sa construction.
    Corrosion_Index : La corrosion potentielle en fonction de l'âge du bâtiment.
    Fatigue_Factor : Un facteur de fatigue basé sur l'âge.
    Degradation_Factor : Un facteur de dégradation qui diminue exponentiellement avec l'âge.
    IPN_Rigidite_Flexion : La rigidité de flexion des IPN, calculée à partir de leur moment d'inertie et du module de Young de l'acier.
    IPN_Stress_Factor : Le facteur de contrainte des IPN, basé sur la charge par section.
    Tassement_Differentiel_IPN : Un facteur logarithmique pour modéliser le tassement différentiel des IPN.
    Soutien_Mur_Tassement : Un facteur pour modéliser le tassement du mur de soutènement.
    Palier_Duration : La durée des paliers en jours.

Le code est maintenant complet pour la modélisation des paliers d'écartement en prenant en compte la structure en IPN, le tassement différentiel, ainsi que la résistance des matériaux. 