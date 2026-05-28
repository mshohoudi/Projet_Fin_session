import os
import numpy as np
from source.gestion_donnees import GestionnaireDonnees
from source.modele_substitution import ModeleSubstitution
from source.visualisation import Visualiseur3D


def main():
    print("=== Lancement du Système Thermoélastique (Couplage FEA & ML) ===\n")

    # Définir le chemin absolu vers le dossier des données
    chemin_dossier_actuel = os.path.dirname(os.path.abspath(__file__))
    dossier_donnees = os.path.join(chemin_dossier_actuel, "donnees")

    print("[ÉTAPE 1] Importation des données paramétriques (32 fichiers)...")
    try:
        # Initialiser le gestionnaire de données et préparer les ensembles pour le ML
        gestionnaire = GestionnaireDonnees(dossier_donnees)
        x_entrees, y_sorties = gestionnaire.preparer_donnees_ml()
        print(" -> Succès ! Les données sont prêtes pour l'entraînement.")
        print(f" -> Taille totale de la base de données : {x_entrees.shape[0]} points analysés.")
    except Exception as e:
        print(f" [ERREUR] Problème lors de la lecture des données : {e}")
        return

    print("\n[ÉTAPE 2] Entraînement du modèle de substitution (Machine Learning)...")
    # Initialiser et entraîner le modèle d'intelligence artificielle (Random Forest)
    modele_ia = ModeleSubstitution()
    modele_ia.entrainer(x_entrees, y_sorties)

    print("\n[ÉTAPE 3] Mode Interactif - Saisie Utilisateur")
    # Obtenir la force et la température de l'utilisateur via la console
    try:
        print("\n---------------------------------------------------")
        force_utilisateur = float(input(" -> Entrez la force appliquée (en kN, entre 10 et 40) : "))
        temp_utilisateur = float(input(" -> Entrez la température (en °C, entre 20 et 200) : "))
        print("---------------------------------------------------\n")
    except ValueError:
        print(" [ERREUR] Veuillez entrer des nombres valides.")
        return

    print(
        f" -> Génération des prédictions instantanées pour F = {force_utilisateur} kN et T = {temp_utilisateur} °C ...")

    # Extraire les coordonnées géométriques (seulement les colonnes X, Y, Z des 621 premiers nœuds)
    noeuds_geometrie = x_entrees[:621, :3]

    # Créer des colonnes contenant les valeurs de l'utilisateur pour chaque point de la géométrie
    colonne_force = np.full((noeuds_geometrie.shape[0], 1), force_utilisateur)
    colonne_temp = np.full((noeuds_geometrie.shape[0], 1), temp_utilisateur)

    # Construire la matrice d'entrée finale à 5 dimensions (X, Y, Z, Force, Température) pour la prédiction
    entrees_prediction = np.hstack((noeuds_geometrie, colonne_force, colonne_temp))

    # Prédire instantanément la distribution des contraintes et des déplacements
    y_predictions = modele_ia.predire(entrees_prediction)

    contraintes_predites = y_predictions[:, 0]
    deplacements_predits = y_predictions[:, 1]

    # --- NOUVEAU CODE : Calcul et affichage des valeurs maximales ---
    max_contrainte = np.max(contraintes_predites)
    max_deplacement = np.max(deplacements_predits)

    print(f"\n[RÉSULTATS CLÉS] Pour F = {force_utilisateur} kN et T = {temp_utilisateur} °C :")
    print(f" -> Contrainte de Von-Mises Maximale : {max_contrainte:.2f} Pa")
    print(f" -> Déplacement Total Maximal        : {max_deplacement:.6f} m")
    print("---------------------------------------------------\n")
    # ---------------------------------------------------------------

    print("[ÉTAPE 4] Affichage des Heatmaps 3D...")
    visualiseur = Visualiseur3D()

    # Générer le graphique 3D pour la distribution des contraintes
    visualiseur.tracer_heatmap_3d(
        coordonnees=noeuds_geometrie,
        valeurs=contraintes_predites,
        titre=f"Contraintes Prédites (ML) | F = {force_utilisateur} kN, T = {temp_utilisateur} °C",
        nom_unite="Contrainte (Pa)"
    )

    # Générer le graphique 3D pour les déplacements totaux
    visualiseur.tracer_heatmap_3d(
        coordonnees=noeuds_geometrie,
        valeurs=deplacements_predits,
        titre=f"Déplacements Prédits (ML) | F = {force_utilisateur} kN, T = {temp_utilisateur} °C",
        nom_unite="Déplacement (m)"
    )

    print(" -> Processus terminé avec succès !")


if __name__ == "__main__":
    main()