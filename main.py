import os
import yaml
import numpy as np
from source.gestion_donnees import GestionnaireDonnees
from source.modele_substitution import ModeleSubstitution
from source.visualisation import Visualiseur3D
from source.solveur_analytique import SolveurAnalytique
from source.solveur_ef import SolveurFEA1D


def charger_configuration():
    """Charge les paramètres depuis le fichier config.yaml"""
    chemin_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    with open(chemin_config, 'r', encoding='utf-8') as fichier:
        return yaml.safe_load(fichier)


def main():
    print("=== Lancement du Système Thermoélastique (Couplage FEA & ML) ===\n")

    # --- LECTURE DU FICHIER YAML ---
    try:
        config = charger_configuration()
        print("[ÉTAPE 0] Configuration YAML chargée avec succès.")
    except Exception as e:
        print(f" [ERREUR] Impossible de charger config.yaml : {e}")
        return

    chemin_dossier_actuel = os.path.dirname(os.path.abspath(__file__))
    nom_dossier = config['chemins']['dossier_donnees'].replace('./', '')
    dossier_donnees = os.path.join(chemin_dossier_actuel, nom_dossier)

    print("\n[ÉTAPE 1] Importation des données paramétriques...")
    try:
        gestionnaire = GestionnaireDonnees(dossier_donnees)
        x_entrees, y_sorties = gestionnaire.preparer_donnees_ml()
        print(" -> Succès ! Les données sont prêtes pour l'entraînement.")
    except Exception as e:
        print(f" [ERREUR] Problème lors de la lecture des données : {e}")
        return

    print("\n[ÉTAPE 2] Entraînement du modèle de substitution (Machine Learning)...")
    params_ml = config['parametres_ml']
    modele_ia = ModeleSubstitution(
        n_estimators=params_ml['n_estimators'],
        k_fold_splits=params_ml['k_fold_splits'],
        random_state=params_ml['random_state']
    )
    modele_ia.entrainer(x_entrees, y_sorties)

    print("\n[ÉTAPE 3] Mode Interactif - Saisie Utilisateur")
    try:
        print("\n---------------------------------------------------")
        force_utilisateur = float(input(" -> Entrez la force appliquée (en kN, entre 10 et 40) : "))
        temp_utilisateur = float(input(" -> Entrez la température (en °C, entre 20 et 200) : "))
        print("---------------------------------------------------\n")
    except ValueError:
        print(" [ERREUR] Veuillez entrer des nombres valides.")
        return

    temp_ambiante = 20.0
    delta_t = temp_utilisateur - temp_ambiante

    print(
        f" -> Génération des prédictions et calculs pour F = {force_utilisateur} kN et T = {temp_utilisateur} °C ...\n")

    # --- 1. Résolution par Intelligence Artificielle (ML) ---
    noeuds_geometrie = x_entrees[:621, :3]
    colonne_force = np.full((noeuds_geometrie.shape[0], 1), force_utilisateur)
    colonne_temp = np.full((noeuds_geometrie.shape[0], 1), temp_utilisateur)
    entrees_prediction = np.hstack((noeuds_geometrie, colonne_force, colonne_temp))

    y_predictions = modele_ia.predire(entrees_prediction)
    contraintes_vm_predites = y_predictions[:, 0]
    contraintes_x_predites = y_predictions[:, 1]
    deplacements_predits = y_predictions[:, 2]

    # --- APPLICATION DU PRINCIPE DE SAINT-VENANT (Depuis YAML) ---
    filtre_sv = config['parametres_ml']['filtre_saint_venant']
    masque_hors_singularite = noeuds_geometrie[:, 0] > filtre_sv

    max_contrainte_vm_ml = np.max(contraintes_vm_predites[masque_hors_singularite])
    max_contrainte_x_ml = np.max(np.abs(contraintes_x_predites[masque_hors_singularite]))
    max_deplacement_ml = np.max(np.abs(deplacements_predits))

    # --- EXTRACTION DES PROPRIÉTÉS PHYSIQUES (Depuis YAML) ---
    geom = config['geometrie']
    mat = config['materiau']

    # --- 2. Résolution par Théorie Analytique (Euler-Bernoulli) ---
    solveur_ana = SolveurAnalytique(
        longueur=geom['longueur'], base=geom['base'], hauteur=geom['hauteur'],
        module_young=mat['module_young'], coeff_thermique=mat['coeff_thermique']
    )
    max_deplacement_ana = solveur_ana.calculer_deplacement_max(force_utilisateur, delta_t)
    max_contrainte_ana = solveur_ana.calculer_contrainte_max(force_utilisateur)
    max_contrainte_vm_ana = abs(max_contrainte_ana)

    # --- 3. Résolution par Éléments Finis 1D (FEA) ---
    solveur_fea = SolveurFEA1D(
        longueur=geom['longueur'], base=geom['base'], hauteur=geom['hauteur'],
        module_young=mat['module_young'], coeff_thermique=mat['coeff_thermique']
    )
    max_deplacement_fea, max_contrainte_x_fea = solveur_fea.calculer_resultats(force_utilisateur, delta_t)

    # --- AFFICHAGE DU RAPPORT COMPARATIF ---
    print("==========================================================")
    print("        RAPPORT DE VALIDATION CROISÉE (TRIPLE-CHECK)      ")
    print("==========================================================")

    print("[1] DÉPLACEMENT DIRECTIONNEL MAXIMAL - AXE Y (m) :")
    print(f"    - Analytique (Euler-Bernoulli) : {max_deplacement_ana:.6f} m")
    print(f"    - Numérique (FEA 1D)           : {max_deplacement_fea:.6f} m")
    print(f"    - Prédiction IA (ML 3D)        : {max_deplacement_ml:.6f} m\n")

    print("[2] CONTRAINTE NORMALE MAXIMALE - AXE X (Pa) :")
    print(f"    - Analytique (Théorie Flexion) : {max_contrainte_ana:.2f} Pa")
    print(f"    - Numérique (FEA 1D)           : {max_contrainte_x_fea:.2f} Pa")
    print(f"    - Prédiction IA (ML 3D)* : {max_contrainte_x_ml:.2f} Pa\n")

    print("[3] CONTRAINTE DE VON-MISES MAXIMALE (Pa) :")
    print(f"    - Analytique (Théorie 1D)      : {max_contrainte_vm_ana:.2f} Pa")
    print(f"    - Prédiction IA (ML 3D)* : {max_contrainte_vm_ml:.2f} Pa")
    print(f"  * Note: IA filtrée par le principe de Saint-Venant (X > {filtre_sv}m)")
    print("==========================================================\n")

    print("[ÉTAPE 4] Affichage des Heatmaps 3D...")
    visualiseur = Visualiseur3D()
    visualiseur.tracer_heatmap_3d(noeuds_geometrie, deplacements_predits,
                                  f"Déplacements Prédits Y (ML) | F={force_utilisateur}kN, T={temp_utilisateur}°C",
                                  "Déplacement Y (m)")
    visualiseur.tracer_heatmap_3d(noeuds_geometrie, contraintes_x_predites,
                                  f"Contrainte Normale X (ML) | F={force_utilisateur}kN, T={temp_utilisateur}°C",
                                  "Contrainte X (Pa)")
    visualiseur.tracer_heatmap_3d(noeuds_geometrie, contraintes_vm_predites,
                                  f"Contrainte Von-Mises (ML) | F={force_utilisateur}kN, T={temp_utilisateur}°C",
                                  "Contrainte VM (Pa)")
    print(" -> Processus terminé avec succès !")


if __name__ == "__main__":
    main()