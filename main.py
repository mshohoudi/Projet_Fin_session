import os
from source.gestion_donnees import GestionnaireDonnees
from source.modele_substitution import ModeleSubstitution
from source.visualisation import Visualiseur3D


def main():
    print("=== Lancement du Système Thermoélastique (Couplage FEA & ML) ===\n")

    # 1. Définir les chemins absolus vers les fichiers de données
    chemin_dossier_actuel = os.path.dirname(os.path.abspath(__file__))
    chemin_contrainte = os.path.join(chemin_dossier_actuel, "donnees", "contrainte_ansys.txt")
    chemin_deplacement = os.path.join(chemin_dossier_actuel, "donnees", "deplacement_ansys.txt")

    print("[ÉTAPE 1] Importation et traitement des données brutes ANSYS...")
    try:
        # Initialiser la classe de gestion des données
        gestionnaire = GestionnaireDonnees(chemin_contrainte, chemin_deplacement)
        x_entrees, y_sorties = gestionnaire.preparer_donnees_ml()
        print(" -> Succès ! Les données sont prêtes pour l'entraînement.")
        print(f" -> Nombre total de nœuds analysés : {x_entrees.shape[0]}")

    except FileNotFoundError:
        print(" [ERREUR] Les fichiers .txt sont introuvables dans le dossier 'donnees/'.")
        return
    except Exception as e:
        print(f" [ERREUR] Un problème inattendu est survenu : {e}")
        return

    print("\n[ÉTAPE 2] Initialisation et entraînement du modèle de substitution (Machine Learning)...")
    # Initialiser et entraîner le modèle d'intelligence artificielle
    modele_ia = ModeleSubstitution()
    modele_ia.entrainer(x_entrees, y_sorties)

    print("\n[ÉTAPE 3] Génération des prédictions et affichage des heatmaps 3D...")
    try:
        # Utiliser le modèle entraîné pour prédire sur l'ensemble de la géométrie
        y_predictions = modele_ia.predire(x_entrees)

        # Séparation des prédictions (Colonne 0 = Contrainte, Colonne 1 = Déplacement)
        contraintes_predites = y_predictions[:, 0]
        deplacements_predits = y_predictions[:, 1]

        # Initialiser l'outil de visualisation
        visualiseur = Visualiseur3D()

        # 1er Graphique : Heatmap 3D des Contraintes de Von-Mises
        visualiseur.tracer_heatmap_3d(
            coordonnees=x_entrees,
            valeurs=contraintes_predites,
            titre="Modèle de Substitution : Contraintes de Von-Mises Prédites (ML)",
            nom_unite="Contrainte (Pa)"
        )

        # 2ème Graphique : Heatmap 3D des Déplacements Totaux
        visualiseur.tracer_heatmap_3d(
            coordonnees=x_entrees,
            valeurs=deplacements_predits,
            titre="Modèle de Substitution : Déplacements Totaux Prédits (ML)",
            nom_unite="Déplacement (m)"
        )

        print(" -> Succès ! Les graphiques 3D ont été générés.")

    except Exception as e:
        print(f" [ERREUR] Impossible de générer les visualisations : {e}")


if __name__ == "__main__":
    main()