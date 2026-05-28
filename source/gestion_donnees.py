import pandas as pd
import numpy as np

class GestionnaireDonnees:
    def __init__(self, chemin_contrainte, chemin_deplacement):
        # Initialisation des chemins vers les fichiers de données brutes ANSYS
        self.chemin_contrainte = chemin_contrainte
        self.chemin_deplacement = chemin_deplacement
        self.donnees_combinees = None

    def charger_et_fusionner(self):
        # Charger le fichier des contraintes de Von-Mises
        # sep='\t' indique que les colonnes sont séparées par des tabulations
        # skiprows=1 permet d'ignorer la ligne d'en-tête textuelle générée par ANSYS
        df_contrainte = pd.read_csv(
            self.chemin_contrainte,
            sep='\t',
            skiprows=1,
            names=['Noeud', 'X', 'Y', 'Z', 'Contrainte']
        )

        # Charger le fichier des déplacements totaux
        df_deplacement = pd.read_csv(
            self.chemin_deplacement,
            sep='\t',
            skiprows=1,
            names=['Noeud', 'X', 'Y', 'Z', 'Deplacement']
        )

        # Fusionner les deux ensembles de données en utilisant le numéro de nœud (Noeud)
        # On ne conserve que la colonne 'Deplacement' du second fichier pour éviter les doublons spatiaux
        self.donnees_combinees = pd.merge(
            df_contrainte,
            df_deplacement[['Noeud', 'Deplacement']],
            on='Noeud'
        )

        return self.donnees_combinees

    def preparer_donnees_ml(self):
        # Séparer les caractéristiques (Features) et les cibles (Targets) pour le Machine Learning
        if self.donnees_combinees is None:
            self.charger_et_fusionner()

        # Matrice des entrées : Coordonnées spatiales (X, Y, Z)
        x_entrees = self.donnees_combinees[['X', 'Y', 'Z']].values

        # Matrice des sorties : Ce que le modèle de substitution doit apprendre à prédire
        y_sorties = self.donnees_combinees[['Contrainte', 'Deplacement']].values

        return x_entrees, y_sorties