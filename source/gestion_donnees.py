import os
import pandas as pd
import numpy as np


class GestionnaireDonnees:
    def __init__(self, dossier_donnees):
        # Maintenant, nous ne prenons que le chemin du dossier contenant les 32 fichiers
        self.dossier_donnees = dossier_donnees
        self.donnees_combinees = None

    def charger_tous_les_fichiers(self):
        toutes_les_donnees = []

        # Lister tous les fichiers dans le dossier
        fichiers = os.listdir(self.dossier_donnees)

        # Filtrer uniquement les fichiers de contrainte
        fichiers_contrainte = [f for f in fichiers if f.startswith('contrainte_') and f.endswith('.txt')]

        for fichier_c in fichiers_contrainte:
            # Extraire la Force et la Température du nom du fichier (ex: contrainte_F10_T20.txt)
            nom_sans_extension = fichier_c.replace('.txt', '')
            parties = nom_sans_extension.split('_')

            force = float(parties[1].replace('F', ''))
            temperature = float(parties[2].replace('T', ''))

            chemin_c = os.path.join(self.dossier_donnees, fichier_c)

            # Trouver le fichier de déplacement correspondant
            fichier_d = fichier_c.replace('contrainte', 'deplacement')
            chemin_d = os.path.join(self.dossier_donnees, fichier_d)

            if not os.path.exists(chemin_d):
                print(f" [AVERTISSEMENT] Le fichier {fichier_d} est introuvable. On l'ignore.")
                continue

            # Charger les données brutes
            df_c = pd.read_csv(chemin_c, sep='\t', skiprows=1, names=['Noeud', 'X', 'Y', 'Z', 'Contrainte'])
            df_d = pd.read_csv(chemin_d, sep='\t', skiprows=1, names=['Noeud', 'X', 'Y', 'Z', 'Deplacement'])

            # Fusionner selon le numéro de nœud
            df_fusion = pd.merge(df_c, df_d[['Noeud', 'Deplacement']], on='Noeud')

            # Ajouter les nouvelles variables paramétriques (Features)
            df_fusion['Force'] = force
            df_fusion['Temperature'] = temperature

            toutes_les_donnees.append(df_fusion)

        # Combiner les 16 DataFrames en un seul super-DataFrame pour l'entraînement
        self.donnees_combinees = pd.concat(toutes_les_donnees, ignore_index=True)
        print(f" -> {len(fichiers_contrainte)} configurations chargées avec succès.")

        return self.donnees_combinees

    def preparer_donnees_ml(self):
        # Séparer les caractéristiques (Features) et les cibles (Targets)
        if self.donnees_combinees is None:
            self.charger_tous_les_fichiers()

        # NOUVEAU FORMAT D'ENTRÉE : 5 dimensions (X, Y, Z, Force, Température)
        x_entrees = self.donnees_combinees[['X', 'Y', 'Z', 'Force', 'Temperature']].values

        # SORTIES : 2 dimensions (Contrainte, Déplacement)
        y_sorties = self.donnees_combinees[['Contrainte', 'Deplacement']].values

        return x_entrees, y_sorties