import os
import pandas as pd
import numpy as np


class GestionnaireDonnees:
    def __init__(self, dossier_donnees):
        self.dossier_donnees = dossier_donnees
        self.donnees_combinees = None

    def charger_tous_les_fichiers(self):
        toutes_les_donnees = []
        fichiers = os.listdir(self.dossier_donnees)

        # Filtrer uniquement les fichiers de contrainte Von-Mises
        fichiers_contrainte = [f for f in fichiers if f.startswith('contrainte_F') and f.endswith('.txt')]

        for fichier_c in fichiers_contrainte:
            nom_sans_extension = fichier_c.replace('.txt', '')
            parties = nom_sans_extension.split('_')

            force = float(parties[1].replace('F', ''))
            temperature = float(parties[2].replace('T', ''))

            chemin_c = os.path.join(self.dossier_donnees, fichier_c)

            # Chercher le fichier de déplacement Y
            fichier_d = fichier_c.replace('contrainte', 'deplacement_y')
            chemin_d = os.path.join(self.dossier_donnees, fichier_d)

            # Chercher le fichier de contrainte X
            fichier_cx = fichier_c.replace('contrainte', 'contrainte_x')
            chemin_cx = os.path.join(self.dossier_donnees, fichier_cx)

            if not os.path.exists(chemin_d) or not os.path.exists(chemin_cx):
                print(f" [AVERTISSEMENT] Fichiers manquants pour {fichier_c}. On l'ignore.")
                continue

            try:
                # skiprows=1 permet d'ignorer la ligne d'en-tête contenant du texte
                df_c = pd.read_csv(chemin_c, sep='\t', skiprows=1, names=['Noeud', 'X', 'Y', 'Z', 'Contrainte'])
                df_d = pd.read_csv(chemin_d, sep='\t', skiprows=1, names=['Noeud', 'X', 'Y', 'Z', 'Deplacement'])
                df_cx = pd.read_csv(chemin_cx, sep='\t', skiprows=1, names=['Noeud', 'X', 'Y', 'Z', 'Contrainte_X'])
            except Exception as e:
                print(f" [ERREUR] Format incorrect dans les fichiers associés à {fichier_c}: {e}")
                continue

            # Fusion des données
            df_fusion = pd.merge(df_c, df_d[['Noeud', 'Deplacement']], on='Noeud')
            df_fusion = pd.merge(df_fusion, df_cx[['Noeud', 'Contrainte_X']], on='Noeud')

            df_fusion['Force'] = force
            df_fusion['Temperature'] = temperature

            toutes_les_donnees.append(df_fusion)

        if not toutes_les_donnees:
            raise ValueError("Aucune donnée valide n'a pu être chargée. Vérifiez les noms et le format de vos fichiers.")

        self.donnees_combinees = pd.concat(toutes_les_donnees, ignore_index=True)
        print(f" -> {len(toutes_les_donnees)} configurations chargées avec succès.")

        return self.donnees_combinees

    def preparer_donnees_ml(self):
        if self.donnees_combinees is None:
            self.charger_tous_les_fichiers()

        x_entrees = self.donnees_combinees[['X', 'Y', 'Z', 'Force', 'Temperature']].values
        y_sorties = self.donnees_combinees[['Contrainte', 'Contrainte_X', 'Deplacement']].values

        return x_entrees, y_sorties