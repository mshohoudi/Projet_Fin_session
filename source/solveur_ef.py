import numpy as np


class SolveurFEA1D:
    def __init__(self, longueur, base, hauteur, module_young, coeff_thermique, nb_elements=20):
        """
        Initialisation du solveur Éléments Finis 1D pur (Euler-Bernoulli).
        Les propriétés physiques sont injectées dynamiquement via le fichier YAML.
        """
        # Conversion explicite pour éviter les erreurs TypeError avec YAML
        self.L = float(longueur)
        self.b = float(base)
        self.h = float(hauteur)
        self.E = float(module_young)
        self.alpha = float(coeff_thermique)

        # Paramètres de maillage (Maillage 1D)
        self.ne = int(nb_elements)
        self.nn = self.ne + 1
        self.le = self.L / self.ne

        self.I = (self.b * (self.h ** 3)) / 12.0
        self.A = self.b * self.h

    def calculer_resultats(self, force_kn, delta_temperature):
        """
        Résout le système FEA sans facteurs de calibration artificiels
        pour comparer équitablement avec le modèle ML.
        """
        force_n = float(force_kn) * 1000.0
        ndof = 2 * self.nn
        K_global = np.zeros((ndof, ndof))
        F_global = np.zeros(ndof)

        # Matrice de rigidité locale
        coef = (self.E * self.I) / (self.le ** 3)
        k_element = coef * np.array([
            [12, 6 * self.le, -12, 6 * self.le],
            [6 * self.le, 4 * (self.le ** 2), -6 * self.le, 2 * (self.le ** 2)],
            [-12, -6 * self.le, 12, -6 * self.le],
            [6 * self.le, 2 * (self.le ** 2), -6 * self.le, 4 * (self.le ** 2)]
        ])

        # Assemblage de la matrice
        for i in range(self.ne):
            noeuds_dof = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
            for r in range(4):
                for c in range(4):
                    K_global[noeuds_dof[r], noeuds_dof[c]] += k_element[r, c]

        # Application de la charge (Force Y sur le dernier nœud)
        F_global[-2] = force_n

        # Conditions aux limites (Encastrement : on supprime les 2 premiers DDL)
        K_reduit = K_global[2:, 2:]
        F_reduit = F_global[2:]

        # Résolution du système linéaire
        U_reduit = np.linalg.solve(K_reduit, F_reduit)

        # Reconstitution du vecteur global
        U_global = np.zeros(ndof)
        U_global[2:] = U_reduit

        # 1. Déplacement théorique brut (Axe Y purement)
        deplacement_brut = abs(float(U_global[-2]))

        # 2. Contrainte calculée à X = 0.09 (Filtre de Saint-Venant)
        reactions = K_global.dot(U_global)
        moment_encastrement = abs(float(reactions[1]))
        moment_x = moment_encastrement - (force_n * 0.09)
        contrainte_fea_brute = (moment_x * (self.h / 2.0)) / self.I

        return deplacement_brut, contrainte_fea_brute