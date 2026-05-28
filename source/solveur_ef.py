import numpy as np


class SolveurFEA1D:
    def __init__(self, longueur=1.0, base=0.05, hauteur=0.1, module_young=200e9, coeff_thermique=1.2e-5,
                 nb_elements=20):
        """
        Initialisation du solveur Éléments Finis 1D pour une poutre Euler-Bernoulli.
        """
        self.L = longueur
        self.b = base
        self.h = hauteur
        self.E = module_young
        self.alpha = coeff_thermique
        self.ne = nb_elements  # Nombre d'éléments
        self.nn = nb_elements + 1  # Nombre de nœuds
        self.le = longueur / nb_elements  # Longueur de chaque élément

        # Propriétés de la section
        self.I = (self.b * (self.h ** 3)) / 12.0
        self.A = self.b * self.h

    def calculer_deplacement_max(self, force_kn, delta_temperature):
        """
        Résout le système FEA pour trouver le déplacement mécanique et y ajoute la dilatation thermique.
        """
        force_n = force_kn * 1000.0

        # --- 1. PARTIE MÉCANIQUE (Flexion - Euler-Bernoulli) ---
        # 2 Degrés de Liberté (DDL) par nœud : Déplacement vertical (v) et Rotation (theta)
        ndof = 2 * self.nn
        K_global = np.zeros((ndof, ndof))
        F_global = np.zeros(ndof)

        # Matrice de rigidité locale d'un élément poutre (Euler-Bernoulli)
        coef = (self.E * self.I) / (self.le ** 3)
        k_element = coef * np.array([
            [12, 6 * self.le, -12, 6 * self.le],
            [6 * self.le, 4 * (self.le ** 2), -6 * self.le, 2 * (self.le ** 2)],
            [-12, -6 * self.le, 12, -6 * self.le],
            [6 * self.le, 2 * (self.le ** 2), -6 * self.le, 4 * (self.le ** 2)]
        ])

        # Assemblage de la matrice de rigidité globale
        for i in range(self.ne):
            noeuds_dof = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
            for r in range(4):
                for c in range(4):
                    K_global[noeuds_dof[r], noeuds_dof[c]] += k_element[r, c]

        # Application de la charge (Force ponctuelle à l'extrémité libre, direction Y)
        # L'extrémité libre est le dernier nœud, le DDL vertical est l'avant-dernier de la matrice (ndof-2)
        F_global[-2] = force_n

        # Conditions aux limites : Encastrement au nœud 0 (X=0) -> v=0, theta=0
        # On supprime les 2 premières lignes et colonnes (Méthode de réduction)
        K_reduit = K_global[2:, 2:]
        F_reduit = F_global[2:]

        # Résolution du système linéaire (K * U = F)
        U_reduit = np.linalg.solve(K_reduit, F_reduit)

        # Reconstitution du vecteur de déplacement complet
        U_global = np.zeros(ndof)
        U_global[2:] = U_reduit

        # Le déplacement vertical maximum (flèche) est au bout de la poutre
        deplacement_mecanique = abs(U_global[-2])

        # --- 2. PARTIE THERMIQUE (Dilatation axiale) ---
        # Calculée de manière découplée (1D axial)
        deplacement_thermique = self.alpha * self.L * delta_temperature

        # Superposition du déplacement transversal et longitudinal (Norme)
        deplacement_total = np.sqrt(deplacement_mecanique ** 2 + deplacement_thermique ** 2)

        return deplacement_total