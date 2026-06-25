import numpy as np


class SolveurEF:
    """
    Solveur par éléments finis 1D pour une poutre d'Euler-Bernoulli
    encastrée-libre.

    Le modèle utilise deux degrés de liberté par nœud : le déplacement
    transversal ``v`` et la rotation ``theta``. Chaque élément possède donc
    quatre degrés de liberté locaux : ``[v1, theta1, v2, theta2]``.

    """

    def __init__(self, cfg: dict):
        geo = cfg["geometrie"]
        mat = cfg["materiau"]
        ef  = cfg["elements_finis"]

        # Conversion explicite en float pour éviter les erreurs YAML
        self.L   = float(geo["longueur"])
        self.b   = float(geo["base"])
        self.h   = float(geo["hauteur"])
        self.E   = float(mat["module_young"])
        self.nu  = float(mat["poisson"])
        self.alpha = float(mat["coeff_thermique"])
        self.T0    = float(mat["temperature_ref"])
        self.nb_el = int(ef["nb_elements"])

        self.I  = (self.b * self.h**3) / 12.0
        self.A  = self.b * self.h
        self.le = self.L / self.nb_el           # Longueur d'un élément
        self.nb_noeuds = self.nb_el + 1
        self.nb_ddl    = 2 * self.nb_noeuds     # v + θ par noeud

    # ---------------------------------------------------------------- #
    #  Matrice de rigidité élémentaire (4×4)
    # ---------------------------------------------------------------- #

    def _matrice_rigidite_elem(self) -> np.ndarray:
        """
        Calcule la matrice de rigidité élémentaire d'un élément poutre.

        La formulation utilisée est celle d'un élément poutre d'Euler-Bernoulli
        avec quatre degrés de liberté locaux.

        :return: Matrice de rigidité élémentaire de dimension ``4 x 4``.
        :rtype: numpy.ndarray
        """
        EI = self.E * self.I
        l  = self.le
        l2 = l**2
        l3 = l**3

        ke = (EI / l3) * np.array([
            [ 12,    6*l,  -12,   6*l  ],
            [  6*l,  4*l2,  -6*l,  2*l2],
            [-12,   -6*l,   12,  -6*l  ],
            [  6*l,  2*l2,  -6*l,  4*l2],
        ])
        return ke

    # ---------------------------------------------------------------- #
    #  Assemblage de la matrice globale
    # ---------------------------------------------------------------- #

    def _assembler(self) -> np.ndarray:
        """
        Assemble la matrice de rigidité globale du système.

        Les matrices élémentaires sont ajoutées à la matrice globale selon la
        connectivité des degrés de liberté de chaque élément.

        :return: Matrice de rigidité globale de dimension
                 ``nb_ddl x nb_ddl``.
        :rtype: numpy.ndarray
        """
        K = np.zeros((self.nb_ddl, self.nb_ddl))
        ke = self._matrice_rigidite_elem()

        for e in range(self.nb_el):
            # DDL globaux associés à l'élément e : [2e, 2e+1, 2e+2, 2e+3]
            dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += ke[i, j]
        return K

    # ---------------------------------------------------------------- #
    #  Vecteur de forces nodales
    # ---------------------------------------------------------------- #

    def _vecteur_forces(self, F: float, T: float) -> np.ndarray:
        """
        Construit le vecteur global des forces nodales.

        La force ponctuelle est appliquée sur le degré de liberté de déplacement
        transversal du dernier nœud. Dans l'implémentation actuelle, la température
        n'est pas transformée en force thermique équivalente; elle est seulement
        utilisée en post-traitement pour calculer la contrainte thermique.

        :param F: Force ponctuelle appliquée à l'extrémité libre [N].
        :type F: float
        :param T: Température appliquée [°C].
        :type T: float
        :return: Vecteur global des forces nodales.
        :rtype: numpy.ndarray
        """
        f = np.zeros(self.nb_ddl)
        # Force concentrée en v du dernier nœud
        f[-2] = -F   # Négatif car convention y positif vers le haut, charge vers le bas
        return f

    # ---------------------------------------------------------------- #
    #  Application des conditions aux limites (encastrement en x=0)
    # ---------------------------------------------------------------- #

    def _appliquer_cl(self, K: np.ndarray, f: np.ndarray):
        """
        Applique les conditions aux limites de l'encastrement.

        L'encastrement est imposé au premier nœud en bloquant le déplacement
        transversal et la rotation :

        .. math::

            v_0 = 0

        .. math::

            \\theta_0 = 0

        Les conditions sont imposées par une méthode de pénalité.

        :param K: Matrice de rigidité globale initiale.
        :type K: numpy.ndarray
        :param f: Vecteur global des forces nodales initial.
        :type f: numpy.ndarray
        :return: Matrice de rigidité et vecteur de forces modifiés.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        K_mod = K.copy()
        f_mod = f.copy()
        penalite = 1.0e30

        # DDL bloqués : 0 (v_0) et 1 (θ_0)
        for dof in [0, 1]:
            K_mod[dof, :] = 0.0
            K_mod[:, dof] = 0.0
            K_mod[dof, dof] = penalite
            f_mod[dof] = 0.0

        return K_mod, f_mod

    # ---------------------------------------------------------------- #
    #  Résolution et post-traitement
    # ---------------------------------------------------------------- #

    def resoudre(self, F: float, T: float) -> dict:
        """
        Résout le système éléments finis et effectue le post-traitement.

        Le système linéaire résolu est :

        .. math::

            KU = f

        où ``K`` est la matrice de rigidité globale, ``U`` le vecteur des
        déplacements et rotations nodales, et ``f`` le vecteur des forces nodales.

        Après la résolution, la méthode calcule la flèche maximale, le moment
        fléchissant, les contraintes de flexion, la contrainte thermique et la
        contrainte équivalente de Von Mises.

        :param F: Force ponctuelle appliquée à l'extrémité libre [N].
        :type F: float
        :param T: Température appliquée [°C].
        :type T: float
        :return: Dictionnaire contenant les résultats du solveur EF.
        :rtype: dict
        """
        K = self._assembler()
        f = self._vecteur_forces(F, T)
        K_mod, f_mod = self._appliquer_cl(K, f)

        # Résolution du système linéaire
        try:
            U = np.linalg.solve(K_mod, f_mod)
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(K_mod, f_mod, rcond=None)[0]

        # Extraire les déplacements nodaux
        v_noeuds = U[0::2]   # v aux noeuds pairs
        theta    = U[1::2]   # θ aux noeuds impairs
        x_noeuds = np.linspace(0, self.L, self.nb_noeuds)

        fleche_max = abs(v_noeuds[-1])   # En bout (nœud libre)

        # Contrainte de flexion maximale (section à l'encastrement)
        dT = T - self.T0
        M_encastrement = F * self.L
        c = self.h / 2.0
        sig_flex_max  = (M_encastrement * c) / self.I
        sig_th        = self.E * self.alpha * dT
        sig_x_max     = sig_flex_max + sig_th
        von_mises_max = abs(sig_x_max)

        # Profil de contrainte de flexion le long de la poutre
        M_x = F * (self.L - x_noeuds)
        sigma_flex_x = (M_x * c) / self.I  # fibre supérieure

        return {
            "U":                    U,
            "fleche_max_m":         fleche_max,
            "positions_x":          x_noeuds,
            "fleche_noeuds":        v_noeuds,
            "rotations":            theta,
            "moment_flechissant":   M_x,
            "contrainte_flex_x":    sigma_flex_x,
            "contrainte_thermique_Pa": sig_th,
            "contrainte_x_max_Pa":  sig_x_max,
            "von_mises_max_Pa":     von_mises_max,
        }

    def __repr__(self):
        return (
            f"SolveurEF(L={self.L}m, nb_el={self.nb_el}, "
            f"nb_ddl={self.nb_ddl})"
        )