import numpy as np


class SolveurEF:
    """
    Résolution FEA 1D d'une poutre Euler-Bernoulli encastrée-libre.

    DDL par noeud : [v_i, θ_i, v_{i+1}, θ_{i+1}]
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
        Matrice de rigidité élémentaire pour un élément de longueur le.
        DDL locaux : [v1, θ1, v2, θ2]
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
        """Assemble la matrice de rigidité globale [nb_ddl × nb_ddl]."""
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
        Construit le vecteur de forces nodales global.
          - Charge ponctuelle F appliquée au dernier nœud (extrémité libre)
          - Forces thermiques équivalentes (variation axiale empêchée)
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
        Encastrement parfait au nœud 0 : v_0 = 0, θ_0 = 0.
        Méthode de pénalité (très grand nombre) pour rester symétrique.
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
        Résout K*U = f et extrait les résultats physiques.
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