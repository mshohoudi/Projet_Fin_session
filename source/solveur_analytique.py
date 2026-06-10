"""
solveur_analytique.py
=====================
Solveur purement théorique basé sur la Résistance des Matériaux :
  - Flèche maximale (Euler-Bernoulli, poutre encastrée-libre)
  - Contrainte de flexion maximale (fibre extrême à l'encastrement)
  - Contrainte thermique axiale (dilatation empêchée)
  - Contrainte Von Mises approximée
"""

import numpy as np


class SolveurAnalytique:
    """
    Résolution analytique thermoélastique d'une poutre encastrée-libre
    soumise à une charge ponctuelle en bout et un gradient de température.
    """

    def __init__(self, cfg: dict):
        geo = cfg["geometrie"]
        mat = cfg["materiau"]

        # Conversion explicite en float pour éviter les erreurs YAML
        self.L     = float(geo["longueur"])       # [m]
        self.b     = float(geo["base"])           # [m]
        self.h     = float(geo["hauteur"])        # [m]
        self.E     = float(mat["module_young"])    # [Pa]
        self.nu    = float(mat["poisson"])         # [-]
        self.alpha = float(mat["coeff_thermique"]) # [1/°C]
        self.T0    = float(mat["temperature_ref"]) # [°C]

        # Propriétés dérivées
        self.I  = (self.b * self.h**3) / 12.0   # Moment quadratique [m⁴]
        self.c  = self.h / 2.0                   # Distance fibre extrême [m]
        self.A  = self.b * self.h                # Aire de section [m²]

    # ---------------------------------------------------------------- #
    #  Flèche
    # ---------------------------------------------------------------- #

    def fleche_max(self, F: float) -> float:
        """
        Flèche maximale à l'extrémité libre (x = L).
        v_max = F * L³ / (3 * E * I)   [m]
        """
        return (F * self.L**3) / (3.0 * self.E * self.I)

    def fleche_en_x(self, F: float, x: np.ndarray) -> np.ndarray:
        """
        Profil de flèche le long de la poutre.
        """
        return (F * x**2 * (3.0 * self.L - x)) / (6.0 * self.E * self.I)

    # ---------------------------------------------------------------- #
    #  Contraintes
    # ---------------------------------------------------------------- #

    def contrainte_flexion_max(self, F: float) -> float:
        """
        Contrainte de flexion maximale à l'encastrement (x = 0), fibre extrême.
        """
        M_max = F * self.L
        return (M_max * self.c) / self.I

    def contrainte_thermique(self, T: float) -> float:
        """
        Contrainte thermique axiale (telle que testée et validée avec Ansys).
        """
        dT = T - self.T0
        return self.E * self.alpha * dT

    def contrainte_x_en_section(self, F: float, T: float, x: float,
                                  y: np.ndarray) -> np.ndarray:
        """
        Contrainte normale σ_x en une section x, pour un tableau d'ordonnées y.
        """
        M_x = F * (self.L - x)
        sigma_flex = (M_x * y) / self.I
        sigma_th   = self.contrainte_thermique(T)
        return sigma_flex + sigma_th

    def von_mises_max(self, F: float, T: float) -> float:
        """
        Contrainte Von Mises approximée (poutre mince, σ_y ≈ 0, τ ≈ 0).
        """
        sigma_x = self.contrainte_flexion_max(F) + self.contrainte_thermique(T)
        return abs(sigma_x)

    # ---------------------------------------------------------------- #
    #  Récapitulatif
    # ---------------------------------------------------------------- #

    def resoudre(self, F: float, T: float) -> dict:
        """
        Retourne un dictionnaire de tous les résultats analytiques clés.
        """
        v_max       = self.fleche_max(F)
        sig_flex    = self.contrainte_flexion_max(F)
        sig_th      = self.contrainte_thermique(T)
        sig_x_max   = sig_flex + sig_th
        sig_vm      = abs(sig_x_max)

        return {
            "fleche_max_m":            v_max,
            "contrainte_flex_max_Pa":  sig_flex,
            "contrainte_thermique_Pa": sig_th,
            "contrainte_x_max_Pa":     sig_x_max,
            "von_mises_max_Pa":        sig_vm,
        }

    def __repr__(self):
        return (
            f"SolveurAnalytique("
            f"L={self.L}m, b={self.b}m, h={self.h}m, "
            f"E={self.E/1e9:.0f}GPa)"
        )