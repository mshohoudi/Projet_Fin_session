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
        Calcule la flèche maximale à l'extrémité libre.

        La solution est obtenue à partir de la théorie des poutres
        d'Euler-Bernoulli.

        .. math::

            v_{max}=\\frac{FL^3}{3EI}

        :param F: Force ponctuelle appliquée à l'extrémité libre [N].
        :type F: float
        :return: Flèche maximale de la poutre [m].
        :rtype: float
        """
        return (F * self.L**3) / (3.0 * self.E * self.I)

    def fleche_en_x(self, F: float, x: np.ndarray) -> np.ndarray:
        """
        Calcule le profil de flèche le long de la poutre.

        .. math::

            v(x)=\\frac{F x^2 (3L-x)}{6EI}

        :param F: Force ponctuelle appliquée à l'extrémité libre [N].
        :type F: float
        :param x: Positions le long de la poutre [m].
        :type x: numpy.ndarray
        :return: Flèche calculée pour chaque position.
        :rtype: numpy.ndarray
        """
        return (F * x**2 * (3.0 * self.L - x)) / (6.0 * self.E * self.I)

    # ---------------------------------------------------------------- #
    #  Contraintes
    # ---------------------------------------------------------------- #

    def contrainte_flexion_max(self, F: float) -> float:
        """
        Calcule la contrainte maximale de flexion.

        La contrainte est évaluée à l'encastrement à l'aide de la formule
        de Navier.

        .. math::

            \\sigma = \\frac{Mc}{I}

        :param F: Force ponctuelle appliquée [N].
        :type F: float
        :return: Contrainte maximale de flexion [Pa].
        :rtype: float
        """
        M_max = F * self.L
        return (M_max * self.c) / self.I

    def contrainte_thermique(self, T: float) -> float:
        """
        Calcule la contrainte thermique uniforme.

        La contrainte thermique est calculée en supposant que la dilatation
        longitudinale est complètement empêchée.

        .. math::

            \\sigma_{th}=E\\alpha\\Delta T

        :param T: Température appliquée [°C].
        :type T: float
        :return: Contrainte thermique uniforme [Pa].
        :rtype: float
        """
        dT = T - self.T0
        return self.E * self.alpha * dT

    def contrainte_x_en_section(self, F: float, T: float, x: float,
                                  y: np.ndarray) -> np.ndarray:
        """
        Calcule la contrainte normale selon l'axe X dans une section de la
        poutre.

        La contrainte totale est obtenue par superposition de la contrainte
        de flexion et de la contrainte thermique.

        .. math::

            \\sigma_x(x,y)=\\frac{M(x)y}{I}+E\\alpha\\Delta T

        :param F: Force ponctuelle appliquée [N].
        :type F: float
        :param T: Température appliquée [°C].
        :type T: float
        :param x: Position de la section le long de la poutre [m].
        :type x: float
        :param y: Coordonnées dans la section mesurées depuis l'axe neutre [m].
        :type y: numpy.ndarray
        :return: Distribution de la contrainte normale selon X [Pa].
        :rtype: numpy.ndarray
        """
        M_x = F * (self.L - x)
        sigma_flex = (M_x * y) / self.I
        sigma_th   = self.contrainte_thermique(T)
        return sigma_flex + sigma_th

    def von_mises_max(self, F: float, T: float) -> float:
        """
        Calcule la contrainte maximale de Von Mises.

        Pour cette modélisation de poutre, seules les contraintes normales
        sont considérées. La contrainte de Von Mises est donc assimilée à
        la valeur absolue de la contrainte normale maximale.

        :param F: Force ponctuelle appliquée [N].
        :type F: float
        :param T: Température appliquée [°C].
        :type T: float
        :return: Contrainte maximale de Von Mises [Pa].
        :rtype: float
        """
        sigma_x = self.contrainte_flexion_max(F) + self.contrainte_thermique(T)
        return abs(sigma_x)

    # ---------------------------------------------------------------- #
    #  Récapitulatif
    # ---------------------------------------------------------------- #

    def resoudre(self, F: float, T: float) -> dict:
        """
        Résout analytiquement le problème thermoélastique.

        Calcule la flèche maximale, les contraintes de flexion, la
        contrainte thermique, la contrainte normale maximale et la
        contrainte équivalente de Von Mises.

        :param F: Force ponctuelle appliquée [N].
        :type F: float
        :param T: Température appliquée [°C].
        :type T: float

        :return: Dictionnaire contenant les résultats analytiques.

                 - ``fleche_max_m`` : flèche maximale [m]
                 - ``contrainte_flex_max_Pa`` : contrainte maximale de flexion [Pa]
                 - ``contrainte_thermique_Pa`` : contrainte thermique [Pa]
                 - ``contrainte_x_max_Pa`` : contrainte normale maximale [Pa]
                 - ``von_mises_max_Pa`` : contrainte maximale de Von Mises [Pa]

        :rtype: dict
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