import math
class SolveurAnalytique:
    def __init__(self, longueur, base, hauteur, module_young, coeff_thermique):
        """
        Initialise les propriétés géométriques et matérielles.
        """
        self.L = longueur
        self.b = base
        self.h = hauteur
        self.E = module_young
        self.alpha = coeff_thermique
        self.I = (self.b * self.h**3) / 12
        self.c = self.h / 2


    def calculer_deplacement_max(self, force_kn, temperature_ambiante, temperature_appliquee):
        """
        Doit retourner le déplacement total maximal en mètres (m).
        Attention : Convertir force_kn en Newtons !
        """
        force_n = force_kn * 1000
        delta_t = temperature_appliquee - temperature_ambiante

        Uy = (force_n * self.L ** 3) / (3 * self.E * self.I)
        Ux = self.alpha * self.L * delta_t

        delta_max = math.sqrt(Ux ** 2 + Uy ** 2)

        return delta_max


    def calculer_contrainte_max(self, force_kn):
        """
        Retourne la contrainte maximale en Pascals (Pa).
        force_kn est convertie en Newtons.
        """
        force_n = force_kn * 1000
        moment_force_nm = force_n * self.L

        sigma_max = (moment_force_nm * self.c) / self.I

        return sigma_max