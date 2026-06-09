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
        F = force_kn * 1000
        delta_T = temperature_appliquee - temperature_ambiante

        Uy = (F * self.L ** 3) / (3 * self.E * self.I)
        Ux = self.alpha * self.L * delta_T

        delta_max = math.sqrt(Ux ** 2 + Uy ** 2)

        return delta_max


