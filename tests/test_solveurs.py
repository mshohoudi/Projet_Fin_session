import unittest
from source.solveur_analytique import SolveurAnalytique
from source.solveur_ef import SolveurFEA1D


class TestSolveurs(unittest.TestCase):
    def setUp(self):
        # Paramètres standards pour les tests
        self.L = 1.0
        self.b = 0.1
        self.h = 0.1
        self.E = 200e9
        self.alpha = 1.2e-5

        self.force_kn = 15.0
        self.temperature_ambiante = 20.0
        self.temperature_appliquee = 100.0


        # Initialisation des solveurs
        self.solveur_ana = SolveurAnalytique(self.L, self.b, self.h, self.E, self.alpha)
        self.solveur_fea = SolveurFEA1D(self.L, self.b, self.h, self.E, self.alpha, nb_elements=20)

    def test_solveur_analytique(self):
        # Vérification du déplacement
        deplacement = self.solveur_ana.calculer_deplacement_max(self.force_kn, self.temperature_ambiante, self.temperature_appliquee)
        self.assertIsNotNone(deplacement)
        self.assertTrue(deplacement > 0.0, "Déplacement doit être positif")

        # Vérification de la contrainte
        contrainte = self.solveur_ana.calculer_contrainte_max(self.force_kn)
        self.assertIsNotNone(contrainte)
        self.assertTrue(contrainte > 0.0, "Contrainte doit être positive")

    def test_deplacement_max(self):
        # Vérification du déplacement max en valeur numérique
        deplacement_max =  self.solveur_ana.calculer_deplacement_max(self.force_kn, self.temperature_ambiante, self.temperature_appliquee)
        self.assertAlmostEqual(deplacement_max, 0.00314985,5,"Les deux valeurs ne sont pas égales à une précision de 5 décimales" )

    def test_contrainte_max(self):
        # Vérification du déplacement max en valeur numérique
        deplacement_max =  self.solveur_ana.calculer_contrainte_max(self.force_kn)
        self.assertAlmostEqual(deplacement_max, 90000000.0,5,"Les deux valeurs ne sont pas égales à une précision de 5 décimales" )



    def test_solveur_fea(self):
        # Vérification des résultats FEA
        deplacement, contrainte = self.solveur_fea.calculer_resultats(self.force_kn, self.delta_t)

        self.assertIsNotNone(deplacement)
        self.assertTrue(deplacement > 0.0, "Déplacement FEA doit être positif")

        self.assertIsNotNone(contrainte)
        self.assertTrue(contrainte > 0.0, "Contrainte FEA doit être positive")


if __name__ == '__main__':
    unittest.main()