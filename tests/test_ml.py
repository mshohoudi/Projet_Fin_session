import unittest
import numpy as np
from source.modele_substitution import ModeleSubstitution


class TestModeleSubstitution(unittest.TestCase):
    def setUp(self):
        # Initialisation avec des paramètres réduits pour un test rapide
        self.modele = ModeleSubstitution(n_estimators=10, k_fold_splits=3, random_state=42)

        # Génération de données factices (Mock Data) pour 20 échantillons
        # 5 Entrées : X, Y, Z, Force, Température
        self.x_dummy = np.random.rand(20, 5)

        # 3 Sorties : Contrainte VM, Contrainte X, Déplacement Y
        self.y_dummy = np.random.rand(20, 3)

    def test_initialisation(self):
        # Le modèle ne doit pas être entraîné au départ
        self.assertFalse(self.modele.est_entraine,
                         "Le modèle ne devrait pas être marqué comme entraîné à l'initialisation.")
        self.assertEqual(self.modele.k_fold_splits, 3)

    def test_entrainement_et_prediction(self):
        # 1. Test de l'entraînement
        self.modele.entrainer(self.x_dummy, self.y_dummy)
        self.assertTrue(self.modele.est_entraine, "Le modèle doit être marqué comme entraîné après entrainer().")

        # 2. Test de la prédiction
        x_test = np.random.rand(5, 5)  # 5 nouveaux échantillons à prédire
        predictions = self.modele.predire(x_test)

        # La sortie doit avoir 5 lignes (échantillons) et 3 colonnes (VM, X, Dep)
        self.assertEqual(predictions.shape, (5, 3), "La forme de la matrice de prédiction est incorrecte.")

    def test_erreur_prediction_sans_entrainement(self):
        # Le modèle doit lever une erreur s'il prédit avant d'être entraîné
        x_test = np.random.rand(1, 5)
        with self.assertRaises(ValueError):
            self.modele.predire(x_test)


if __name__ == '__main__':
    unittest.main()