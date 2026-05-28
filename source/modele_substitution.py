import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class ModeleSubstitution:
    def __init__(self):
        # Utilisation de l'algorithme Random Forest (très robuste pour les données FEA non linéaires)
        # n_estimators=100 signifie que le modèle utilisera 100 "arbres de décision" pour prédire
        self.modele = RandomForestRegressor(n_estimators=100, random_state=42)
        self.est_entraine = False

    def entrainer(self, x_entrees, y_sorties):
        print(" -> Division des données en ensembles d'entraînement (80%) et de test (20%)...")
        # Séparation des données pour valider la précision du modèle sur des points qu'il n'a jamais vus
        x_train, x_test, y_train, y_test = train_test_split(
            x_entrees, y_sorties, test_size=0.2, random_state=42
        )

        print(" -> Entraînement du modèle Random Forest en cours...")
        self.modele.fit(x_train, y_train)
        self.est_entraine = True

        # Évaluation des performances sur les 20% de données de test
        y_pred = self.modele.predict(x_test)

        # Le R2 Score varie de 0 à 1 (1 étant une prédiction parfaite)
        r2_contrainte = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_deplacement = r2_score(y_test[:, 1], y_pred[:, 1])

        print(f" -> Précision (R2 Score) - Contrainte   : {r2_contrainte:.4f}")
        print(f" -> Précision (R2 Score) - Déplacement  : {r2_deplacement:.4f}")

    def predire(self, x_nouvelle_entree):
        # Cette fonction sera utilisée plus tard pour faire des prédictions instantanées
        if not self.est_entraine:
            raise ValueError("Erreur : Le modèle doit être entraîné avant de faire des prédictions.")

        # S'assurer que l'entrée est sous le bon format mathématique (matrice 2D)
        x_nouvelle_entree = np.array(x_nouvelle_entree)
        if x_nouvelle_entree.ndim == 1:
            x_nouvelle_entree = x_nouvelle_entree.reshape(1, -1)

        return self.modele.predict(x_nouvelle_entree)