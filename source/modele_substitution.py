import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class ModeleSubstitution:
    def __init__(self):
        """
        Initialisation du modèle de Machine Learning (Random Forest).
        """
        self.modele = RandomForestRegressor(n_estimators=100, random_state=42)
        self.est_entraine = False

    def entrainer(self, x_entrees, y_sorties):
        """
        Entraîne le modèle en utilisant la Validation Croisée (K-Fold)
        pour garantir la stabilité, puis l'entraîne sur 100% des données.
        """
        print(" -> Application de la Validation Croisée (K-Fold avec 5 plis)...")

        # Séparation en 5 sous-ensembles (Chaque sous-ensemble servira de test une fois)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scores_r2_vm = []
        scores_r2_x = []
        scores_r2_dep = []

        # Boucle d'entraînement et de test pour chaque pli (Fold)
        for train_index, test_index in kf.split(x_entrees):
            # Séparation 80% Entraînement / 20% Test pour cette itération
            X_train, X_test = x_entrees[train_index], x_entrees[test_index]
            y_train, y_test = y_sorties[train_index], y_sorties[test_index]

            # Entraînement sur le 80% actuel
            self.modele.fit(X_train, y_train)
            y_pred = self.modele.predict(X_test)

            # Enregistrement des scores de précision sur le 20% actuel
            scores_r2_vm.append(r2_score(y_test[:, 0], y_pred[:, 0]))
            scores_r2_x.append(r2_score(y_test[:, 1], y_pred[:, 1]))
            scores_r2_dep.append(r2_score(y_test[:, 2], y_pred[:, 2]))

        # Affichage de la moyenne et de l'écart-type (stabilité) des 5 itérations
        print(" -> Résultats de la Validation Croisée (Moyenne des 5 entraînements) :")
        print(f"    * Précision Von-Mises   : {np.mean(scores_r2_vm):.4f} (Stabilité: ±{np.std(scores_r2_vm):.4f})")
        print(f"    * Précision Normale X   : {np.mean(scores_r2_x):.4f} (Stabilité: ±{np.std(scores_r2_x):.4f})")
        print(f"    * Précision Déplacement : {np.mean(scores_r2_dep):.4f} (Stabilité: ±{np.std(scores_r2_dep):.4f})")

        print(" -> Entraînement final du modèle sur 100% des données pour le mode interactif...")
        # Une fois la fiabilité prouvée, on entraîne sur toutes les données pour un maximum de précision en production
        self.modele.fit(x_entrees, y_sorties)
        self.est_entraine = True

    def predire(self, x_entree):
        """
        Effectue une prédiction si le modèle est entraîné.
        """
        if not self.est_entraine:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions.")
        return self.modele.predict(x_entree)