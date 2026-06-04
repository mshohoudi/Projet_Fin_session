import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class ModeleSubstitution:
    def __init__(self, n_estimators=100, k_fold_splits=5, random_state=42):
        """
        Initialisation du modèle de Machine Learning avec paramètres dynamiques.
        """
        self.modele = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.k_fold_splits = k_fold_splits
        self.random_state = random_state
        self.est_entraine = False

    def entrainer(self, x_entrees, y_sorties):
        """
        Entraîne le modèle en utilisant la Validation Croisée (K-Fold).
        """
        print(f" -> Application de la Validation Croisée (K-Fold avec {self.k_fold_splits} plis)...")

        kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=self.random_state)

        scores_r2_vm = []
        scores_r2_x = []
        scores_r2_dep = []

        for train_index, test_index in kf.split(x_entrees):
            X_train, X_test = x_entrees[train_index], x_entrees[test_index]
            y_train, y_test = y_sorties[train_index], y_sorties[test_index]

            self.modele.fit(X_train, y_train)
            y_pred = self.modele.predict(X_test)

            scores_r2_vm.append(r2_score(y_test[:, 0], y_pred[:, 0]))
            scores_r2_x.append(r2_score(y_test[:, 1], y_pred[:, 1]))
            scores_r2_dep.append(r2_score(y_test[:, 2], y_pred[:, 2]))

        print(" -> Résultats de la Validation Croisée (Moyenne) :")
        print(f"    * Précision Von-Mises   : {np.mean(scores_r2_vm):.4f} (Stabilité: ±{np.std(scores_r2_vm):.4f})")
        print(f"    * Précision Normale X   : {np.mean(scores_r2_x):.4f} (Stabilité: ±{np.std(scores_r2_x):.4f})")
        print(f"    * Précision Déplacement : {np.mean(scores_r2_dep):.4f} (Stabilité: ±{np.std(scores_r2_dep):.4f})")

        print(" -> Entraînement final du modèle sur 100% des données...")
        self.modele.fit(x_entrees, y_sorties)
        self.est_entraine = True

    def predire(self, x_entree):
        if not self.est_entraine:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions.")
        return self.modele.predict(x_entree)