"""
modele_substitution.py
======================
Modèle de substitution par Random Forest (régression multi-cible).
Prédit, pour un nœud 3D (X, Y, Z) et des conditions (F, T) :
  - Contrainte Von Mises   [Pa]
  - Contrainte normale X   [Pa]
  - Déplacement Y          [m]

Inclut :
  - Validation croisée K-Fold avec rapport de scores R²
  - Entraînement final sur 100 % des données
  - Filtrage par principe de Saint-Venant avant extraction du max
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


# Colonnes utilisées comme features d'entrée
FEATURES  = ["X", "Y", "Z", "force", "temperature"]
# Colonnes cibles
CIBLES    = ["von_mises", "contrainte_x", "deplacement_y"]
CIBLES_LABELS = {
    "von_mises":    "Von Mises [Pa]",
    "contrainte_x": "Contrainte X [Pa]",
    "deplacement_y": "Déplacement Y [m]",
}


class ModeleSubstitution:
    """
    Modèle Random Forest pour la prédiction thermoélastique 3D.
    """

    def __init__(self, cfg: dict):
        ml = cfg["machine_learning"]
        sv = cfg["saint_venant"]

        self.n_estimators   = ml["n_estimators"]
        self.max_depth      = ml.get("max_depth", None)
        self.random_state   = ml["random_state"]
        self.k_fold_splits  = ml["k_fold_splits"]
        self.seuil_sv_min   = sv["seuil_x"]       # Saint-Venant côté encastrement
        self.seuil_sv_max   = sv["seuil_bout"]     # Saint-Venant côté bout libre
        self.L              = cfg["geometrie"]["longueur"]

        self.modeles: dict[str, Pipeline] = {}
        self.scores_cv: dict[str, dict]   = {}
        self._entraine = False

    # ---------------------------------------------------------------- #
    #  Construction du pipeline (scaler + forêt)
    # ---------------------------------------------------------------- #

    def _creer_pipeline(self) -> Pipeline:
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     rf),
        ])

    # ---------------------------------------------------------------- #
    #  Validation croisée K-Fold
    # ---------------------------------------------------------------- #

    def valider(self, df: pd.DataFrame) -> None:
        """
        Effectue une validation croisée K-Fold pour chaque cible.
        Affiche et stocke les scores R².
        """
        X_data = df[FEATURES].values
        kf = KFold(n_splits=self.k_fold_splits, shuffle=True,
                   random_state=self.random_state)

        print(f"\n{'='*55}")
        print(f"  VALIDATION CROISÉE K-FOLD  (k = {self.k_fold_splits})")
        print(f"{'='*55}")

        for cible in CIBLES:
            if cible not in df.columns or df[cible].isna().all():
                print(f"  {CIBLES_LABELS[cible]:<25} : données manquantes")
                continue

            y_data = df[cible].fillna(0).values
            pipeline = self._creer_pipeline()

            scores = cross_val_score(
                pipeline, X_data, y_data,
                cv=kf, scoring="r2", n_jobs=-1
            )

            self.scores_cv[cible] = {
                "mean": scores.mean(),
                "std":  scores.std(),
                "scores": scores.tolist(),
            }

            barre = "█" * int(scores.mean() * 20)
            print(f"  {CIBLES_LABELS[cible]:<25} R² = {scores.mean():.4f} "
                  f"± {scores.std():.4f}  [{barre:<20}]")

        print(f"{'='*55}\n")

    # ---------------------------------------------------------------- #
    #  Entraînement final sur 100 % des données
    # ---------------------------------------------------------------- #

    def entrainer(self, df: pd.DataFrame) -> None:
        """
        Entraîne un modèle final pour chaque cible sur tout le dataset.
        """
        X_data = df[FEATURES].values

        print("  Entraînement final sur 100 % des données...")
        for cible in CIBLES:
            if cible not in df.columns or df[cible].isna().all():
                continue
            y_data = df[cible].fillna(0).values
            pipeline = self._creer_pipeline()
            pipeline.fit(X_data, y_data)
            self.modeles[cible] = pipeline

        self._entraine = True
        print("  Entraînement terminé.\n")

    # ---------------------------------------------------------------- #
    #  Prédiction sur une grille de noeuds
    # ---------------------------------------------------------------- #

    def predire(self, df_noeuds: pd.DataFrame,
                force: float, temp: float) -> pd.DataFrame:
        """
        Prédit les 3 cibles pour un ensemble de nœuds (X, Y, Z)
        avec les conditions de chargement (F, T).
        """
        if not self._entraine:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")

        df_pred = df_noeuds[["X", "Y", "Z"]].copy()
        df_pred["force"]       = force
        df_pred["temperature"] = temp

        X_input = df_pred[FEATURES].values

        for cible, pipeline in self.modeles.items():
            df_pred[cible] = pipeline.predict(X_input)

        return df_pred

    # ---------------------------------------------------------------- #
    #  Extraction des valeurs représentatives (avec Saint-Venant)
    # ---------------------------------------------------------------- #

    def extraire_valeurs_max(self, df_pred: pd.DataFrame) -> dict:
        """
        Applique le filtre de Saint-Venant et retourne les valeurs
        caractéristiques (max von Mises, max |contrainte_x|, max |déplacement_y|).
        """
        # Filtre Saint-Venant : exclure les zones singulières
        masque = (
            (df_pred["X"] >= self.seuil_sv_min) &
            (df_pred["X"] <= self.seuil_sv_max * self.L)
        )
        df_filtre = df_pred[masque]

        if df_filtre.empty:
            df_filtre = df_pred  # Repli sans filtre si trop restrictif

        resultats = {}
        # Modifier la logique pour éviter le code dupliqué
        for cible in CIBLES:
            if cible in df_filtre.columns:
                resultats[cible] = df_filtre[cible].abs().max()
            else:
                resultats[cible] = float("nan")

        return resultats

    # ---------------------------------------------------------------- #
    #  Rapport d'importance des features
    # ---------------------------------------------------------------- #

    def importance_features(self) -> dict:
        """Retourne l'importance des features pour chaque cible."""
        importances = {}
        for cible, pipeline in self.modeles.items():
            rf = pipeline.named_steps["rf"]
            imp = dict(zip(FEATURES, rf.feature_importances_))
            importances[cible] = imp
        return importances

    def __repr__(self):
        statut = "entraîné" if self._entraine else "non entraîné"
        return (
            f"ModeleSubstitution(n_trees={self.n_estimators}, "
            f"k_fold={self.k_fold_splits}, statut={statut})"
        )