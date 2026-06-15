"""
test_ml.py
==========
Tests unitaires et d'intégration pour le Modèle de Substitution ML.

Lancement :
    pytest tests/test_ml.py -v
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# ── Chemins ──────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.abspath(os.path.join(_HERE, ".."))
_SOURCE = os.path.join(_ROOT, "source")

for p in [_SOURCE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
from gestion_donnees     import charger_ou_generer_donnees, extraire_parametres_nom_fichier
from modele_substitution import ModeleSubstitution, FEATURES, CIBLES


# ════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════

def charger_cfg():
    with open(os.path.join(_ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def creer_df_minimal(n=200) -> pd.DataFrame:
    """DataFrame synthétique minimal pour tests rapides."""
    rng = np.random.default_rng(0)
    L=1.0; b=0.1; h=0.1; E=200e9; alpha=12e-6; T0=20.0
    I = (b * h**3) / 12.0
    X = rng.uniform(0.01, L, n)
    Y = rng.uniform(-h/2, h/2, n)
    Z = rng.uniform(0, b, n)
    F = rng.uniform(10000, 40000, n)
    T = rng.uniform(20, 200, n)
    sigma_x = (F * (L-X) * Y) / I + E * alpha * (T - T0)
    vm      = np.abs(sigma_x) * rng.uniform(1.0, 1.05, n)
    vy      = -(F * X**2 * (3*L-X)) / (6*E*I)
    return pd.DataFrame({
        "X": X, "Y": Y, "Z": Z,
        "von_mises": vm, "contrainte_x": sigma_x,
        "deplacement_y": vy, "force": F, "temperature": T,
    })


def _cfg_rapide():
    """Config avec n_estimators réduit pour accélérer les tests."""
    cfg = charger_cfg()
    cfg["machine_learning"]["n_estimators"] = 50
    cfg["machine_learning"]["k_fold_splits"] = 3
    return cfg


# ════════════════════════════════════════════════════════════════
#  1. Tests — Extraction du nom de fichier
# ════════════════════════════════════════════════════════════════

class TestExtractionNomFichier(unittest.TestCase):

    def test_format_standard(self):
        f, t = extraire_parametres_nom_fichier("contrainte_F10_T80.txt")
        self.assertAlmostEqual(f, 10000.0)
        self.assertAlmostEqual(t, 80.0)

    def test_format_grand_nombre(self):
        f, t = extraire_parametres_nom_fichier("contrainte_x_F40_T200.txt")
        self.assertAlmostEqual(f, 40000.0)
        self.assertAlmostEqual(t, 200.0)

    def test_format_invalide(self):
        f, t = extraire_parametres_nom_fichier("fichier_inconnu.txt")
        self.assertIsNone(f)
        self.assertIsNone(t)

    def test_tous_types_fichiers(self):
        for prefix in ["contrainte", "contrainte_x", "deplacement_y"]:
            f, t = extraire_parametres_nom_fichier(f"{prefix}_F20_T140.txt")
            self.assertAlmostEqual(f, 20000.0)
            self.assertAlmostEqual(t, 140.0)


# ════════════════════════════════════════════════════════════════
#  2. Tests — Chargement des données
# ════════════════════════════════════════════════════════════════

class TestChargementDonnees(unittest.TestCase):

    def setUp(self):
        self.cfg = charger_cfg()

    def test_chargement_retourne_dataframe(self):
        df = charger_ou_generer_donnees(self.cfg)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_colonnes_requises_presentes(self):
        df = charger_ou_generer_donnees(self.cfg)
        for col in ["X", "Y", "Z", "von_mises", "contrainte_x",
                    "deplacement_y", "force", "temperature"]:
            self.assertIn(col, df.columns, f"Colonne manquante : {col}")

    def test_pas_de_nan_dans_features(self):
        df = charger_ou_generer_donnees(self.cfg)
        for col in FEATURES:
            self.assertEqual(df[col].isna().sum(), 0, f"NaN dans {col}")

    def test_coordonnees_dans_plage_geometrique(self):
        df  = charger_ou_generer_donnees(self.cfg)
        geo = self.cfg["geometrie"]
        self.assertTrue((df["X"] >= 0).all() and (df["X"] <= geo["longueur"]).all())
        self.assertTrue((df["Y"] >= -geo["hauteur"]/2).all())
        self.assertTrue((df["Z"] >= 0).all() and (df["Z"] <= geo["base"]).all())

    def test_cas_FT_uniques_detectes(self):
        df = charger_ou_generer_donnees(self.cfg)
        cas = df[["force", "temperature"]].drop_duplicates()
        self.assertGreaterEqual(len(cas), 4, f"Seulement {len(cas)} cas détectés")

    def test_forces_positives(self):
        df = charger_ou_generer_donnees(self.cfg)
        self.assertTrue((df["force"] > 0).all())

    def test_von_mises_positif(self):
        df = charger_ou_generer_donnees(self.cfg)
        self.assertTrue((df["von_mises"] >= 0).all())


# ════════════════════════════════════════════════════════════════
#  3. Tests — Validation K-Fold
# ════════════════════════════════════════════════════════════════

class TestValidationKFold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg    = _cfg_rapide()
        cls.df     = creer_df_minimal(n=300)
        cls.modele = ModeleSubstitution(cls.cfg)
        cls.modele.valider(cls.df)

    def test_scores_calcules_pour_toutes_cibles(self):
        for cible in CIBLES:
            self.assertIn(cible, self.modele.scores_cv)

    def test_r2_von_mises_acceptable(self):
        r2 = self.modele.scores_cv["von_mises"]["mean"]
        self.assertGreater(r2, 0.85, f"R² Von Mises : {r2:.4f}")

    def test_r2_deplacement_acceptable(self):
        r2 = self.modele.scores_cv["deplacement_y"]["mean"]
        self.assertGreater(r2, 0.90, f"R² Déplacement : {r2:.4f}")

    def test_r2_contrainte_x_acceptable(self):
        r2 = self.modele.scores_cv["contrainte_x"]["mean"]
        self.assertGreater(r2, 0.80, f"R² Contrainte X : {r2:.4f}")

    def test_ecart_type_stable(self):
        for cible in CIBLES:
            std = self.modele.scores_cv[cible]["std"]
            self.assertLess(std, 0.15, f"Instabilité pour {cible} : {std:.4f}")

    def test_nb_scores_egal_k(self):
        k = self.cfg["machine_learning"]["k_fold_splits"]
        for cible in CIBLES:
            self.assertEqual(len(self.modele.scores_cv[cible]["scores"]), k)


# ════════════════════════════════════════════════════════════════
#  4. Tests — Entraînement
# ════════════════════════════════════════════════════════════════

class TestEntrainement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg    = _cfg_rapide()
        cls.modele = ModeleSubstitution(cls.cfg)
        cls.modele.entrainer(creer_df_minimal(300))

    def test_modeles_tous_entraines(self):
        for cible in CIBLES:
            self.assertIn(cible, self.modele.modeles)

    def test_flag_entraine_actif(self):
        self.assertTrue(self.modele._entraine)

    def test_predire_sans_entrainement_leve_erreur(self):
        modele_vierge = ModeleSubstitution(self.cfg)
        grille = pd.DataFrame({"X":[0.5], "Y":[0.0], "Z":[0.05]})
        with self.assertRaises(RuntimeError):
            modele_vierge.predire(grille, 20000, 100)


# ════════════════════════════════════════════════════════════════
#  5. Tests — Prédiction
# ════════════════════════════════════════════════════════════════

class TestPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg    = _cfg_rapide()
        cls.modele = ModeleSubstitution(cls.cfg)
        cls.modele.entrainer(creer_df_minimal(400))
        L=1.0; h=0.1; b=0.1
        X = np.arange(0.0, 1.025, 0.025)
        Y = np.array([-h/2,-h/4,0,h/4,h/2])
        Z = np.array([0, b/4, b/2, 3*b/4, b])
        Xg,Yg,Zg = np.meshgrid(X,Y,Z,indexing='ij')
        cls.grille = pd.DataFrame({
            "X":Xg.flatten(), "Y":Yg.flatten(), "Z":Zg.flatten()
        })

    def _predire(self, F=20000, T=100):
        return self.modele.predire(self.grille.copy(), F, T)

    def test_shape_sortie_correcte(self):
        df_pred = self._predire()
        self.assertEqual(len(df_pred), len(self.grille))
        for col in CIBLES:
            self.assertIn(col, df_pred.columns)

    def test_pas_de_nan_dans_predictions(self):
        df_pred = self._predire()
        for col in CIBLES:
            self.assertFalse(df_pred[col].isna().any())

    def test_von_mises_toujours_positif(self):
        vals = self.modele.extraire_valeurs_max(self._predire())
        self.assertGreater(vals["von_mises"], 0)

    def test_deplacement_negatif_en_bout(self):
        df_pred = self._predire()
        bout = df_pred[df_pred["X"] > 0.95]
        if len(bout):
            self.assertTrue((bout["deplacement_y"] < 0).any())

    def test_force_plus_grande_donne_fleche_plus_grande(self):
        v1 = self.modele.extraire_valeurs_max(self._predire(F=10000, T=20))
        v2 = self.modele.extraire_valeurs_max(self._predire(F=40000, T=20))
        self.assertGreater(v2["deplacement_y"], v1["deplacement_y"])

    def test_force_plus_grande_donne_vm_plus_grande(self):
        v1 = self.modele.extraire_valeurs_max(self._predire(F=10000, T=20))
        v2 = self.modele.extraire_valeurs_max(self._predire(F=40000, T=20))
        self.assertGreater(v2["von_mises"], v1["von_mises"])

    def test_temperature_plus_haute_donne_vm_plus_grande(self):
        v1 = self.modele.extraire_valeurs_max(self._predire(F=20000, T=20))
        v2 = self.modele.extraire_valeurs_max(self._predire(F=20000, T=200))
        self.assertGreater(v2["von_mises"], v1["von_mises"])

    def test_fleche_independante_de_temperature(self):
        v1 = self.modele.extraire_valeurs_max(self._predire(F=20000, T=20))
        v2 = self.modele.extraire_valeurs_max(self._predire(F=20000, T=200))
        ratio = abs(v2["deplacement_y"] - v1["deplacement_y"]) \
                / (v1["deplacement_y"] + 1e-12)
        self.assertLess(ratio, 0.10, f"Flèche varie avec T : {ratio:.1%}")


# ════════════════════════════════════════════════════════════════
#  6. Tests — Filtre de Saint-Venant
# ════════════════════════════════════════════════════════════════

class TestSaintVenant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg    = _cfg_rapide()
        cls.cfg["machine_learning"]["n_estimators"] = 30
        cls.modele = ModeleSubstitution(cls.cfg)
        cls.modele.entrainer(creer_df_minimal(200))
        cls.df_test = pd.DataFrame({
            "X":  [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99],
            "Y":  [0.0]*7, "Z": [0.05]*7,
            "von_mises":    [1e8]*7,
            "contrainte_x": [1e8]*7,
            "deplacement_y":[-1e-3]*7,
            "force": [20000]*7, "temperature": [100]*7,
        })

    def test_noeuds_encastrement_exclus(self):
        seuil = self.cfg["saint_venant"]["seuil_x"]
        df_filtre = self.df_test[self.df_test["X"] >= seuil]
        self.assertGreaterEqual(len(df_filtre), 1)

    def test_filtre_ne_vide_pas_le_dataframe(self):
        df_centre = self.df_test[self.df_test["X"].between(0.10, 0.90)].copy()
        vals = self.modele.extraire_valeurs_max(df_centre)
        for cible in CIBLES:
            self.assertFalse(np.isnan(vals.get(cible, float("nan"))))

    def test_seuils_dans_config(self):
        sv = self.cfg["saint_venant"]
        self.assertIn("seuil_x", sv)
        self.assertIn("seuil_bout", sv)
        self.assertGreater(sv["seuil_bout"], sv["seuil_x"])


# ════════════════════════════════════════════════════════════════
#  7. Tests — Importance des features
# ════════════════════════════════════════════════════════════════

class TestImportanceFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = _cfg_rapide()
        cls.modele = ModeleSubstitution(cfg)
        cls.modele.entrainer(creer_df_minimal(300))
        cls.importances = cls.modele.importance_features()

    def test_importances_pour_toutes_cibles(self):
        for cible in CIBLES:
            self.assertIn(cible, self.importances)

    def test_somme_importances_egale_1(self):
        for cible, imp in self.importances.items():
            self.assertAlmostEqual(sum(imp.values()), 1.0, places=5)

    def test_X_feature_importante_pour_contrainte(self):
        imp_vm = self.importances.get("von_mises", {})
        if imp_vm:
            top = max(imp_vm, key=imp_vm.get)
            self.assertIn(top, ["X", "force", "Y", "temperature"])

    def test_force_importante_pour_deplacement(self):
        imp_dy = self.importances.get("deplacement_y", {})
        if imp_dy:
            top2 = sorted(imp_dy, key=imp_dy.get, reverse=True)[:2]
            self.assertTrue("force" in top2 or "X" in top2)


# ════════════════════════════════════════════════════════════════
#  8. Test d'intégration — Pipeline complet ML
# ════════════════════════════════════════════════════════════════

class TestIntegrationML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = charger_cfg()
        cls.cfg["machine_learning"]["n_estimators"] = 100
        cls.cfg["machine_learning"]["k_fold_splits"] = 3

    def test_pipeline_complet_avec_donnees_reelles(self):
        df = charger_ou_generer_donnees(self.cfg)
        self.assertGreater(len(df), 0)
        modele = ModeleSubstitution(self.cfg)
        modele.valider(df)
        modele.entrainer(df)
        for cible in CIBLES:
            if cible in modele.scores_cv:
                r2 = modele.scores_cv[cible]["mean"]
                self.assertGreater(r2, 0.80, f"R² insuffisant : {cible} = {r2:.4f}")
        L=1.0; h=0.1; b=0.1
        X = np.arange(0.0, 1.025, 0.025)
        Y = np.array([-h/2, 0, h/2])
        Z = np.array([0, b/2, b])
        Xg,Yg,Zg = np.meshgrid(X,Y,Z,indexing='ij')
        grille  = pd.DataFrame({"X":Xg.flatten(),"Y":Yg.flatten(),"Z":Zg.flatten()})
        df_pred = modele.predire(grille, force=25000, temp=100)
        self.assertEqual(len(df_pred), len(grille))
        self.assertFalse(df_pred[CIBLES].isna().any().any())
        vals = modele.extraire_valeurs_max(df_pred)
        self.assertGreater(vals["von_mises"], 0)
        self.assertGreater(vals["deplacement_y"], 0)

    def test_monotonie_force(self):
        df = charger_ou_generer_donnees(self.cfg)
        modele = ModeleSubstitution(self.cfg)
        modele.entrainer(df)
        grille = pd.DataFrame({
            "X": np.linspace(0.1, 0.9, 50),
            "Y": np.full(50, 0.05),
            "Z": np.full(50, 0.05),
        })
        vm_vals = []
        for F in [10000, 20000, 30000, 40000]:
            pred = modele.predire(grille.copy(), F, 100)
            vm_vals.append(modele.extraire_valeurs_max(pred)["von_mises"])
        for i in range(len(vm_vals)-1):
            self.assertLess(vm_vals[i], vm_vals[i+1],
                f"VM non monotone : F={[10,20,30,40][i]}→{[10,20,30,40][i+1]}kN")


# ════════════════════════════════════════════════════════════════
#  Runner standalone
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    print("\n" + "═"*62)
    print("  TEST ML — Modèle de Substitution Random Forest")
    print("═"*62 + "\n")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    t0 = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - t0
    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print("\n" + "═"*62)
    print(f"  Résultat : {passed}/{total} tests réussis  ({elapsed:.1f}s)")
    print("  ✓  OK" if result.wasSuccessful() else f"  ✗  {len(result.failures)} échecs")
    print("═"*62 + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)