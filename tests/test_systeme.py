"""
test_systeme.py
===============
Tests unitaires et d'intégration pour les modules système :
  - solveur_analytique.py  (Euler-Bernoulli)
  - solveur_ef.py          (FEA 1D)
  - Cohérence Analytique vs FEA
  - Principe de Saint-Venant
  - Config YAML

Lancement :
    pytest tests/test_systeme.py -v
"""

import sys
import os
import unittest
import numpy as np

# ── Chemins : fonctionne que le fichier soit dans tests/ ou à la racine ──
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, ".."))   # racine du projet
_SOURCE  = os.path.join(_ROOT, "source")

for p in [_SOURCE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
from solveur_analytique import SolveurAnalytique
from solveur_ef         import SolveurEF


# ════════════════════════════════════════════════════════════════
#  Configuration partagée
# ════════════════════════════════════════════════════════════════

def charger_cfg():
    chemin = os.path.join(_ROOT, "config.yaml")
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)

F_TEST = 20000.0
T_TEST = 100.0
T0     = 20.0


# ════════════════════════════════════════════════════════════════
#  1. Tests — Configuration YAML
# ════════════════════════════════════════════════════════════════

class TestConfiguration(unittest.TestCase):

    def setUp(self):
        self.cfg = charger_cfg()

    def test_fichier_yaml_lisible(self):
        self.assertIsInstance(self.cfg, dict)

    def test_sections_obligatoires_presentes(self):
        for section in ["geometrie", "materiau", "elements_finis",
                        "machine_learning", "saint_venant", "chemins"]:
            self.assertIn(section, self.cfg, f"Section manquante : {section}")

    def test_geometrie_positive(self):
        geo = self.cfg["geometrie"]
        self.assertGreater(geo["longueur"], 0)
        self.assertGreater(geo["base"],     0)
        self.assertGreater(geo["hauteur"],  0)

    def test_module_young_est_float(self):
        E = self.cfg["materiau"]["module_young"]
        self.assertIsInstance(E, float,
            f"module_young doit être float, pas {type(E).__name__}")
        self.assertGreater(E, 1e9)

    def test_alpha_est_float(self):
        a = self.cfg["materiau"]["coeff_thermique"]
        self.assertIsInstance(a, float,
            f"coeff_thermique doit être float, pas {type(a).__name__}")
        self.assertGreater(a, 0)

    def test_saint_venant_coherent(self):
        sv = self.cfg["saint_venant"]
        self.assertGreater(sv["seuil_bout"], sv["seuil_x"])
        L = self.cfg["geometrie"]["longueur"]
        self.assertLess(sv["seuil_x"],    L)
        self.assertLess(sv["seuil_bout"], L * 1.01)

    def test_nb_elements_positif(self):
        n = self.cfg["elements_finis"]["nb_elements"]
        self.assertGreater(n, 0)
        self.assertIsInstance(n, int)

    def test_dimensions_poutre_carree(self):
        geo = self.cfg["geometrie"]
        self.assertAlmostEqual(geo["base"],     0.1, places=5)
        self.assertAlmostEqual(geo["hauteur"],  0.1, places=5)
        self.assertAlmostEqual(geo["longueur"], 1.0, places=5)


# ════════════════════════════════════════════════════════════════
#  2. Tests — Solveur Analytique
# ════════════════════════════════════════════════════════════════

class TestSolveurAnalytique(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg     = charger_cfg()
        cls.solveur = SolveurAnalytique(cls.cfg)
        cls.res     = cls.solveur.resoudre(F_TEST, T_TEST)

    def test_moment_quadratique(self):
        b = self.cfg["geometrie"]["base"]
        h = self.cfg["geometrie"]["hauteur"]
        self.assertAlmostEqual(self.solveur.I, (b * h**3) / 12.0, places=12)

    def test_aire_section(self):
        b = self.cfg["geometrie"]["base"]
        h = self.cfg["geometrie"]["hauteur"]
        self.assertAlmostEqual(self.solveur.A, b * h, places=10)

    def test_fleche_positive(self):
        self.assertGreater(self.res["fleche_max_m"], 0)

    def test_fleche_formule(self):
        L = self.solveur.L; E = self.solveur.E; I = self.solveur.I
        v_attendu = F_TEST * L**3 / (3 * E * I)
        self.assertAlmostEqual(self.res["fleche_max_m"], v_attendu, places=10)

    def test_fleche_proportionnelle_a_F(self):
        res2 = self.solveur.resoudre(F_TEST * 2, T_TEST)
        ratio = res2["fleche_max_m"] / self.res["fleche_max_m"]
        self.assertAlmostEqual(ratio, 2.0, places=6)

    def test_fleche_independante_de_T(self):
        res_froid = self.solveur.resoudre(F_TEST, 20.0)
        res_chaud = self.solveur.resoudre(F_TEST, 300.0)
        self.assertAlmostEqual(
            res_froid["fleche_max_m"], res_chaud["fleche_max_m"], places=10)

    def test_fleche_nulle_sans_force(self):
        res = self.solveur.resoudre(0.0, T_TEST)
        self.assertAlmostEqual(res["fleche_max_m"], 0.0, places=10)

    def test_profil_fleche_monotone(self):
        x = np.linspace(0, self.solveur.L, 20)
        v = self.solveur.fleche_en_x(F_TEST, x)
        self.assertTrue(np.all(np.diff(v) >= 0))

    def test_contrainte_flex_positive(self):
        self.assertGreater(self.res["contrainte_flex_max_Pa"], 0)

    def test_contrainte_flex_formule(self):
        sigma_attendu = F_TEST * self.solveur.L * self.solveur.c / self.solveur.I
        self.assertAlmostEqual(self.res["contrainte_flex_max_Pa"], sigma_attendu, places=3)

    def test_contrainte_flex_proportionnelle_a_F(self):
        res2 = self.solveur.resoudre(F_TEST * 3, T_TEST)
        ratio = res2["contrainte_flex_max_Pa"] / self.res["contrainte_flex_max_Pa"]
        self.assertAlmostEqual(ratio, 3.0, places=6)

    def test_contrainte_thermique_nulle_a_T_ref(self):
        res = self.solveur.resoudre(F_TEST, T0)
        self.assertAlmostEqual(res["contrainte_thermique_Pa"], 0.0, places=3)

    def test_contrainte_thermique_proportionnelle_a_deltaT(self):
        res1 = self.solveur.resoudre(0, T0 + 50.0)
        res2 = self.solveur.resoudre(0, T0 + 100.0)
        ratio = res2["contrainte_thermique_Pa"] / res1["contrainte_thermique_Pa"]
        self.assertAlmostEqual(ratio, 2.0, places=6)

    def test_contrainte_thermique_formule(self):
        sigma_th_attendu = self.solveur.E * self.solveur.alpha * (T_TEST - T0)
        self.assertAlmostEqual(
            self.res["contrainte_thermique_Pa"], sigma_th_attendu, places=0)

    def test_von_mises_positif(self):
        self.assertGreater(self.res["von_mises_max_Pa"], 0)

    def test_von_mises_egal_abs_sigma_x(self):
        sigma_x = abs(self.res["contrainte_x_max_Pa"])
        vm      = self.res["von_mises_max_Pa"]
        self.assertAlmostEqual(vm, sigma_x, places=0)

    def test_toutes_cles_presentes(self):
        for cle in ["fleche_max_m", "contrainte_flex_max_Pa",
                    "contrainte_thermique_Pa", "contrainte_x_max_Pa",
                    "von_mises_max_Pa"]:
            self.assertIn(cle, self.res)

    def test_valeurs_physiquement_raisonnables(self):
        self.assertLess(self.res["fleche_max_m"],     0.05)
        self.assertLess(self.res["von_mises_max_Pa"], 2000e6)
        self.assertGreater(self.res["fleche_max_m"],  1e-6)


# ════════════════════════════════════════════════════════════════
#  3. Tests — Solveur EF 1D
# ════════════════════════════════════════════════════════════════

class TestSolveurEF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg     = charger_cfg()
        cls.solveur = SolveurEF(cls.cfg)
        cls.res     = cls.solveur.resoudre(F_TEST, T_TEST)

    def test_nb_noeuds(self):
        nb_el = self.cfg["elements_finis"]["nb_elements"]
        self.assertEqual(self.solveur.nb_noeuds, nb_el + 1)

    def test_nb_ddl(self):
        self.assertEqual(self.solveur.nb_ddl, 2 * self.solveur.nb_noeuds)

    def test_vecteur_U_bonne_taille(self):
        self.assertEqual(len(self.res["U"]), self.solveur.nb_ddl)

    def test_deplacement_nul_a_encastrement(self):
        self.assertAlmostEqual(self.res["fleche_noeuds"][0], 0.0, places=8)

    def test_rotation_nulle_a_encastrement(self):
        self.assertAlmostEqual(self.res["rotations"][0], 0.0, places=8)

    def test_fleche_positive(self):
        self.assertGreater(self.res["fleche_max_m"], 0)

    def test_profil_fleche_monotone(self):
        v = np.abs(self.res["fleche_noeuds"])
        self.assertTrue(np.all(np.diff(v) >= -1e-12))

    def test_positions_x_correctes(self):
        x = self.res["positions_x"]
        self.assertAlmostEqual(x[0],  0.0, places=10)
        self.assertAlmostEqual(x[-1], self.solveur.L, places=10)
        self.assertEqual(len(x), self.solveur.nb_noeuds)

    def test_moment_nul_en_bout_libre(self):
        self.assertAlmostEqual(self.res["moment_flechissant"][-1], 0.0, places=3)

    def test_moment_max_a_encastrement(self):
        M = self.res["moment_flechissant"]
        self.assertAlmostEqual(abs(M[0]), abs(M).max(), places=3)

    def test_contrainte_thermique(self):
        E  = float(self.cfg["materiau"]["module_young"])
        a  = float(self.cfg["materiau"]["coeff_thermique"])
        sigma_th_attendu = E * a * (T_TEST - T0)
        self.assertAlmostEqual(
            self.res["contrainte_thermique_Pa"], sigma_th_attendu, places=0)

    def test_von_mises_positif(self):
        self.assertGreater(self.res["von_mises_max_Pa"], 0)

    def test_toutes_cles_presentes(self):
        for cle in ["U", "fleche_max_m", "positions_x", "fleche_noeuds",
                    "rotations", "moment_flechissant", "contrainte_flex_x",
                    "contrainte_thermique_Pa", "contrainte_x_max_Pa",
                    "von_mises_max_Pa"]:
            self.assertIn(cle, self.res)


# ════════════════════════════════════════════════════════════════
#  4. Tests — Cohérence Analytique vs FEA
# ════════════════════════════════════════════════════════════════

class TestCoherenceAnalytiqueVsEF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = charger_cfg()
        cls.ana = SolveurAnalytique(cls.cfg)
        cls.ef  = SolveurEF(cls.cfg)

    def _ecart_pct(self, val_ef, val_ana):
        if abs(val_ana) < 1e-15:
            return 0.0
        return abs(val_ef - val_ana) / abs(val_ana) * 100.0

    def test_fleche_max_accord(self):
        r_ana = self.ana.resoudre(F_TEST, T_TEST)
        r_ef  = self.ef.resoudre(F_TEST, T_TEST)
        ecart = self._ecart_pct(r_ef["fleche_max_m"], r_ana["fleche_max_m"])
        self.assertLess(ecart, 1.0, f"Écart flèche : {ecart:.3f}%")

    def test_contrainte_x_accord(self):
        r_ana = self.ana.resoudre(F_TEST, T_TEST)
        r_ef  = self.ef.resoudre(F_TEST, T_TEST)
        ecart = self._ecart_pct(
            r_ef["contrainte_x_max_Pa"], r_ana["contrainte_x_max_Pa"])
        self.assertLess(ecart, 1.0, f"Écart σ_x : {ecart:.3f}%")

    def test_von_mises_accord(self):
        r_ana = self.ana.resoudre(F_TEST, T_TEST)
        r_ef  = self.ef.resoudre(F_TEST, T_TEST)
        ecart = self._ecart_pct(
            r_ef["von_mises_max_Pa"], r_ana["von_mises_max_Pa"])
        self.assertLess(ecart, 1.0, f"Écart VM : {ecart:.3f}%")

    def test_linearite_force_coherente(self):
        for F in [5000, 15000, 30000, 40000]:
            r_ana = self.ana.resoudre(F, T_TEST)
            r_ef  = self.ef.resoudre(F, T_TEST)
            ecart = self._ecart_pct(r_ef["fleche_max_m"], r_ana["fleche_max_m"])
            self.assertLess(ecart, 1.0, f"Désaccord à F={F/1000:.0f}kN : {ecart:.2f}%")

    def test_contrainte_thermique_identique(self):
        r_ana = self.ana.resoudre(0.0, T_TEST)
        r_ef  = self.ef.resoudre(0.0, T_TEST)
        self.assertAlmostEqual(
            r_ef["contrainte_thermique_Pa"],
            r_ana["contrainte_thermique_Pa"],
            delta=1.0)

    def test_plusieurs_temperatures(self):
        for T in [20, 80, 140, 200]:
            r_ana = self.ana.resoudre(F_TEST, T)
            r_ef  = self.ef.resoudre(F_TEST, T)
            ecart = self._ecart_pct(
                r_ef["von_mises_max_Pa"], r_ana["von_mises_max_Pa"])
            self.assertLess(ecart, 1.0, f"Désaccord VM à T={T}°C : {ecart:.2f}%")


# ════════════════════════════════════════════════════════════════
#  5. Tests — Physique générale
# ════════════════════════════════════════════════════════════════

class TestPhysiqueGenerale(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = charger_cfg()
        cls.ana = SolveurAnalytique(cls.cfg)
        cls.ef  = SolveurEF(cls.cfg)

    def test_superposition_charge_et_thermique(self):
        res = self.ana.resoudre(F_TEST, T_TEST)
        sigma_total = res["contrainte_flex_max_Pa"] + res["contrainte_thermique_Pa"]
        self.assertAlmostEqual(res["contrainte_x_max_Pa"], sigma_total, places=1)

    def test_cas_limite_force_nulle(self):
        res_ana = self.ana.resoudre(0.0, T_TEST)
        res_ef  = self.ef.resoudre(0.0, T_TEST)
        self.assertAlmostEqual(res_ana["fleche_max_m"], 0.0, places=10)
        self.assertAlmostEqual(res_ef["fleche_max_m"],  0.0, places=6)
        self.assertGreater(abs(res_ana["contrainte_thermique_Pa"]), 0)

    def test_cas_limite_temperature_ref(self):
        res = self.ana.resoudre(F_TEST, T0)
        self.assertAlmostEqual(res["contrainte_thermique_Pa"], 0.0, places=3)

    def test_valeurs_acier_realistes(self):
        res = self.ana.resoudre(20000, 20.0)
        self.assertAlmostEqual(res["fleche_max_m"] * 1e3, 4.0, delta=0.1)
        self.assertAlmostEqual(res["contrainte_flex_max_Pa"] / 1e6, 120.0, delta=1.0)

    def test_rigidite_matrice_symetrique(self):
        K = self.ef._assembler()
        diff = np.max(np.abs(K - K.T))
        self.assertLess(diff, 1e-10)

    def test_rigidite_matrice_definie_positive(self):
        K = self.ef._assembler()
        f = np.zeros(self.ef.nb_ddl)
        K_mod, _ = self.ef._appliquer_cl(K, f)
        indices_libres = list(range(2, self.ef.nb_ddl))
        K_libre = K_mod[np.ix_(indices_libres, indices_libres)]
        vp = np.linalg.eigvalsh(K_libre)
        self.assertGreater(vp.min(), 0)


# ════════════════════════════════════════════════════════════════
#  Runner standalone
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    print("\n" + "═"*62)
    print("  TEST SYSTÈME — Analytique, FEA 1D, Config, Physique")
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