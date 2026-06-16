"""
main.py — Orchestrateur Principal
===================================
Système Thermoélastique Hybride : FEA 1D & Modèle de Substitution ML
Maîtrise en Génie Aérospatial — ÉTS

Flux d'exécution :
  1. Chargement de config.yaml
  2. Ingestion des données Ansys (ou synthétiques)
  3. Validation K-Fold + Entraînement Random Forest
  4. Saisie interactive des conditions (F, T)
  5. Prédiction ML 3D  +  Solveurs Analytique & FEA 1D
  6. Rapport comparatif terminal + fichier texte
  7. Visualisations 3D
"""

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

# ── Modules internes ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solveurs"))

from gestion_donnees     import charger_ou_generer_donnees
from modele_substitution import ModeleSubstitution
from solveur_analytique  import SolveurAnalytique
from solveur_ef          import SolveurEF
from visualisation       import afficher_toutes_les_figures


# ═══════════════════════════════════════════════════════════════════════
#  Utilitaires d'affichage
# ═══════════════════════════════════════════════════════════════════════

LIGNE  = "═" * 62
LIGNE2 = "─" * 62

def banniere():
    print(f"\n{LIGNE}")
    print("  SYSTÈME THERMOÉLASTIQUE HYBRIDE — FEA 1D & ML 3D")
    print("  MGA 802 PROJET FIN DE SESSION")
    print("  MOHAMMAD SHOHOUDIMOJDEHI, NICOLAS ALLARD")
    print(LIGNE)

def section(titre: str):
    print(f"\n{LIGNE2}")
    print(f"  {titre}")
    print(LIGNE2)

def ok(msg: str):
    print(f"  ✓  {msg}")

def info(msg: str):
    print(f"  ►  {msg}")


# ═══════════════════════════════════════════════════════════════════════
#  Saisie interactive avec validation
# ═══════════════════════════════════════════════════════════════════════

def saisir_float(invite: str, min_val: float, max_val: float,
                 defaut: float = None) -> float:
    """Demande un flottant dans [min_val, max_val] avec valeur par défaut."""
    while True:
        suffixe = f" [défaut={defaut}]" if defaut is not None else ""
        try:
            entree = input(f"  {invite}{suffixe} : ").strip()
            if entree == "" and defaut is not None:
                return defaut
            val = float(entree)
            if min_val <= val <= max_val:
                return val
            print(f"  ⚠  Valeur hors plage [{min_val}, {max_val}]. Réessayez.")
        except ValueError:
            print("  ⚠  Entrée invalide. Saisissez un nombre.")


def saisir_conditions(cfg: dict) -> tuple[float, float]:
    """
    Demande à l'utilisateur la force (kN) et la température (°C).
    Retourne (force_N, temp_C).
    """
    section("SAISIE DES CONDITIONS DE CHARGEMENT")
    print()
    print("  Plage des données d'entraînement :")
    print("    Force       : 10 kN  →  40 kN")
    print("    Température : 20 °C  →  200 °C")
    print()
    print("  ℹ  Vous pouvez saisir des valeurs hors plage (extrapolation ML).")
    print()

    force_kN = saisir_float("Force appliquée [kN]",  1.0, 200.0, defaut=25.0)
    temp_C   = saisir_float("Température     [°C]",  0.0, 500.0, defaut=100.0)

    force_N  = force_kN * 1000.0
    return force_N, temp_C


# ═══════════════════════════════════════════════════════════════════════
#  Génération de la grille de nœuds pour la prédiction ML
# ═══════════════════════════════════════════════════════════════════════

def generer_grille_prediction(cfg: dict) -> pd.DataFrame:
    """
    Crée une grille 3D de nœuds représentative de la poutre
    (même résolution que les données Ansys : 41×5×5 = 1025 nœuds).
    """
    geo = cfg["geometrie"]
    L = geo["longueur"]
    b = geo["base"]
    h = geo["hauteur"]

    X_vals = np.arange(0.0, L + 0.001, 0.025)
    Y_vals = np.array([-h/2, -h/4, 0.0, h/4, h/2])
    Z_vals = np.array([0.0, b/4, b/2, 3*b/4, b])

    Xg, Yg, Zg = np.meshgrid(X_vals, Y_vals, Z_vals, indexing="ij")
    df = pd.DataFrame({
        "X": Xg.flatten(),
        "Y": Yg.flatten(),
        "Z": Zg.flatten(),
    })
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Rapport comparatif
# ═══════════════════════════════════════════════════════════════════════

def afficher_rapport(res_ana: dict, res_ef: dict, res_ml: dict,
                     force_N: float, temp_C: float,
                     scores_cv: dict) -> str:
    """
    Affiche et retourne le rapport comparatif tri-méthodes.
    """
    force_kN = force_N / 1000.0

    lignes = []
    lignes.append("")
    lignes.append(LIGNE)
    lignes.append("  RAPPORT COMPARATIF — RÉSULTATS THERMOÉLASTIQUES")
    lignes.append(LIGNE)
    lignes.append(f"  Force appliquée  : {force_kN:.1f} kN  ({force_N:.0f} N)")
    lignes.append(f"  Température      : {temp_C:.1f} °C")
    lignes.append(LIGNE2)

    # ── Tableau flèche ──────────────────────────────────────────────
    lignes.append("")
    lignes.append("  FLÈCHE MAXIMALE (en bout, x = L)")
    lignes.append("")
    lignes.append(f"  {'Méthode':<22} {'Valeur [mm]':>14}")
    lignes.append(f"  {'-'*22} {'-'*14}")

    v_ana = res_ana["fleche_max_m"] * 1e3
    v_ef  = res_ef["fleche_max_m"]  * 1e3
    v_ml  = abs(res_ml.get("deplacement_y", 0) or 0) * 1e3

    lignes.append(f"  {'Analytique':<22} {v_ana:>14.4f}")
    lignes.append(f"  {'FEA 1D (Euler-Bernoulli)':<22} {v_ef:>14.4f}")
    lignes.append(f"  {'ML 3D (Random Forest)':<22} {v_ml:>14.4f}")

    if v_ana > 0:
        lignes.append("")
        lignes.append(f"  Écart FEA / Analytique  : {abs(v_ef-v_ana)/v_ana*100:.2f} %")
        lignes.append(f"  Écart ML  / Analytique  : {abs(v_ml-v_ana)/v_ana*100:.2f} %")

    # ── Tableau contrainte Von Mises ────────────────────────────────
    lignes.append("")
    lignes.append(LIGNE2)
    lignes.append("  CONTRAINTE VON MISES MAXIMALE")
    lignes.append("")
    lignes.append(f"  {'Méthode':<22} {'Valeur [MPa]':>14}")
    lignes.append(f"  {'-'*22} {'-'*14}")

    vm_ana = res_ana["von_mises_max_Pa"] / 1e6
    vm_ef  = res_ef["von_mises_max_Pa"]  / 1e6
    vm_ml  = abs(res_ml.get("von_mises", 0) or 0) / 1e6

    lignes.append(f"  {'Analytique':<22} {vm_ana:>14.2f}")
    lignes.append(f"  {'FEA 1D (Euler-Bernoulli)':<22} {vm_ef:>14.2f}")
    lignes.append(f"  {'ML 3D (Random Forest)':<22} {vm_ml:>14.2f}")

    if vm_ana > 0:
        lignes.append("")
        lignes.append(f"  Écart FEA / Analytique  : {abs(vm_ef-vm_ana)/vm_ana*100:.2f} %")
        lignes.append(f"  Écart ML  / Analytique  : {abs(vm_ml-vm_ana)/vm_ana*100:.2f} %")

    # ── Tableau contrainte X ────────────────────────────────────────
    lignes.append("")
    lignes.append(LIGNE2)
    lignes.append("  CONTRAINTE NORMALE σ_x MAXIMALE")
    lignes.append("")
    lignes.append(f"  {'Méthode':<22} {'Valeur [MPa]':>14}")
    lignes.append(f"  {'-'*22} {'-'*14}")

    sx_ana = res_ana["contrainte_x_max_Pa"] / 1e6
    sx_ef  = res_ef["contrainte_x_max_Pa"]  / 1e6
    sx_ml  = abs(res_ml.get("contrainte_x", 0) or 0) / 1e6

    lignes.append(f"  {'Analytique':<22} {sx_ana:>14.2f}")
    lignes.append(f"  {'FEA 1D (Euler-Bernoulli)':<22} {sx_ef:>14.2f}")
    lignes.append(f"  {'ML 3D (Random Forest)':<22} {sx_ml:>14.2f}")

    # ── Scores ML ───────────────────────────────────────────────────
    lignes.append("")
    lignes.append(LIGNE2)
    lignes.append("  PRÉCISION DU MODÈLE ML (Validation K-Fold)")
    lignes.append("")
    lignes.append(f"  {'Cible':<28} {'R² moyen':>10}  {'± écart-type':>14}")
    lignes.append(f"  {'-'*28} {'-'*10}  {'-'*14}")

    labels = {
        "von_mises":    "Von Mises",
        "contrainte_x": "Contrainte X",
        "deplacement_y": "Déplacement Y",
    }
    for cle, label in labels.items():
        if cle in scores_cv:
            s = scores_cv[cle]
            lignes.append(f"  {label:<28} {s['mean']:>10.4f}  {s['std']:>14.4f}")

    # ── Principe de Saint-Venant ────────────────────────────────────
    lignes.append("")
    lignes.append(LIGNE2)
    lignes.append("  NOTE : Principe de Saint-Venant appliqué")
    lignes.append("  Les singularités à l'encastrement (x < 0.09 m) et en bout")
    lignes.append("  (x > 0.91 m) sont exclues du maximum ML.")
    lignes.append("")
    lignes.append(LIGNE)
    lignes.append("")

    rapport = "\n".join(lignes)
    print(rapport)
    return rapport


# ═══════════════════════════════════════════════════════════════════════
#  Sauvegarde du rapport
# ═══════════════════════════════════════════════════════════════════════

def sauvegarder_rapport(rapport: str, cfg: dict,
                        force_N: float, temp_C: float) -> None:
    nom = cfg["chemins"]["rapport_txt"]
    # Ajouter le timestamp dans le nom si le fichier existe déjà
    if os.path.exists(nom):
        ts  = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(nom)
        nom = f"{base}_{ts}{ext}"

    with open(nom, "w", encoding="utf-8") as f:
        f.write(rapport)
    ok(f"Rapport sauvegardé → {nom}")


# ═══════════════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

def main():
    banniere()

    # ── 1. Configuration ─────────────────────────────────────────────
    section("CHARGEMENT DE LA CONFIGURATION")
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ok("config.yaml chargé")

    geo = cfg["geometrie"]
    info(f"Poutre : L={geo['longueur']}m  b={geo['base']*100:.0f}cm  "
         f"h={geo['hauteur']*100:.0f}cm")
    info(f"Matériau : E={cfg['materiau']['module_young']/1e9:.0f} GPa  "
         f"α={cfg['materiau']['coeff_thermique']*1e6:.1f}×10⁻⁶ /°C")

    # ── 2. Données ───────────────────────────────────────────────────
    section("INGESTION DES DONNÉES ANSYS")
    df_donnees = charger_ou_generer_donnees(cfg)
    ok(f"{len(df_donnees):,} nœuds  |  "
       f"{df_donnees[['force','temperature']].drop_duplicates().shape[0]} cas (F×T)")

    # ── 3. ML : Validation + Entraînement ───────────────────────────
    section("MODÈLE DE SUBSTITUTION — RANDOM FOREST")
    modele = ModeleSubstitution(cfg)
    modele.valider(df_donnees)
    modele.entrainer(df_donnees)

    # ── 4. Saisie interactive ────────────────────────────────────────
    continuer = True
    while continuer:
        force_N, temp_C = saisir_conditions(cfg)

        # ── 5a. Solveur Analytique ───────────────────────────────────
        section("SOLVEUR ANALYTIQUE (Euler-Bernoulli)")
        solveur_ana = SolveurAnalytique(cfg)
        res_ana = solveur_ana.resoudre(force_N, temp_C)
        res_ana["_EI"] = solveur_ana.E * solveur_ana.I   # pour la visu
        ok(f"Flèche max         : {res_ana['fleche_max_m']*1e3:.4f} mm")
        ok(f"Von Mises max      : {res_ana['von_mises_max_Pa']/1e6:.2f} MPa")
        ok(f"Contrainte therm.  : {res_ana['contrainte_thermique_Pa']/1e6:.2f} MPa")

        # ── 5b. Solveur FEA 1D ───────────────────────────────────────
        section("SOLVEUR FEA 1D (Éléments Finis Euler-Bernoulli)")
        solveur_ef = SolveurEF(cfg)
        res_ef = solveur_ef.resoudre(force_N, temp_C)
        ok(f"Flèche max         : {res_ef['fleche_max_m']*1e3:.4f} mm")
        ok(f"Von Mises max      : {res_ef['von_mises_max_Pa']/1e6:.2f} MPa")
        ok(f"Nb éléments        : {solveur_ef.nb_el}  |  Nb DDL : {solveur_ef.nb_ddl}")

        # ── 5c. Prédiction ML ────────────────────────────────────────
        section("PRÉDICTION ML 3D (Random Forest)")
        grille = generer_grille_prediction(cfg)
        df_pred = modele.predire(grille, force_N, temp_C)
        res_ml = modele.extraire_valeurs_max(df_pred)
        ok(f"Flèche max (ML)    : {abs(res_ml.get('deplacement_y',0))*1e3:.4f} mm")
        ok(f"Von Mises max (ML) : {res_ml.get('von_mises',0)/1e6:.2f} MPa")
        ok(f"Saint-Venant       : x ∈ [{cfg['saint_venant']['seuil_x']:.2f} m, "
           f"{cfg['saint_venant']['seuil_bout']:.2f} m]")

        # ── 6. Rapport comparatif ────────────────────────────────────
        section("RAPPORT COMPARATIF")
        rapport = afficher_rapport(
            res_ana, res_ef, res_ml,
            force_N, temp_C,
            modele.scores_cv
        )
        sauvegarder_rapport(rapport, cfg, force_N, temp_C)

        # ── 7. Visualisations ────────────────────────────────────────
        print()
        afficher_visu = input("  Afficher les visualisations 3D ? [O/n] : ").strip().lower()
        if afficher_visu in ("", "o", "oui", "y", "yes"):
            importances = modele.importance_features()
            afficher_toutes_les_figures(
                df_pred, res_ana, res_ef, res_ml,
                importances, force_N, temp_C,
                cfg["geometrie"]["longueur"]
            )

        # ── 8. Nouvelle analyse ? ────────────────────────────────────
        print()
        rejouer = input("  Effectuer une nouvelle analyse ? [o/N] : ").strip().lower()
        continuer = rejouer in ("o", "oui", "y", "yes")

    print()
    print(LIGNE)
    print("  Analyse terminée. Merci d'avoir utilisé le système.")
    print(LIGNE)
    print()


if __name__ == "__main__":
    main()