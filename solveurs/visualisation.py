"""
visualisation.py
================
Génération des cartographies 3D (heatmaps) des contraintes et déplacements
prédits par le modèle ML, ainsi que des comparaisons graphiques
entre la théorie, le FEA 1D et l'IA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------------ #
#  Utilitaires internes
# ------------------------------------------------------------------ #

def _titre_avec_conditions(titre: str, F: float, T: float) -> str:
    return f"{titre}\n(F = {F / 1e3:.1f} kN, T = {T:.0f} °C)"


# ------------------------------------------------------------------ #
#  Heatmap 3D principale
# ------------------------------------------------------------------ #

def heatmap_3d(df_pred: pd.DataFrame, cible: str,
               F: float, T: float,
               etiquette: str = None, unite: str = "Pa") -> plt.Figure:
    """
    Affiche un nuage de points 3D coloré par la valeur de `cible`.
    """
    if cible not in df_pred.columns:
        print(f"  [AVERTISSEMENT] Colonne '{cible}' absente du DataFrame.")
        return None

    valeurs = df_pred[cible].values
    norm = Normalize(vmin=valeurs.min(), vmax=valeurs.max())
    cmap = plt.cm.jet

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df_pred["X"].values,
        df_pred["Y"].values,
        df_pred["Z"].values,
        c=valeurs, cmap=cmap, norm=norm,
        s=5, alpha=0.7
    )

    label_cb = etiquette if etiquette else cible
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cb.set_label(f"{label_cb} [{unite}]", fontsize=10)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    titre_map = {
        "von_mises": "Contrainte Von Mises",
        "contrainte_x": "Contrainte Normale σ_x",
        "deplacement_y": "Déplacement Y",
    }
    titre = titre_map.get(cible, cible)
    ax.set_title(_titre_avec_conditions(titre, F, T), fontsize=11)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  Comparaison graphique triple : Analytique / EF / ML
# ------------------------------------------------------------------ #

def graphique_comparaison(
        res_analytique: dict,
        res_ef: dict,
        res_ml: dict,
        F: float, T: float,
        L: float
) -> plt.Figure:
    """
    Génère un graphique comparatif à 3 panneaux.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        _titre_avec_conditions("Comparaison Analytique / EF 1D / ML 3D", F, T),
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Panneau 1 : Flèche ----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    x_th = np.linspace(0, L, 200)

    # Flèche analytique (Correction : calcul de EI via reverse-engineering)
    v_max_th = res_analytique.get("fleche_max_m", 0)
    if v_max_th > 0:
        EI = (F * L ** 3) / (3.0 * v_max_th)
        v_th = (F * x_th ** 2 * (3.0 * L - x_th)) / (6.0 * EI)
        ax1.plot(x_th, v_th * 1e3, "b-", lw=2, label="Analytique")

    # Flèche EF
    x_ef = res_ef.get("positions_x", np.array([]))
    v_ef = res_ef.get("fleche_noeuds", np.array([]))
    if len(x_ef) and len(v_ef):
        ax1.plot(x_ef, np.abs(v_ef) * 1e3, "r--", lw=2, label="FEA 1D")

    # Point ML (valeur max en bout)
    v_ml = res_ml.get("deplacement_y", None)
    if v_ml is not None:
        ax1.plot(L, float(v_ml) * 1e3, "g^", ms=10, label=f"ML (max={v_ml * 1e3:.3f}mm)")

    ax1.set_xlabel("Position x [m]")
    ax1.set_ylabel("Flèche |v(x)| [mm]")
    ax1.set_title("Flèche transversale")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Panneau 2 : Contrainte σ_x le long de la fibre extrême ----
    ax2 = fig.add_subplot(gs[0, 1])

    # Profil EF de contrainte
    sig_ef_x = res_ef.get("contrainte_flex_x", np.array([]))
    sig_th = res_ef.get("contrainte_thermique_Pa", 0.0)
    if len(x_ef) and len(sig_ef_x):
        ax2.plot(x_ef, (sig_ef_x + sig_th) / 1e6, "r--", lw=2, label="FEA 1D (fibre +)")

    # Point ML contrainte max
    sx_ml = res_ml.get("contrainte_x", None)
    if sx_ml is not None:
        ax2.axhline(float(sx_ml) / 1e6, color="g", linestyle=":", lw=2,
                    label=f"ML max = {sx_ml / 1e6:.2f} MPa")

    ax2.set_xlabel("Position x [m]")
    ax2.set_ylabel("σ_x [MPa]")
    ax2.set_title("Contrainte Normale σ_x (fibre supérieure)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Panneau 3 : Barres de comparaison flèche -------------------
    ax3 = fig.add_subplot(gs[1, 0])
    labels_barre = ["Analytique", "FEA 1D", "ML 3D"]
    vals_fleche = [
        res_analytique.get("fleche_max_m", 0) * 1e3,
        res_ef.get("fleche_max_m", 0) * 1e3,
        (res_ml.get("deplacement_y", 0) or 0) * 1e3,
    ]
    colors = ["steelblue", "tomato", "mediumseagreen"]
    bars = ax3.bar(labels_barre, vals_fleche, color=colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, vals_fleche):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Flèche max [mm]")
    ax3.set_title("Comparaison : Flèche Maximale")
    ax3.grid(True, axis="y", alpha=0.3)

    # ---- Panneau 4 : Barres de comparaison Von Mises ----------------
    ax4 = fig.add_subplot(gs[1, 1])
    vals_vm = [
        res_analytique.get("von_mises_max_Pa", 0) / 1e6,
        res_ef.get("von_mises_max_Pa", 0) / 1e6,
        (res_ml.get("von_mises", 0) or 0) / 1e6,
    ]
    bars2 = ax4.bar(labels_barre, vals_vm, color=colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars2, vals_vm):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Von Mises max [MPa]")
    ax4.set_title("Comparaison : Contrainte Von Mises Max")
    ax4.grid(True, axis="y", alpha=0.3)

    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ------------------------------------------------------------------ #
#  Graphique importance des features
# ------------------------------------------------------------------ #

def graphique_importance_features(importances: dict) -> plt.Figure:
    """
    Affiche un graphique d'importance des variables pour chaque cible ML.
    """
    cibles = list(importances.keys())
    n = len(cibles)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["steelblue", "tomato", "mediumseagreen"]
    labels_map = {
        "von_mises": "Von Mises",
        "contrainte_x": "Contrainte X",
        "deplacement_y": "Déplacement Y",
    }

    for ax, (cible, imp), color in zip(axes, importances.items(), colors):
        features = list(imp.keys())
        vals = list(imp.values())
        idx = np.argsort(vals)[::-1]
        ax.barh([features[i] for i in idx],
                [vals[i] for i in idx],
                color=color, alpha=0.8, edgecolor="black")
        ax.set_title(labels_map.get(cible, cible), fontsize=11)
        ax.set_xlabel("Importance relative")
        ax.set_xlim(0, max(vals) * 1.2)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Importance des Variables — Modèle Random Forest",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  Affichage de toutes les figures
# ------------------------------------------------------------------ #

def afficher_toutes_les_figures(
        df_pred: pd.DataFrame,
        res_analytique: dict,
        res_ef: dict,
        res_ml: dict,
        importances: dict,
        F: float, T: float, L: float
) -> None:
    """Lance la génération et l'affichage de toutes les visualisations."""

    print("\n  Génération des visualisations...")

    unites = {
        "von_mises": ("Von Mises [Pa]", "Pa"),
        "contrainte_x": ("Contrainte X [Pa]", "Pa"),
        "deplacement_y": ("Déplacement Y [m]", "m"),
    }
    for cible, (label, unite) in unites.items():
        fig = heatmap_3d(df_pred, cible, F, T, label, unite)
        if fig:
            fig.canvas.manager.set_window_title(f"Heatmap 3D — {label}")

    fig_comp = graphique_comparaison(res_analytique, res_ef, res_ml, F, T, L)
    fig_comp.canvas.manager.set_window_title("Comparaison Tri-méthodes")

    if importances:
        fig_imp = graphique_importance_features(importances)
        if fig_imp:
            fig_imp.canvas.manager.set_window_title("Importance des Variables ML")

    print("  Affichage des figures (fermer les fenêtres pour quitter).\n")
    plt.show()