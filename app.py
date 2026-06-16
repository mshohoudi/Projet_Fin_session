"""
app.py — Dashboard Streamlit
==============================
Interface graphique pour le Système Thermoélastique Hybride.
Remplace l'interface terminal de main.py.

Lancement :
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# ── Chemins ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
# Assurez-vous que le dossier contenant les solveurs s'appelle bien "solveurs"
_SOURCE = os.path.join(_ROOT, "solveurs")
sys.path.insert(0, _SOURCE)
sys.path.insert(0, _ROOT)

from gestion_donnees import charger_ou_generer_donnees
from modele_substitution import ModeleSubstitution
from solveur_analytique import SolveurAnalytique
from solveur_ef import SolveurEF

# ════════════════════════════════════════════════════════════════
#  Config page
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Système Thermoélastique Hybride",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé (Thème Clair / Light Mode)
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-label { color: #495057; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #0056b3; font-size: 24px; font-weight: bold; }
    .metric-unit  { color: #6c757d; font-size: 12px; }
    .ecart-ok   { color: #28a745; font-weight: bold; }
    .ecart-warn { color: #fd7e14; font-weight: bold; }
    .section-title {
        font-size: 18px; font-weight: 600;
        color: #6f42c1; margin: 12px 0 8px 0;
    }
    div[data-testid="stSidebar"] { background: #f1f3f5; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  Cache : chargement config + données + modèle
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Chargement de la configuration...")
def charger_config():
    with open(os.path.join(_ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Ingestion des données Ansys...")
def charger_donnees(cfg):
    return charger_ou_generer_donnees(cfg)


@st.cache_resource(show_spinner="Entraînement du modèle Random Forest...")
def entrainer_modele(_df, cfg):
    modele = ModeleSubstitution(cfg)
    modele.valider(_df)
    modele.entrainer(_df)
    return modele


def generer_grille(cfg) -> pd.DataFrame:
    L = cfg["geometrie"]["longueur"]
    h = cfg["geometrie"]["hauteur"]
    b = cfg["geometrie"]["base"]
    X = np.arange(0.0, L + 0.001, 0.025)
    Y = np.array([-h / 2, -h / 4, 0, h / 4, h / 2])
    Z = np.array([0, b / 4, b / 2, 3 * b / 4, b])
    Xg, Yg, Zg = np.meshgrid(X, Y, Z, indexing="ij")
    return pd.DataFrame({"X": Xg.flatten(), "Y": Yg.flatten(), "Z": Zg.flatten()})


# ════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════

def sidebar(cfg):
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/2a/Ets_quebec_logo.png",
                     use_container_width=True)

    st.sidebar.markdown("## 🔩 Surrogate ML — Poutre")
    st.sidebar.markdown("**Projet Fin de Session MGA 802**")
    st.sidebar.markdown("**MOHAMMAD SHOHOUDIMOJDEHI, NICOLAS ALLARD**")
    st.sidebar.divider()

    # ── Géométrie (lecture seule) ──
    geo = cfg["geometrie"]
    st.sidebar.markdown("### 📐 Géométrie")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Longueur", f"{geo['longueur']} m")
    c2.metric("Section", f"{geo['base'] * 100:.0f}×{geo['hauteur'] * 100:.0f} cm")

    mat = cfg["materiau"]
    st.sidebar.markdown("### ⚙️ Matériau")
    c3, c4 = st.sidebar.columns(2)
    c3.metric("E", f"{mat['module_young'] / 1e9:.0f} GPa")
    c4.metric("α", f"{mat['coeff_thermique'] * 1e6:.1f}×10⁻⁶")
    st.sidebar.divider()

    # ── Saisie des conditions ──
    st.sidebar.markdown("### 🎛️ Conditions aux limites")

    # Curseur pour la force (Élargi pour tester l'extrapolation : 0 à 100 kN)
    force_kN = st.sidebar.slider(
        "Force appliquée F [kN] (Entraînement: 10-40)",
        min_value=0.0, max_value=100.0,
        value=25.0, step=1.0,
        help="Force ponctuelle à l'extrémité libre (X=L)"
    )

    # Curseur pour la température (Élargi pour tester l'extrapolation : -20 à 400 °C)
    temp_C = st.sidebar.slider(
        "Température T [°C] (Entraînement: 20-200)",
        min_value=-20.0, max_value=400.0,
        value=100.0, step=5.0,
        help="Température uniforme de la poutre"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 Configuration ML")
    sv = cfg["saint_venant"]
    st.sidebar.caption(f"Validité x : [{sv['seuil_x']}, {sv['seuil_bout']}] m")
    st.sidebar.caption(
        f"RF : {cfg['machine_learning']['n_estimators']} arbres — "
        f"K-Fold={cfg['machine_learning']['k_fold_splits']}"
    )

    return force_kN * 1000.0, temp_C


# ════════════════════════════════════════════════════════════════
#  Graphiques Plotly
# ════════════════════════════════════════════════════════════════

def fig_comparaison_barres(res_ana, res_ef, res_ml, F, T):
    """Barres comparatives : flèche max et Von Mises max."""
    methodes = ["Analytique", "FEA 1D", "ML 3D"]
    couleurs = ["#007bff", "#28a745", "#fd7e14"]

    v_vals = [
        res_ana["fleche_max_m"] * 1e3,
        res_ef["fleche_max_m"] * 1e3,
        abs(res_ml.get("deplacement_y", 0)) * 1e3,
    ]
    vm_vals = [
        res_ana["von_mises_max_Pa"] / 1e6,
        res_ef["von_mises_max_Pa"] / 1e6,
        abs(res_ml.get("von_mises", 0)) / 1e6,
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Flèche Maximale [mm] (Déplacement Y)", "Contrainte Von Mises Max [MPa]"),
        horizontal_spacing=0.12,
    )

    for i, (m, c) in enumerate(zip(methodes, couleurs)):
        fig.add_trace(go.Bar(
            name=m, x=[m], y=[v_vals[i]],
            marker_color=c, showlegend=False,
            text=f"{v_vals[i]:.4f}", textposition="outside",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=m, x=[m], y=[vm_vals[i]],
            marker_color=c, showlegend=False,
            text=f"{vm_vals[i]:.2f}", textposition="outside",
        ), row=1, col=2)

    fig.update_layout(
        title=f"<b>Comparaison des résultats</b> — F={F / 1e3:.1f} kN | T={T:.0f}°C",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"),
        barmode="group", height=380,
        margin=dict(t=60, b=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e9ecef")
    return fig


def fig_profil_fleche(res_ef, res_ana, res_ml, F, T, L):
    """Profil de flèche v(x) le long de la poutre."""
    x = np.linspace(0, L, 200)
    E = res_ana.get("_E");
    I = res_ana.get("_I")

    fig = go.Figure()

    if E and I:
        v_ana = (F * x ** 2 * (3 * L - x)) / (6 * E * I)
        fig.add_trace(go.Scatter(
            x=x, y=v_ana * 1e3, name="Analytique",
            line=dict(color="#007bff", width=2.5),
        ))

    x_ef = res_ef.get("positions_x", [])
    v_ef = np.abs(res_ef.get("fleche_noeuds", []))
    if len(x_ef):
        fig.add_trace(go.Scatter(
            x=x_ef, y=v_ef * 1e3, name="FEA 1D",
            line=dict(color="#28a745", width=2, dash="dash"),
        ))

    v_ml = abs(res_ml.get("deplacement_y", 0))
    fig.add_trace(go.Scatter(
        x=[L], y=[v_ml * 1e3], name=f"ML 3D (bout)",
        mode="markers",
        marker=dict(color="#fd7e14", size=12, symbol="triangle-up"),
    ))

    fig.update_layout(
        title="<b>Profil de Flèche v(x) (Déplacement Y)</b>",
        xaxis_title="Position x [m]",
        yaxis_title="Flèche |v(x)| (Déplacement Y) [mm]",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"), height=350,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=70),
    )
    fig.update_xaxes(gridcolor="#e9ecef")
    fig.update_yaxes(gridcolor="#e9ecef")
    return fig


def fig_profil_contrainte(res_ef, res_ml, F, T, L):
    """Profil de contrainte σ_x le long de la poutre (fibre sup.)."""
    x_ef = res_ef.get("positions_x", [])
    sig_ef_x = res_ef.get("contrainte_flex_x", [])
    sig_th = res_ef.get("contrainte_thermique_Pa", 0)

    fig = go.Figure()

    if len(x_ef) and len(sig_ef_x):
        fig.add_trace(go.Scatter(
            x=x_ef, y=(sig_ef_x + sig_th) / 1e6,
            name="FEA 1D (fibre sup.)",
            line=dict(color="#28a745", width=2.5),
            fill="tozeroy", fillcolor="rgba(40,167,69,0.15)",
        ))

    sx_ml = abs(res_ml.get("contrainte_x", 0))
    fig.add_hline(
        y=sx_ml / 1e6, line_dash="dot", line_color="#fd7e14",
        annotation_text=f"ML max = {sx_ml / 1e6:.1f} MPa",
        annotation_font_color="#fd7e14",
    )

    fig.update_layout(
        title="<b>Profil de Contrainte normale σ_x</b> (fibre supérieure)",
        xaxis_title="Position x [m]",
        yaxis_title="σ_x [MPa]",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"), height=350,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=70),
    )
    fig.update_xaxes(gridcolor="#e9ecef")
    fig.update_yaxes(gridcolor="#e9ecef", zeroline=True, zerolinecolor="#adb5bd")
    return fig


def fig_heatmap_3d(df_pred, cible, F, T, label, unite):
    """Nuage de points 3D coloré (cartographie ML de la poutre)."""
    if cible not in df_pred.columns:
        return None

    vals = df_pred[cible].values
    fig = go.Figure(go.Scatter3d(
        x=df_pred["X"], y=df_pred["Y"], z=df_pred["Z"],
        mode="markers",
        marker=dict(
            size=3,
            color=vals,
            colorscale="Jet",
            colorbar=dict(
                title=dict(text=f"{label}<br>[{unite}]", font=dict(color="#212529")),
                tickfont=dict(color="#212529"),
            ),
            opacity=0.8,
        ),
    ))
    fig.update_layout(
        title=f"<b>Cartographie 3D — {label}</b><br>F={F / 1e3:.1f} kN | T={T:.0f}°C",
        scene=dict(
            xaxis=dict(title="X [m]", backgroundcolor="#f8f9fa", gridcolor="#e9ecef"),
            yaxis=dict(title="Y [m]", backgroundcolor="#f8f9fa", gridcolor="#e9ecef"),
            zaxis=dict(title="Z [m]", backgroundcolor="#f8f9fa", gridcolor="#e9ecef"),
            aspectmode='manual',
            aspectratio=dict(x=3, y=1, z=1),
            bgcolor="#ffffff",
        ),
        paper_bgcolor="#ffffff",
        font=dict(color="#212529"),
        height=500,
        margin=dict(t=60, b=10),
    )
    return fig


def fig_importance_features(importances):
    """Importance des variables par cible ML."""
    labels_map = {
        "von_mises": "Von Mises",
        "contrainte_x": "Contrainte X",
        "deplacement_y": "Déplacement Y",
    }
    couleurs = {"von_mises": "#007bff", "contrainte_x": "#28a745", "deplacement_y": "#fd7e14"}

    fig = make_subplots(
        rows=1, cols=len(importances),
        subplot_titles=[labels_map.get(k, k) for k in importances],
    )
    for col, (cible, imp) in enumerate(importances.items(), 1):
        features = list(imp.keys())
        vals = list(imp.values())
        idx = np.argsort(vals)
        fig.add_trace(go.Bar(
            x=[vals[i] for i in idx],
            y=[features[i] for i in idx],
            orientation="h",
            marker_color=couleurs.get(cible, "#6f42c1"),
            showlegend=False,
        ), row=1, col=col)

    fig.update_layout(
        title="<b>Importance des Variables — Random Forest</b>",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"), height=300,
        margin=dict(t=60, b=20),
    )
    fig.update_xaxes(gridcolor="#e9ecef")
    fig.update_yaxes(gridcolor="#e9ecef")
    return fig


def fig_scores_kfold(scores_cv):
    """Graphique R² moyen par cible (Validation Croisée K-Fold)."""
    labels_map = {
        "von_mises": "Von Mises",
        "contrainte_x": "Contrainte X",
        "deplacement_y": "Déplacement Y",
    }
    cibles = [c for c in scores_cv]
    moyennes = [scores_cv[c]["mean"] for c in cibles]
    erreurs = [scores_cv[c]["std"] for c in cibles]
    noms = [labels_map.get(c, c) for c in cibles]

    fig = go.Figure(go.Bar(
        x=noms, y=moyennes,
        error_y=dict(type="data", array=erreurs, visible=True, color="#dc3545"),
        marker_color=["#007bff", "#28a745", "#fd7e14"],
        text=[f"{m:.4f}" for m in moyennes],
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#adb5bd")
    fig.update_layout(
        title="<b>Précision du Modèle (R²) — K-Fold CV</b>",
        yaxis=dict(range=[0.8, 1.05], title="Score R² moyen"),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"), height=300,
        margin=dict(t=50, b=20),
    )
    fig.update_xaxes(gridcolor="#e9ecef")
    fig.update_yaxes(gridcolor="#e9ecef")
    return fig


def fig_extrapolation_force(modele, cfg, T_fixe):
    """
    Génère un graphique montrant le comportement du modèle ML vs le modèle Analytique
    en dehors de son domaine d'entraînement (Extrapolation) pour la Force.
    """
    forces_kN = np.linspace(0, 100, 21)
    forces_N = forces_kN * 1000

    v_ana_list = []
    v_ml_list = []

    grille = generer_grille(cfg)
    solveur = SolveurAnalytique(cfg)

    for F in forces_N:
        # Résultat Analytique
        res_ana = solveur.resoudre(F, T_fixe)
        v_ana_list.append(res_ana["fleche_max_m"] * 1e3)

        # Résultat ML
        df_pred = modele.predire(grille, F, T_fixe)
        res_ml = modele.extraire_valeurs_max(df_pred)
        v_ml_list.append(abs(res_ml.get("deplacement_y", 0)) * 1e3)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forces_kN, y=v_ana_list, mode='lines+markers',
        name='Analytique (Théorie)', line=dict(color='#007bff', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=forces_kN, y=v_ml_list, mode='lines+markers',
        name='ML 3D (Random Forest)', line=dict(color='#fd7e14', width=3)
    ))

    fig.add_vrect(
        x0=10, x1=40, fillcolor="rgba(40,167,69,0.2)", layer="below", line_width=0,
        annotation_text="Domaine d'entraînement (10-40 kN)", annotation_position="top left"
    )

    fig.update_layout(
        title=f"<b>Analyse d'Extrapolation</b> (Flèche Max vs Force) à T={T_fixe}°C",
        xaxis_title="Force appliquée F [kN]",
        yaxis_title="Flèche Maximale [mm] (Déplacement Y)",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"), height=400,
        margin=dict(t=50, b=50)
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  Métriques résumé
# ════════════════════════════════════════════════════════════════

def afficher_metriques(res_ana, res_ef, res_ml):
    # Unités correctement gérées
    v_ana = res_ana["fleche_max_m"] * 1e3
    v_ef = res_ef["fleche_max_m"] * 1e3
    v_ml = abs(res_ml.get("deplacement_y", 0)) * 1e3

    vm_ana = res_ana["von_mises_max_Pa"] / 1e6
    vm_ef = res_ef["von_mises_max_Pa"] / 1e6
    vm_ml = abs(res_ml.get("von_mises", 0)) / 1e6

    ecart_v_ef = abs(v_ef - v_ana) / (v_ana + 1e-12) * 100
    ecart_v_ml = abs(v_ml - v_ana) / (v_ana + 1e-12) * 100
    ecart_vm_ef = abs(vm_ef - vm_ana) / (vm_ana + 1e-12) * 100
    ecart_vm_ml = abs(vm_ml - vm_ana) / (vm_ana + 1e-12) * 100

    st.markdown("### 📊 Résultats")

    # ── Flèche ──
    st.markdown("**Flèche Maximale (Déplacement Y)**")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔵 Analytique", f"{v_ana:.4f} mm")
    c2.metric("🟢 FEA 1D", f"{v_ef:.4f} mm",
              delta=f"{ecart_v_ef:.2f}% vs Ana",
              delta_color="inverse" if ecart_v_ef > 2 else "normal")
    c3.metric("🟠 ML 3D", f"{v_ml:.4f} mm",
              delta=f"{ecart_v_ml:.2f}% vs Ana",
              delta_color="inverse" if ecart_v_ml > 10 else "normal")

    # ── Von Mises ──
    st.markdown("**Von Mises Maximale**")
    c4, c5, c6 = st.columns(3)
    c4.metric("🔵 Analytique", f"{vm_ana:.2f} MPa")
    c5.metric("🟢 FEA 1D", f"{vm_ef:.2f} MPa",
              delta=f"{ecart_vm_ef:.2f}% vs Ana",
              delta_color="inverse" if ecart_vm_ef > 2 else "normal")
    c6.metric("🟠 ML 3D", f"{vm_ml:.2f} MPa",
              delta=f"{ecart_vm_ml:.2f}% vs Ana",
              delta_color="inverse" if ecart_vm_ml > 10 else "normal")

    # ── Contrainte X ──
    st.markdown("**Contrainte Normale σ_x Maximale**")
    c7, c8, c9 = st.columns(3)
    sx_ana = res_ana["contrainte_x_max_Pa"] / 1e6
    sx_ef = res_ef["contrainte_x_max_Pa"] / 1e6
    sx_ml = abs(res_ml.get("contrainte_x", 0)) / 1e6
    c7.metric("🔵 Analytique", f"{sx_ana:.2f} MPa")
    c8.metric("🟢 FEA 1D", f"{sx_ef:.2f} MPa")
    c9.metric("🟠 ML 3D", f"{sx_ml:.2f} MPa")


# ════════════════════════════════════════════════════════════════
#  Application principale
# ════════════════════════════════════════════════════════════════

def main():
    # ── Chargement ──────────────────────────────────────────────
    cfg = charger_config()
    df = charger_donnees(cfg)
    modele = entrainer_modele(df, cfg)

    # ── Sidebar ─────────────────────────────────────────────────
    force_N, temp_C = sidebar(cfg)

    # ── Titre principal ─────────────────────────────────────────
    st.title("🔩 Système Thermoélastique Hybride : Métamodèle 3D")
    st.caption("Comparaison : Solution Analytique, Éléments Finis 1D & Apprentissage Automatique 3D (Random Forest)")
    st.divider()

    # ── Calculs ─────────────────────────────────────────────────
    with st.spinner("Calcul en cours..."):
        res_ana = SolveurAnalytique(cfg).resoudre(force_N, temp_C)
        res_ef = SolveurEF(cfg).resoudre(force_N, temp_C)

        # Ajouter E et I pour le profil de flèche analytique
        solveur = SolveurAnalytique(cfg)
        res_ana["_E"] = solveur.E
        res_ana["_I"] = solveur.I

        grille = generer_grille(cfg)
        df_pred = modele.predire(grille, force_N, temp_C)
        res_ml = modele.extraire_valeurs_max(df_pred)

    L = cfg["geometrie"]["longueur"]

    # ── Section 1 : Métriques Classiques ────────────────────────
    afficher_metriques(res_ana, res_ef, res_ml)
    st.divider()

    # ── Section 2 : Graphiques comparatifs 1D ───────────────────
    st.markdown("### 📈 Profils le long de la poutre (X)")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            fig_profil_fleche(res_ef, res_ana, res_ml, force_N, temp_C, L),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            fig_profil_contrainte(res_ef, res_ml, force_N, temp_C, L),
            use_container_width=True
        )

    st.plotly_chart(
        fig_comparaison_barres(res_ana, res_ef, res_ml, force_N, temp_C),
        use_container_width=True
    )
    st.divider()

    # ── Section 3 : Heatmaps 3D ─────────────────────────────────
    st.markdown("### 🌡️ Cartographies 3D de la Poutre (Prédictions ML)")
    tab1, tab2, tab3 = st.tabs(["Von Mises", "Contrainte X", "Flèche (Déplacement Y)"])

    with tab1:
        fig = fig_heatmap_3d(df_pred, "von_mises", force_N, temp_C, "Von Mises", "Pa")
        if fig: st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = fig_heatmap_3d(df_pred, "contrainte_x", force_N, temp_C, "Contrainte X", "Pa")
        if fig: st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = fig_heatmap_3d(df_pred, "deplacement_y", force_N, temp_C, "Déplacement Y", "m")
        if fig: st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section 4 : Modèle ML ───────────────────────────────────
    st.markdown("### 🤖 Performance du Modèle ML")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_scores_kfold(modele.scores_cv), use_container_width=True)
    with col4:
        st.plotly_chart(fig_importance_features(modele.importance_features()), use_container_width=True)

    st.divider()

    # ── Section 5 : Analyse d'Extrapolation (Force) ─────────────
    st.markdown("### ⚠️ Comportement hors domaine (Extrapolation)")
    st.info(
        "Ce graphique illustre la principale limitation des modèles basés sur les arbres de décision (Random Forest). "
        "Il démontre visuellement l'incapacité du modèle à extrapoler la tendance physique au-delà de sa zone d'entraînement [10-40 kN].")

    st.plotly_chart(
        fig_extrapolation_force(modele, cfg, temp_C),
        use_container_width=True
    )

    st.divider()

    # ── Section 6 : Données brutes ──────────────────────────────
    with st.expander("🗃️ Aperçu des données d'entraînement (Ansys nodes sample)"):
        num_FT = df[['force', 'temperature']].drop_duplicates().shape[0]
        st.caption(
            f"Total : {len(df):,} nœuds d'éléments finis — Issus de {num_FT} simulations Ansys (combinaisons F×T)")
        st.dataframe(
            df.sample(min(200, len(df)), random_state=42).round(6).reset_index(drop=True),
            use_container_width=True, height=250,
        )

    # ── Footer ──────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; color:#6c757d; font-size:12px; margin-top:20px;'>
        Projet Fin de Session MGA 802 · Système Thermoélastique Hybride · ÉTS Montréal
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()