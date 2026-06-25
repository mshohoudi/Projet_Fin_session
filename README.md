# 🔩 Système Thermoélastique Hybride — FEA 1D & Métamodèle ML 3D

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![ÉTS](https://img.shields.io/badge/ÉTS-Génie_Aérospatial-CE181E?style=flat-square)

**Projet de fin de session — MGA 802**
École de technologie supérieure (ÉTS Montréal)

</div>

---

## 📌 Description

Ce projet implémente un système d'analyse hybride couplant un solveur **Éléments Finis 1D (FEA)**, un **solveur analytique** (théorie des poutres) et un **métamodèle de Machine Learning** (Random Forest) entraîné sur des données 3D extraites d'ANSYS.

L'objectif est de comparer une approche classique avec un modèle d'apprentissage automatique capable de prédire **instantanément**, en tout point 3D d'une poutre encastrée-libre :

- le **déplacement transversal** (flèche),
- la **contrainte normale** σₓ,
- la **contrainte équivalente de Von Mises**,

sous l'effet combiné d'une **force ponctuelle** et d'un **gradient de température**.

---

## ✨ Fonctionnalités

| Module | Description |
|---|---|
| 📐 **Solveur Analytique** | Référence théorique (Euler-Bernoulli) pour la flèche et les contraintes de flexion/thermique. |
| 🟢 **Solveur FEA 1D** | Assemblage matriciel (rigidité, 2 DDL/nœud), résolution `K·U = F`, sans facteur de calibration. |
| 🤖 **Métamodèle ML** | `RandomForestRegressor` dans un `Pipeline` avec `StandardScaler`, validé par **K-Fold** (k=5). Filtrage des singularités par le **principe de Saint-Venant**. |
| 🛡️ **Fallback de données** | Génération automatique de données synthétiques physiquement cohérentes si les fichiers ANSYS sont absents. |
| 📊 **Dashboard Streamlit** | Interface interactive : profils 2D, heatmaps 3D (Plotly), métriques comparatives, et analyse d'**extrapolation** du modèle ML hors de son domaine d'entraînement. |
| 🧪 **Suite de tests** | 88 tests unitaires/intégration (`pytest`) couvrant la physique, la cohérence inter-solveurs et le pipeline ML. |

---

## 📂 Architecture du Projet

```text
Projet_Fin_session/
│
├── donnees/                     # Données ANSYS (contrainte_F*_T*.txt, etc.)
├── solveurs/                    # Cœur du moteur de calcul
│   ├── __init__.py               # Expose les modules du package
│   ├── gestion_donnees.py       # Ingestion, fusion (NodeID) et fallback synthétique
│   ├── solveur_analytique.py    # Théorie des poutres (Euler-Bernoulli)
│   ├── solveur_ef.py            # Solveur Éléments Finis 1D
│   ├── modele_substitution.py   # Pipeline ML (Random Forest + K-Fold)
│   └── visualisation.py         # Graphiques Matplotlib (mode terminal)
│
├── tests/                       # Suite de tests automatisés
│   ├── __init__.py               # Expose les modules de test
│   ├── conftest.py              # Configuration des chemins pour Pytest
│   ├── test_systeme.py          # Solveurs Analytique/FEA + cohérence physique
│   └── test_ml.py               # Entraînement, prédiction, Saint-Venant
│
├── app.py                        # Dashboard interactif (Streamlit + Plotly)
├── main.py                       # Exécution en ligne de commande (Terminal)
├── config.yaml                   # Configuration centrale (géométrie, matériau, ML)
├── pyproject.toml                # Métadonnées, dépendances et config Pytest
├── LICENSE                       # Licence MIT
└── README.md
```

---

## ⚙️ Modèles Mathématiques

### Solveur Analytique (Résistance des Matériaux)

Pour une poutre rectangulaire encastrée-libre de longueur *L*, soumise à une force ponctuelle *F* en bout et un écart de température Δ*T* :

| Quantité | Formule |
|---|---|
| Moment quadratique | $I = \dfrac{b h^3}{12}$ |
| Flèche maximale | $\delta_{max} = \dfrac{F L^3}{3EI}$ |
| Contrainte de flexion max. | $\sigma_{flex} = \dfrac{Mc}{I} = \dfrac{F L \, c}{I}$ |
| Contrainte thermique | $\sigma_{th} = E \, \alpha \, \Delta T$ |
| Contrainte normale totale | $\sigma_x = \sigma_{flex} + \sigma_{th}$ |

### Solveur FEA 1D

Discrétisation en éléments de poutre Euler-Bernoulli (2 DDL/nœud : déplacement *v*, rotation *θ*), assemblage de la matrice de rigidité globale, encastrement appliqué par pénalité, résolution du système linéaire `K·U = F`.

### Métamodèle ML

`RandomForestRegressor` (scikit-learn) entraîné séparément pour les 3 cibles (Von Mises, σₓ, déplacement Y), à partir des variables d'entrée **(X, Y, Z, Force, Température)**. Validation croisée **K-Fold (k=5)** pour estimer le score R² hors échantillon avant l'entraînement final sur 100 % des données.

---

## 🚀 Installation et Démarrage

### 1. Prérequis

Python 3.9 ou plus récent.

```bash
git clone https://github.com/votre-utilisateur/Projet_Fin_session.git
cd Projet_Fin_session
pip install .
```

> Les dépendances sont déclarées dans `pyproject.toml`. Pour un environnement de développement modifiable :
> ```bash
> pip install -e ".[test]"
> ```

### 2. Dashboard interactif (recommandé)

Interface web avec sliders, heatmaps 3D et analyse d'extrapolation :

```bash
streamlit run app.py
```

### 3. Exécution terminal

Mode ligne de commande classique avec saisie interactive :

```bash
python main.py
```

---

## 🧪 Tests et Validation

```bash
pytest
```

> Les chemins de tests et options (`-ra -v`) sont préconfigurés dans `pyproject.toml`.

| Fichier | Couverture |
|---|---|
| `test_systeme.py` | Config YAML, formules analytiques, FEA 1D, cohérence Analytique↔FEA (< 1 % d'écart), symétrie/définie-positivité de la matrice K |
| `test_ml.py` | Parsing des noms de fichiers, chargement des données, scores K-Fold, monotonie des prédictions, filtre de Saint-Venant, importance des variables |

---
## 📚 Documentation Officielle (Sphinx)

Le code source de ce projet est entièrement documenté. Une documentation interactive générée automatiquement avec **Sphinx** est incluse dans ce dépôt.

Pour consulter les détails des modules, classes et fonctions :
1. Naviguez vers le dossier de documentation (par exemple : `docs/_build/html/`).
2. Ouvrez le fichier `index.html` avec votre navigateur web préféré (Chrome, Edge, Safari, etc.).
---
## 👨‍💻 Auteurs

- **Mohammad Shohoudimojdehi**
- **Nicolas Allard**

Développé dans le cadre du cours **MGA 802 — Projet de fin de session**
🎓 École de technologie supérieure (ÉTS Montréal)