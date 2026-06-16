# Projet Thermoélastique : Solveur EF et Modèle de Substitution (Surrogate Model)

## 📌 Description du Projet
Ce projet implémente un solveur par éléments finis (FEA) couplé à un modèle de substitution (Machine Learning) en Python. Il est conçu pour analyser le comportement thermoélastique tridimensionnel d'une poutre encastrée soumise à des charges mécaniques et thermiques.

L'objectif principal est d'utiliser un modèle d'apprentissage automatique (entraîné sur des données ANSYS) pour prédire instantanément :
1. Le champ de déplacement 3D en tout point de la structure.
2. La distribution spatiale des contraintes de Von Mises.

## Fonctionnalités

- Calcul analytique des déplacements
- Calcul analytique des contraintes
- Calcul avec ML Random Forest
- Visualisation interactive des résultats
- Export des données
- Validation des calculs avec ML Kfold

## 📂 Structure du Projet
Le code est organisé de manière modulaire (Programmation Orientée Objet) :

```text
projet_thermoelastique/
│
├── donnees/                    # Dossier contenant les données d'entraînement (ignoré par Git)
├── solveurs/                        # Code source du projet
│   ├── __init__.py
│   ├── gestion_donnees.py      # Importation et nettoyage des données ANSYS
│   ├── solveur_ef.py           # Solveur éléments finis 
│   ├── solveur_analytique.py   # Solveur analytique 
│   ├── modele_substitution.py  # Entraînement et prédiction (Machine Learning)
│   └── visualisation.py        # Génération des heatmaps 3D pour les résultats
├── tests/                        # Dossier de tests pytest
│   ├── __init__.py
│   ├── conftest.py      # Fichier de configuration pour pytest
│   ├── test_ml.py           # Test du modèle de substitution
│   └── test_systeme.py       # Test des solveurs analytique et solveur éléments finis, validation de la cohérence des résultas
├── app.py                     # Interface utilisateur développée avec Streamlit
├── config.yaml                     # Paramètres de configuration du système thermoélastique
├── main.py                     # Script principal d'exécution
├── requirements.txt            # Dépendances du projet
└── README.md                   # Documentation du projet

```

## Installation
Installer les dépendances :
```text
pip install -r requirements.txt
```
## Lancement du programme
```text
streamlit run app.py
```

## Méthodes utilisées

### Solveur analytique

Le solveur analytique utilise les équations classiques de la résistance des matériaux :

- Moment d'inertie :
  I = bh³/12

- Déplacement maximal :
  δ = FL³/(3EI)

- Contrainte maximale :
  σ = Mc/I

### Modèle de substitution

Le modèle de substitution est basé sur l'algorithme Random Forest et une étape de validation utilise le modèle Kfold.

## Tests
Éxécuter les tests :
```text
python -m pytest -v
```

## Auteurs

Mohammad Shohoudimojdehi et Nicolas Allard  
MGA 802 : Projet de fin de session  
École de technologie supérieure (ÉTS)  

