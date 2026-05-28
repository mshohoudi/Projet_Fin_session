# Projet Thermoélastique : Solveur EF et Modèle de Substitution (Surrogate Model)

## 📌 Description du Projet
Ce projet implémente un solveur par éléments finis (FEA) couplé à un modèle de substitution (Machine Learning) en Python. Il est conçu pour analyser le comportement thermoélastique tridimensionnel d'une poutre encastrée soumise à des charges mécaniques et thermiques.

L'objectif principal est d'utiliser un modèle d'apprentissage automatique (entraîné sur des données ANSYS) pour prédire instantanément :
1. Le champ de déplacement 3D en tout point de la structure.
2. La distribution spatiale des contraintes de Von Mises.

## 📂 Structure du Projet
Le code est organisé de manière modulaire (Programmation Orientée Objet) :

```text
projet_thermoelastique/
│
├── donnees/                    # Dossier contenant les données d'entraînement (ignoré par Git)
├── source/                        # Code source du projet
│   ├── __init__.py
│   ├── gestion_donnees.py      # Importation et nettoyage des données ANSYS
│   ├── solveur_ef.py           # Solveur analytique 
│   ├── solveur_ef.py           # Solveur analytique 
│   ├── modele_substitution.py  # Entraînement et prédiction (Machine Learning)
│   └── visualisation.py        # Génération des heatmaps 3D
│
├── main.py                     # Script principal d'exécution
├── requirements.txt            # Dépendances du projet
└── README.md                   # Documentation
