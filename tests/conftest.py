"""
conftest.py
===========
Fichier de configuration pour pytest.
Permet d'ajouter les répertoires nécessaires au PYTHONPATH avant l'exécution des tests.
"""

import sys
import os

# Ajouter le dossier 'source' au chemin Python (Python path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solveurs"))

# Ajouter la racine du projet (pour pouvoir lire le fichier config.yaml)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
