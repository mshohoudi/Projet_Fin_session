import matplotlib.pyplot as plt
import numpy as np


class Visualiseur3D:
    def __init__(self):
        pass

    def tracer_heatmap_3d(self, coordonnees, valeurs, titre, nom_unite):
        """
        Trace un nuage de points 3D (heatmap) représentant la pièce mécanique.
        :param coordonnees: Matrice (N, 3) contenant X, Y, Z.
        :param valeurs: Matrice (N,) contenant la grandeur physique (Contrainte ou Déplacement).
        :param titre: Titre du graphique.
        :param nom_unite: Unité de la grandeur physique (ex: Pa, m).
        """
        # Création de la figure
        fig = plt.figure(figsize=(10, 8))
        # Ajout d'un axe 3D
        ax = fig.add_subplot(111, projection='3d')

        # Extraction des coordonnées X, Y, Z
        x = coordonnees[:, 0]
        y = coordonnees[:, 1]
        z = coordonnees[:, 2]

        # Création du nuage de points (scatter)
        # c=valeurs détermine la couleur, cmap='jet' est le style de couleur standard en FEA (bleu à rouge)
        img = ax.scatter(x, y, z, c=valeurs, cmap='jet', marker='o', s=30, alpha=0.9)

        # Ajout de la barre de couleur (légende à droite)
        cbar = fig.colorbar(img, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label(nom_unite, fontsize=12)

        # Configuration des axes
        ax.set_xlabel('Position X (m)')
        ax.set_ylabel('Position Y (m)')
        ax.set_zlabel('Position Z (m)')
        ax.set_title(titre, fontsize=14, fontweight='bold')

        # Affichage
        plt.show()