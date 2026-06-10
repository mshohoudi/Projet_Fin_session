"""
gestion_donnees.py
==================
Classe pour la lecture, fusion et nettoyage des fichiers texte Ansys,
et génération de données synthétiques.
"""

import os
import re
import numpy as np
import pandas as pd


class GestionnaireDonnees:
    """
    Classe responsable de la gestion des jeux de données (chargement Ansys ou synthétiques).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dossier_donnees = cfg["chemins"]["dossier_donnees"]
        self.config_synthetique = cfg.get("donnees_synthetiques", {})
        self.geo = cfg.get("geometrie", {})
        self.mat = cfg.get("materiau", {})

    # ------------------------------------------------------------------ #
    #  Extraction Force/Température depuis le nom de fichier
    # ------------------------------------------------------------------ #

    def _extraire_parametres_nom_fichier(self, nom_fichier: str):
        """
        Extrait (force_kN, temp_C) depuis le nom de fichier.
        """
        pattern = r"_F([\d.]+)_T([\d.]+)"
        match = re.search(pattern, nom_fichier, re.IGNORECASE)
        if match:
            f_kn = float(match.group(1))
            t_c = float(match.group(2))
            return f_kn * 1000.0, t_c  # Conversion kN → N
        return None, None

    # ------------------------------------------------------------------ #
    #  Lecture d'un fichier Ansys
    # ------------------------------------------------------------------ #

    def _lire_fichier_ansys(self, chemin: str, nom_valeur: str) -> pd.DataFrame:
        """
        Lit un fichier Ansys tab-séparé et le nettoie.
        """
        try:
            df = pd.read_csv(chemin, sep="\t", header=0, engine="python")
            df.columns = ["NodeID", "X", "Y", "Z", nom_valeur]

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["NodeID", "X", "Z", nom_valeur])
            df["NodeID"] = df["NodeID"].astype(int)
            df["Y"] = df["Y"].round(6)

            return df.reset_index(drop=True)

        except Exception as e:
            print(f"  [ERREUR] Lecture de {os.path.basename(chemin)} : {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Chargement et fusion de tous les fichiers
    # ------------------------------------------------------------------ #

    def _charger_donnees_ansys(self) -> pd.DataFrame:
        """
        Scanne le dossier et fusionne les fichiers par (F, T) et NodeID.
        """
        if not os.path.isdir(self.dossier_donnees):
            print(f"  [INFO] Dossier introuvable : {self.dossier_donnees}")
            return pd.DataFrame()

        fichiers = [f for f in os.listdir(self.dossier_donnees) if f.endswith(".txt")]
        if not fichiers:
            print(f"  [INFO] Aucun fichier .txt dans : {self.dossier_donnees}")
            return pd.DataFrame()

        groupes = {}
        for f in fichiers:
            force_N, temp_C = self._extraire_parametres_nom_fichier(f)
            if force_N is None:
                continue
            cle = (force_N, temp_C)
            groupes.setdefault(cle, {})
            chemin = os.path.join(self.dossier_donnees, f)
            nom = f.lower()

            if "deplacement_y" in nom:
                groupes[cle]["deplacement_y"] = chemin
            elif "contrainte_x" in nom:
                groupes[cle]["contrainte_x"] = chemin
            else:
                groupes[cle]["von_mises"] = chemin

        if not groupes:
            print("  [INFO] Aucun fichier reconnu (vérifier le format de nommage).")
            return pd.DataFrame()

        print(f"\n  [DONNÉES] {len(groupes)} cas (F×T) détectés dans '{self.dossier_donnees}'")

        frames = []
        for (force_N, temp_C), fichiers_cas in sorted(groupes.items()):
            f_kn = force_N / 1000

            if "von_mises" not in fichiers_cas:
                continue

            df_vm = self._lire_fichier_ansys(fichiers_cas["von_mises"], "von_mises")
            if df_vm.empty:
                continue

            df_merge = df_vm.copy()

            if "contrainte_x" in fichiers_cas:
                df_sx = self._lire_fichier_ansys(fichiers_cas["contrainte_x"], "contrainte_x")
                if not df_sx.empty:
                    df_merge = df_merge.merge(df_sx[["NodeID", "contrainte_x"]], on="NodeID", how="left")

            if "deplacement_y" in fichiers_cas:
                df_dy = self._lire_fichier_ansys(fichiers_cas["deplacement_y"], "deplacement_y")
                if not df_dy.empty:
                    df_merge = df_merge.merge(df_dy[["NodeID", "deplacement_y"]], on="NodeID", how="left")

            df_merge["force"] = force_N
            df_merge["temperature"] = temp_C
            df_merge = df_merge.dropna(subset=["X", "Z", "von_mises"])
            frames.append(df_merge)

            print(f"    ✓  F={f_kn:5.0f} kN  T={temp_C:6.1f}°C  →  {len(df_merge):4d} noeuds")

        if not frames:
            return pd.DataFrame()

        resultat = pd.concat(frames, ignore_index=True)
        print(f"\n  [TOTAL] {len(resultat):,} noeuds chargés ({len(frames)} cas fusionnés)\n")
        return resultat

    # ------------------------------------------------------------------ #
    #  Génération de données synthétiques (fallback)
    # ------------------------------------------------------------------ #

    def _generer_donnees_synthetiques(self) -> pd.DataFrame:
        """
        Génère des données de simulation si aucun fichier Ansys n'est trouvé.
        """
        rng = np.random.default_rng(42)
        L = self.geo["longueur"]
        b = self.geo["base"]
        h = self.geo["hauteur"]
        E = self.mat["module_young"]

        I = (b * h ** 3) / 12.0

        forces = rng.uniform(*self.config_synthetique["plage_force"], size=self.config_synthetique["nb_cas"])
        temps = rng.uniform(*self.config_synthetique["plage_temp"], size=self.config_synthetique["nb_cas"])

        frames = []
        for F, T in zip(forces, temps):
            n = self.config_synthetique["nb_noeuds_par_cas"]
            X = rng.uniform(0.001, L, size=n)
            Y = rng.uniform(-h / 2, h / 2, size=n)
            Z = rng.uniform(0, b, size=n)

            M_x = F * (L - X)
            # Correction de la contrainte thermique appliquée ici
            sigma_x = (M_x * Y) / I
            tau = 1.5 * F / (b * h) * (1 - (2 * Y / h) ** 2)
            von_mises = np.sqrt(sigma_x ** 2 + 3 * tau ** 2) * rng.normal(1.0, 0.01, n)
            v = -(F * X ** 2 * (3 * L - X)) / (6 * E * I) * rng.normal(1.0, 0.005, n)

            frames.append(pd.DataFrame({
                "NodeID": range(1, n + 1),
                "X": X, "Y": Y, "Z": Z,
                "von_mises": np.abs(von_mises),
                "contrainte_x": sigma_x,
                "deplacement_y": v,
                "force": F, "temperature": T,
            }))

            donnees = pd.concat(frames, ignore_index=True)

        print(f"  [SYNTHÉTIQUE] {len(donnees):,} noeuds générés ({self.config_synthetique['nb_cas']} cas)\n")
        return donnees

    # ------------------------------------------------------------------ #
    #  Point d'entrée principal
    # ------------------------------------------------------------------ #

    def charger_ou_generer_donnees(self) -> pd.DataFrame:
        """
        Méthode publique principale pour récupérer le DataFrame final.
        """
        df = self._charger_donnees_ansys()

        if df.empty:
            if self.config_synthetique.get("activer", False):
                print("  [INFO] Bascule sur données synthétiques.")
                df = self._generer_donnees_synthetiques()
            else:
                raise FileNotFoundError(
                    f"Aucune donnée dans '{self.dossier_donnees}' et synthèse désactivée."
                )

        # Garantir toutes les colonnes requises
        for col in ["von_mises", "contrainte_x", "deplacement_y"]:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=["X", "Y", "Z", "force", "temperature"])
        return df.reset_index(drop=True)