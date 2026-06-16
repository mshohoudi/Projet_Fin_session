"""
gestion_donnees.py
==================
Lecture, fusion et nettoyage des fichiers texte Ansys (format tab-séparé).
Merge basé sur NodeID (robuste aux Y/Z quasi-nuls d'Ansys).
"""

import os
import re
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Extraction Force/Température depuis le nom de fichier
# ------------------------------------------------------------------ #

def extraire_parametres_nom_fichier(nom_fichier: str):
    """
    Extrait la force et la température contenues dans un nom de fichier ANSYS.

    :param nom_fichier: Nom du fichier à analyser.
    :type nom_fichier: str

    :return: la force et la température contenus dans le fichier.
    :rtype: object
    """
    pattern = r"_F([\d.]+)_T([\d.]+)"
    match = re.search(pattern, nom_fichier, re.IGNORECASE)
    if match:
        f_kn = float(match.group(1))
        t_c  = float(match.group(2))
        return f_kn * 1000.0, t_c   # Conversion kN → N
    return None, None


# ------------------------------------------------------------------ #
#  Lecture d'un fichier Ansys (tab-séparé, 1 ligne d'en-tête)
# ------------------------------------------------------------------ #

def lire_fichier_ansys(chemin: str, nom_valeur: str) -> pd.DataFrame:
    """
    Lit un fichier de résultats ANSYS et le convertit en DataFrame.

    :param chemin: Chemin du fichier à lire.
    :type chemin: str
    :param nom_valeur: Nom de la colonne de valeurs à créer.
    :type nom_valeur: str

    :return: Fichier converti en DataFrame.
    :rtype: pd.DataFrame
    """
    try:
        df = pd.read_csv(chemin, sep="\t", header=0, engine="python")

        # Renommer les colonnes indépendamment de leur libellé exact
        df.columns = ["NodeID", "X", "Y", "Z", nom_valeur]

        # Tout convertir en numérique
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["NodeID", "X", "Z", nom_valeur])
        df["NodeID"] = df["NodeID"].astype(int)

        # Arrondir Y à 6 décimales (Ansys donne des Y ~1e-32 ≈ 0)
        df["Y"] = df["Y"].round(6)

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"  [ERREUR] Lecture de {os.path.basename(chemin)} : {e}")
        return pd.DataFrame()


# ------------------------------------------------------------------ #
#  Chargement et fusion de tous les fichiers d'un dossier
# ------------------------------------------------------------------ #

def charger_donnees_ansys(dossier: str) -> pd.DataFrame:
    """
    Charge et fusionne les fichiers de résultats ANSYS d’un dossier.

    :param dossier: Chemin du dossier à lire.
    :type dossier: str

    :return: Tous les fichiers d'entrée sous la forme d'un DataFrame
    :rtype: pd.DataFrame
    """
    if not os.path.isdir(dossier):
        print(f"  [INFO] Dossier introuvable : {dossier}")
        return pd.DataFrame()

    fichiers = [f for f in os.listdir(dossier) if f.endswith(".txt")]
    if not fichiers:
        print(f"  [INFO] Aucun fichier .txt dans : {dossier}")
        return pd.DataFrame()

    # ---- Groupement par (force_N, temp_C) ----
    groupes = {}
    for f in fichiers:
        force_N, temp_C = extraire_parametres_nom_fichier(f)
        if force_N is None:
            continue
        cle = (force_N, temp_C)
        groupes.setdefault(cle, {})
        chemin = os.path.join(dossier, f)
        nom = f.lower()
        if "deplacement_y" in nom:
            groupes[cle]["deplacement_y"] = chemin
        elif "contrainte_x" in nom:
            groupes[cle]["contrainte_x"] = chemin
        else:
            groupes[cle]["von_mises"] = chemin

    if not groupes:
        print("  [INFO] Aucun fichier reconnu (vérifier le format du nom du fichier).")
        return pd.DataFrame()

    print(f"\n  [DONNÉES] {len(groupes)} cas (F×T) détectés dans '{dossier}'")

    # ---- Fusion par cas ----
    frames = []
    for (force_N, temp_C), fichiers_cas in sorted(groupes.items()):
        f_kn = force_N / 1000

        if "von_mises" not in fichiers_cas:
            print(f"  [SKIP] F={f_kn:.0f}kN T={temp_C:.0f}°C → fichier Von Mises manquant")
            continue

        df_vm = lire_fichier_ansys(fichiers_cas["von_mises"], "von_mises")
        if df_vm.empty:
            continue

        df_merge = df_vm.copy()

        if "contrainte_x" in fichiers_cas:
            df_sx = lire_fichier_ansys(fichiers_cas["contrainte_x"], "contrainte_x")
            if not df_sx.empty:
                df_merge = df_merge.merge(
                    df_sx[["NodeID", "contrainte_x"]], on="NodeID", how="left"
                )

        if "deplacement_y" in fichiers_cas:
            df_dy = lire_fichier_ansys(fichiers_cas["deplacement_y"], "deplacement_y")
            if not df_dy.empty:
                df_merge = df_merge.merge(
                    df_dy[["NodeID", "deplacement_y"]], on="NodeID", how="left"
                )

        df_merge["force"]       = force_N
        df_merge["temperature"] = temp_C
        df_merge = df_merge.dropna(subset=["X", "Z", "von_mises"])
        frames.append(df_merge)

        print(f"    ✓  F={f_kn:5.0f} kN  T={temp_C:6.1f}°C  →  {len(df_merge):4d} noeuds")

    if not frames:
        return pd.DataFrame()

    resultat = pd.concat(frames, ignore_index=True)
    print(f"\n  [TOTAL] {len(resultat):,} noeuds chargés "
          f"({len(frames)} cas fusionnés)\n")
    return resultat


# ------------------------------------------------------------------ #
#  Génération de données synthétiques (fallback)
# ------------------------------------------------------------------ #

def generer_donnees_synthetiques(cfg: dict) -> pd.DataFrame:
    """
    Génère un jeu de données synthétiques lorsque les données ANSYS sont absentes.

    :param cfg: Dictionnaire de configuration du projet.
    :type cfg: dict

    :return: Jeu de données synthétiques.
    :rtype: pd.DataFrame
    """

    geo   = cfg["geometrie"]
    mat   = cfg["materiau"]
    synt  = cfg["donnees_synthetiques"]
    rng   = np.random.default_rng(42)

    L = geo["longueur"]; b = geo["base"]; h = geo["hauteur"]
    E = mat["module_young"]; alpha = mat["coeff_thermique"]
    T0 = mat["temperature_ref"]
    I  = (b * h**3) / 12.0

    forces = rng.uniform(*synt["plage_force"], size=synt["nb_cas"])
    temps  = rng.uniform(*synt["plage_temp"],  size=synt["nb_cas"])

    frames = []
    for F, T in zip(forces, temps):
        dT = T - T0
        n  = synt["nb_noeuds_par_cas"]
        X  = rng.uniform(0.001, L,    size=n)
        Y  = rng.uniform(-h/2,  h/2,  size=n)
        Z  = rng.uniform(0,     b,    size=n)

        M_x       = F * (L - X)
        sigma_x   = (M_x * Y) / I + E * alpha * dT
        tau       = 1.5 * F / (b*h) * (1 - (2*Y/h)**2)
        von_mises = np.sqrt(sigma_x**2 + 3*tau**2) * rng.normal(1.0, 0.01, n)
        v         = -(F * X**2 * (3*L - X)) / (6*E*I) * rng.normal(1.0, 0.005, n)

        frames.append(pd.DataFrame({
            "NodeID": range(1, n+1),
            "X": X, "Y": Y, "Z": Z,
            "von_mises":    np.abs(von_mises),
            "contrainte_x": sigma_x,
            "deplacement_y": v,
            "force": F, "temperature": T,
        }))

    donnees = pd.concat(frames, ignore_index=True)
    print(f"  [SYNTHÉTIQUE] {len(donnees):,} noeuds générés "
          f"({synt['nb_cas']} cas)\n")
    return donnees


# ------------------------------------------------------------------ #
#  Point d'entrée principal
# ------------------------------------------------------------------ #

def charger_ou_generer_donnees(cfg: dict) -> pd.DataFrame:
    """
    Charge les données ANSYS ou génère des données synthétiques de secours.

    :param cfg: Dictionnaire de configuration du projet.
    :type cfg: dict

    :return: Les données ANSYS ou celes synthétiques selon le cas.
    :rtype: pd.DataFrame

    :raises FileNotFoundError: Si aucune donnée n’est trouvée et si la génération synthétique est désactivée.
    """
    dossier = cfg["chemins"]["dossier_donnees"]
    df = charger_donnees_ansys(dossier)

    if df.empty:
        if cfg["donnees_synthetiques"]["activer"]:
            print("  [INFO] Bascule sur données synthétiques.")
            df = generer_donnees_synthetiques(cfg)
        else:
            raise FileNotFoundError(
                f"Aucune donnée dans '{dossier}' et synthèse désactivée."
            )

    # Garantir toutes les colonnes requises
    for col in ["von_mises", "contrainte_x", "deplacement_y"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=["X", "Y", "Z", "force", "temperature"])
    return df.reset_index(drop=True)