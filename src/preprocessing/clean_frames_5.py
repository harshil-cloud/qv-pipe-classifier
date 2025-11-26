"""
############# ETAPE1-2  ############

Nettoyage des frames extraites pour l'étape 1 :
- Détection et suppression des images floues (variance du Laplacien)
- Détection et suppression des doublons (pHash exact)
- Déplacement des images rejetées dans un dossier séparé

BUT :
Garder uniquement des frames propres et uniques pour l'étape 2 (baseline).
"""

from pathlib import Path
import shutil  # pour déplacer les fichiers
import cv2     # OpenCV : lecture images + Laplacien
import numpy as np
from PIL import Image  # pour pHash
import imagehash       # pour pHash
from tqdm import tqdm   # barre de progression


# ===============================================================
# --- 1) LOCALISATION DES IMAGES À NETTOYER ---
# ===============================================================

# Dossier contenant les frames extraites (5 par vidéo)
FRAMES_DIR = Path("data/frames/5_forstep1and2")

# Dossiers où les images rejetées seront stockées
REMOVED_BLUR_DIR = Path("data/frames/5_forstep1and2_removed/blurred")
REMOVED_DUP_DIR  = Path("data/frames/5_forstep1and2_removed/duplicates")

# Création des dossiers si pas existants
REMOVED_BLUR_DIR.mkdir(parents=True, exist_ok=True)
REMOVED_DUP_DIR.mkdir(parents=True, exist_ok=True)


# ===============================================================
# --- 2) PARAMÈTRES ---
# ===============================================================

BLUR_THRESHOLD = 100.0  
"""
Seuil de flou : si variance du Laplacien < seuil → image jugée floue.

Explication :
La variance du Laplacien mesure les détails d’une image :
- Une image nette = beaucoup d'arêtes = variance élevée
- Une image floue = peu d'arêtes = variance faible

On choisit donc un seuil empirique (entre 50 et 200 généralement).

Avec BLUR_THRESHOLD = 100.0 on obtient 17098 photos jugées comme floues sur un total de 48005.
"""


# ===============================================================
# --- 3) FONCTION DE DÉTECTION DU FLOU ---
# ===============================================================

def is_blurry(img_path: Path, threshold: float = BLUR_THRESHOLD):
    """
    Retourne (True, variance) si l'image est floue,
    (False, variance) sinon.

    LOGIQUE :
    - On convertit en niveaux de gris
    - On applique le Laplacien (détecteur d'arêtes)
    - On calcule la variance
    - Si variance < seuil → image floue
    """

    # Lecture de l'image avec OpenCV
    img = cv2.imread(str(img_path))

    # Si OpenCV n'arrive pas à lire l'image → la traiter comme floue
    if img is None:
        return True, 0.0

    # Passage en niveaux de gris (simplifie le calcul)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcul du Laplacien (détection d'arêtes)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    # Variance = niveau de détails
    lap_var = lap.var()

    # Image considérée floue si variance < seuil
    return lap_var < threshold, lap_var


# ===============================================================
# --- 4) SCRIPT PRINCIPAL ---
# ===============================================================

def main():

    # ---------------------------------------------------------------
    # 4.1 Récupération de toutes les images extraites
    # ---------------------------------------------------------------
    img_paths = sorted(
        list(FRAMES_DIR.glob("*.jpg")) +
        list(FRAMES_DIR.glob("*.jpeg")) +
        list(FRAMES_DIR.glob("*.png"))
    )

    if not img_paths:
        print(f"Aucune image trouvée dans {FRAMES_DIR.resolve()}")
        return

    print(f"Nombre d'images détectées : {len(img_paths)}")
    print(f"Seuil flou utilisé : {BLUR_THRESHOLD}")

    # ---------------------------------------------------------------
    # 4.2 PREMIÈRE PASSE : SUPPRESSION DES IMAGES FLOUES
    # ---------------------------------------------------------------

    kept_images = []  # images conservées (nettes)

    for img_path in tqdm(img_paths, desc="Filtrage des images floues"):

        blurry, lap_var = is_blurry(img_path)

        if blurry:
            # Déplacer vers dossier des images floues
            target = REMOVED_BLUR_DIR / img_path.name
            shutil.move(str(img_path), str(target))
        else:
            # Image conservée pour la suite
            kept_images.append(img_path)

    print(f"Images nettes conservées après filtre flou : {len(kept_images)}")


    # ---------------------------------------------------------------
    # 4.3 DEUXIÈME PASSE : SUPPRESSION DES DOUBLONS (pHash exact)
    # ---------------------------------------------------------------

    """
    LOGIQUE :
    - On calcule le pHash de chaque image nette
    - Si un pHash a déjà été vu → image = doublon
    - On garde la première occurrence, on déplace les autres
    """

    hash_dict = {}  # dictionnaire : hash → chemin image gardée
    dup_count = 0   # nombre de doublons supprimés

    for img_path in tqdm(kept_images, desc="Filtrage des doublons (pHash)"):
        try:
            # Ouvrir l'image avec PIL (nécessaire pour pHash)
            img = Image.open(img_path)

            # Calcul du pHash (perceptual hash)
            h = imagehash.phash(img)

        except Exception as e:
            # Problème de lecture = image "à problème"
            target = REMOVED_DUP_DIR / img_path.name
            shutil.move(str(img_path), str(target))
            continue

        if h in hash_dict:
            # pHash déjà rencontré → image doublon
            target = REMOVED_DUP_DIR / img_path.name
            shutil.move(str(img_path), str(target))
            dup_count += 1
        else:
            # Première image avec ce hash → on la garde
            hash_dict[h] = img_path


    # ---------------------------------------------------------------
    # 4.4 Résumé du nettoyage
    # ---------------------------------------------------------------

    print("================================================")
    print("   ✔ NETTOYAGE TERMINÉ")
    print("================================================")
    print(f"Images floues déplacées  : {len(img_paths) - len(kept_images)}")
    print(f"Images doublons déplacées : {dup_count}")
    print("Images finales conservées  :", len(kept_images) - dup_count)
    print("Dossier final : ", FRAMES_DIR.resolve())


# Exécution du script
if __name__ == "__main__":
    main()
