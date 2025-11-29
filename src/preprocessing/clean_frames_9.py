"""
############# ETAPE1-2 (préparation étape 3) ############

Nettoyage des frames extraites pour l'étape 3 :
- Détection et suppression des images floues (variance du Laplacien)
- Détection et suppression des doublons (pHash exact)
- Déplacement des images rejetées dans un dossier séparé

BUT :
Garder uniquement des frames propres et uniques pour la construction
des super-images 3×3.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# ===============================================================
# --- 1) LOCALISATION DES IMAGES À NETTOYER ---
# ===============================================================

# Dossier contenant les frames extraites (9 par vidéo)
FRAMES_DIR = Path("data/frames/9_forstep3")

# Dossiers où les images rejetées seront stockées
REMOVED_BLUR_DIR = Path("data/frames/9_forstep3_removed/blurred")
REMOVED_DUP_DIR  = Path("data/frames/9_forstep3_removed/duplicates")

REMOVED_BLUR_DIR.mkdir(parents=True, exist_ok=True)
REMOVED_DUP_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================
# --- 2) PARAMÈTRES ---
# ===============================================================

BLUR_THRESHOLD = 100.0


def is_blurry(img_path: Path, threshold: float = BLUR_THRESHOLD):
    """
    Retourne (True, variance) si l'image est floue,
    (False, variance) sinon.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return True, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()

    return lap_var < threshold, lap_var


def main():
    # 4.1 Récupération de toutes les images extraites
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

    # 4.2 Première passe : filtre flou
    kept_images = []

    for img_path in tqdm(img_paths, desc="Filtrage des images floues"):
        blurry, lap_var = is_blurry(img_path)

        if blurry:
            target = REMOVED_BLUR_DIR / img_path.name
            shutil.move(str(img_path), str(target))
        else:
            kept_images.append(img_path)

    print(f"Images nettes conservées après filtre flou : {len(kept_images)}")

    # 4.3 Deuxième passe : suppression des doublons (pHash)
    hash_dict = {}
    dup_count = 0

    for img_path in tqdm(kept_images, desc="Filtrage des doublons (pHash)"):
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img)
        except Exception:
            target = REMOVED_DUP_DIR / img_path.name
            shutil.move(str(img_path), str(target))
            continue

        if h in hash_dict:
            target = REMOVED_DUP_DIR / img_path.name
            shutil.move(str(img_path), str(target))
            dup_count += 1
        else:
            hash_dict[h] = img_path

    print("================================================")
    print("   ✔ NETTOYAGE TERMINÉ (9 frames/vidéo)")
    print("================================================")
    print(f"Images floues déplacées  : {len(img_paths) - len(kept_images)}")
    print(f"Images doublons déplacées : {dup_count}")
    print("Images finales conservées  :", len(kept_images) - dup_count)
    print("Dossier final : ", FRAMES_DIR.resolve())


if __name__ == "__main__":
    main()
