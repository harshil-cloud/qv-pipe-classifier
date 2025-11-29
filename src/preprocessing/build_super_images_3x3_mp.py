"""
Construction des super-images 3×3 (multiprocess) pour l'étape 3.

Règles :
- On part du CSV : data/splits/frames_9_forstep3_folds.csv
- On regroupe par video_stem.
- On ne garde que :
    - vidéos avec 9 frames -> super-image 3×3 normale
    - vidéos avec 8 frames -> on duplique 1 frame (la dernière) -> 9 frames
- On IGNORE les vidéos avec <= 7 frames.
- On construit une super-image 3×3 (9 tuiles) de taille 3 * 224 x 3 * 224.
- On sauvegarde dans : data/super_images/<video_stem>_3x3.jpg
- On génère le CSV : data/splits/super_images_3x3_folds.csv

Multiprocessing : 1 vidéo = 1 tâche.
"""

from pathlib import Path
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image

# Chemins d'entrée / sortie
FRAMES_SPLIT_CSV = Path("data/splits/frames_9_forstep3_folds.csv")
SUPERIM_DIR = Path("data/super_images")
SUPERIM_DIR.mkdir(parents=True, exist_ok=True)

OUT_SPLIT_CSV = Path("data/splits/super_images_3x3_folds.csv")

# Taille des tuiles
TILE_SIZE = (224, 224)  # (width, height)


def parse_frame_index(frame_path: str) -> int:
    """
    Extrait l'indice de frame à partir du nom de fichier.

    Exemple : d16427_f03.jpg -> 3
    """
    stem = Path(frame_path).stem
    m = re.search(r"_f(\d+)$", stem)
    if m:
        return int(m.group(1))
    return 0


def build_superimage_3x3(frame_paths):
    """
    Construit une super-image 3x3 à partir d'une liste de 9 chemins d'images.
    Retourne un objet PIL.Image.
    """
    assert len(frame_paths) == 9, "Il faut exactement 9 frames pour la super-image."

    w, h = TILE_SIZE
    grid_w, grid_h = 3 * w, 3 * h

    super_img = Image.new("RGB", (grid_w, grid_h))

    idx = 0
    for row in range(3):
        for col in range(3):
            fp = frame_paths[idx]
            img = Image.open(fp).convert("RGB")
            img = img.resize(TILE_SIZE, Image.BILINEAR)
            x = col * w
            y = row * h
            super_img.paste(img, (x, y))
            idx += 1

    return super_img


def worker_build_superimage(task):
    """
    Fonction appelée en parallèle.
    task est un dict avec :
        - video_stem
        - frame_paths (list de str)
        - labels_str
        - fold
    Retourne un dict avec les infos pour le CSV, ou None si échec.
    """
    video_stem = task["video_stem"]
    frame_paths = task["frame_paths"]
    labels_str = task["labels_str"]
    fold = task["fold"]

    n = len(frame_paths)
    if n < 8:
        # Ne devrait pas arriver si le filtrage est fait avant, mais on sécurise.
        return None

    # Si 8 frames : on duplique la dernière pour arriver à 9
    if n == 8:
        frame_paths = frame_paths + [frame_paths[-1]]
    elif n > 9:
        # Par sécurité, on garde les 9 premières triées
        frame_paths = frame_paths[:9]

    try:
        super_img = build_superimage_3x3(frame_paths)
    except Exception as e:
        print(f"[ERREUR] Construction super-image pour {video_stem} : {e}")
        return None

    out_name = f"{video_stem}_3x3.jpg"
    out_path = SUPERIM_DIR / out_name

    try:
        super_img.save(out_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"[ERREUR] Sauvegarde super-image pour {video_stem} : {e}")
        return None

    return {
        "video_stem": video_stem,
        "superimage_path": str(out_path).replace("\\", "/"),
        "labels_str": labels_str,
        "fold": fold,
    }


def main():
    if not FRAMES_SPLIT_CSV.exists():
        print(f"[ERREUR] CSV non trouvé : {FRAMES_SPLIT_CSV.resolve()}")
        return

    df = pd.read_csv(FRAMES_SPLIT_CSV)
    required = {"frame_path", "video_stem", "labels_str", "fold"}
    if not required.issubset(df.columns):
        print(f"[ERREUR] Colonnes manquantes : {required - set(df.columns)}")
        return

    # 1) On enlève les lignes sans fold (frames orphelines)
    n_nan_fold = df["fold"].isna().sum()
    if n_nan_fold > 0:
        print(f"[INFO] Frames avec fold manquant : {n_nan_fold} (elles seront ignorées)")
        df = df.dropna(subset=["fold"]).copy()

    # 2) Normalisation des types
    df["frame_path"] = df["frame_path"].astype(str)
    df["fold"] = df["fold"].astype(int)

    # Optionnel : si jamais labels_str a des NaN (rare mais possible)
    n_nan_labels = df["labels_str"].isna().sum()
    if n_nan_labels > 0:
        print(f"[INFO] Frames avec labels_str manquant : {n_nan_labels} (elles seront ignorées)")
        df = df.dropna(subset=["labels_str"]).copy()

    # 3) Ajout de l'indice de frame
    df["frame_idx"] = df["frame_path"].apply(parse_frame_index)

    # 4) Tri
    df = df.sort_values(["video_stem", "frame_idx"]).reset_index(drop=True)

    # 5) Nombre de frames par vidéo (après nettoyage)
    counts = df["video_stem"].value_counts().sort_index()

    # 6) Filtrage : on garde uniquement les vidéos avec >= 8 frames
    valid_videos = counts[counts >= 8].index.tolist()
    df_valid = df[df["video_stem"].isin(valid_videos)].copy()

    print(f"Vidéos totales dans le CSV : {counts.shape[0]}")
    print(f"Vidéos avec >= 8 frames    : {len(valid_videos)}")
    print(f"On ignore les vidéos avec <= 7 frames.")


    print(f"Vidéos totales dans le CSV : {counts.shape[0]}")
    print(f"Vidéos avec >= 8 frames    : {len(valid_videos)}")
    print(f"On ignore les vidéos avec <= 7 frames.")

    # Préparation des tâches pour multiprocessing
    tasks = []
    for video_stem, group in df_valid.groupby("video_stem"):
        frame_paths = group["frame_path"].tolist()
        n = len(frame_paths)
        if n < 8:
            continue  # au cas où

        # labels_str et fold supposés constants pour la vidéo
        labels_str = group["labels_str"].iloc[0]
        fold = int(group["fold"].iloc[0])

        tasks.append(
            {
                "video_stem": video_stem,
                "frame_paths": frame_paths,
                "labels_str": labels_str,
                "fold": fold,
            }
        )

    print(f"Tâches de super-image à traiter : {len(tasks)}")
    if not tasks:
        print("Aucune tâche, arrêt.")
        return

    # Multiprocessing
    n_workers = cpu_count()
    print(f"Utilisation de {n_workers} workers CPU pour la construction des super-images...")

    results = []
    with Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(worker_build_superimage, tasks, chunksize=16):
            if res is not None:
                results.append(res)

    print(f"Super-images construites avec succès : {len(results)}")

    if not results:
        print("Aucune super-image générée, arrêt.")
        return

    df_super = pd.DataFrame(results)
    OUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_super.to_csv(OUT_SPLIT_CSV, index=False, encoding="utf-8")
    print(f"CSV des super-images sauvegardé dans : {OUT_SPLIT_CSV.resolve()}")


if __name__ == "__main__":
    main()
