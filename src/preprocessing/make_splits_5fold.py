"""
############# ETAPE1-3 ############

Étape folds : création des splits 5-fold stratifiés multi-label
et mapping vidéo ↔ frames ↔ labels ↔ fold.

Étapes :
1) Lire le JSON des labels par vidéo (track1-qv_pipe_train.json)
2) Construire un DataFrame vidéo avec vecteur multi-label
3) Appliquer MultilabelStratifiedKFold (n_splits=5)
4) Sauvegarder un CSV au niveau vidéo
5) Propager les folds aux frames 5_forstep1and2
6) Générer un tableau de répartition des classes par fold (pour le rapport)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# --- chemins ---
# JSON des labels multi-label par vidéo
LABELS_JSON = Path("data/labels/track1-qv_pipe_train.json")

# Dossier des frames déjà extraites (5 par vidéo, nettoyées)
FRAMES_DIR = Path("data/frames/5_forstep1and2")

# Sorties "techniques"
OUT_VIDEO_CSV = Path("data/splits/video_folds_5fold.csv")
OUT_FRAMES_CSV = Path("data/splits/frames_5_forstep1and2_folds.csv")

# Sortie "rapport" pour la distribution des classes
REPORTS_DIR = Path("reports/tables/preprocessing")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_labels():
    """
    Charge le JSON des labels par vidéo.
    Format attendu :
        {"d16427.mp4": [1, 4], ...}
    Retourne :
        df_videos : DataFrame avec colonnes ['video_id', 'labels_list']
    """
    with open(LABELS_JSON, "r") as f:
        data = json.load(f)

    records = []
    for vid, labs in data.items():
        # on trie + on enlève les doublons éventuels
        records.append({"video_id": vid, "labels_list": sorted(set(labs))})

    df = pd.DataFrame(records)
    return df


def build_multilabel_matrix(df_videos):
    """
    Construit la matrice multi-label Y (n_videos x n_classes)
    à partir de la colonne 'labels_list'.

    Retourne :
        Y (np.ndarray), classes_sorted (liste des id de classe triés)
    """
    # Ensemble de toutes les classes présentes dans le JSON
    all_labels = sorted({l for labs in df_videos["labels_list"] for l in labs})

    # Mapping classe -> index de colonne
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}

    # Matrice binaire (multi-hot) : 1 si la vidéo a la classe, 0 sinon
    Y = np.zeros((len(df_videos), len(all_labels)), dtype=int)
    for i, labs in enumerate(df_videos["labels_list"]):
        for lab in labs:
            Y[i, label_to_idx[lab]] = 1

    return Y, all_labels


def make_video_folds(df_videos, n_splits=5, random_state=42):
    """
    Ajoute une colonne 'fold' à df_videos, en utilisant
    une stratification multi-label (MultilabelStratifiedKFold).
    """
    Y, classes_sorted = build_multilabel_matrix(df_videos)

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = np.zeros(len(df_videos), dtype=int)  # fold pour chaque vidéo

    # X peut être un dummy (ici indices), la stratification se fait sur Y
    X_dummy = np.arange(len(df_videos))

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(X_dummy, Y)):
        # On affecte le numéro de fold aux indices de validation
        folds[val_idx] = fold_idx

    df_videos["fold"] = folds

    # Colonnes utiles : labels sous forme de chaîne
    df_videos["labels_str"] = df_videos["labels_list"].apply(
        lambda labs: " ".join(str(x) for x in sorted(labs))
    )

    return df_videos, classes_sorted


def build_frames_mapping(df_videos):
    """
    Construit un DataFrame au niveau frame :
    - frame_path
    - video_stem
    - labels_str
    - fold

    en partant des frames déjà extraites dans data/frames/5_forstep1and2.
    """

    # Ajouter une colonne 'video_stem' aux vidéos (sans extension .mp4)
    df_videos = df_videos.copy()
    df_videos["video_stem"] = df_videos["video_id"].str.replace(".mp4", "", regex=False)

    # Récupérer toutes les frames 5_forstep1and2
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))

    records = []
    for fp in frame_paths:
        # Ex: d16427_f00 -> video_stem = "d16427"
        stem = fp.stem  # "d16427_f00"
        video_stem = stem.split("_f")[0]

        records.append(
            {
                "frame_path": str(fp).replace("\\", "/"),  # chemin portable
                "video_stem": video_stem,
            }
        )

    df_frames = pd.DataFrame(records)

    # Merge avec df_videos pour récupérer labels et fold de la vidéo
    df_frames = df_frames.merge(
        df_videos[["video_stem", "labels_str", "fold"]],
        on="video_stem",
        how="left",
    )

    # Optionnel : vérifier si certaines frames n'ont pas trouvé de vidéo
    missing = df_frames["fold"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} frames n'ont pas trouvé de vidéo correspondante.")

    return df_frames


def compute_class_distribution_per_fold(df_videos, classes_sorted):
    """
    Construit un tableau (DataFrame) qui donne, pour chaque classe,
    le nombre de vidéos par fold + le total (pour le rapport).
    """
    # matrice (n_classes x n_folds)
    n_folds = df_videos["fold"].nunique()
    class_fold_counts = np.zeros((len(classes_sorted), n_folds), dtype=int)

    # Pour chaque vidéo, incrémenter les compteurs pour ses classes et son fold
    for _, row in df_videos.iterrows():
        labs = row["labels_list"]
        fold = int(row["fold"])
        for lab in labs:
            class_idx = classes_sorted.index(lab)
            class_fold_counts[class_idx, fold] += 1

    df_class_dist = pd.DataFrame(
        class_fold_counts,
        index=[f"class_{c}" for c in classes_sorted],
        columns=[f"fold_{i}" for i in range(n_folds)],
    )

    # Ajout colonne Total (toutes vidéos contenant cette classe)
    df_class_dist["Total"] = df_class_dist.sum(axis=1)

    return df_class_dist


def main():
    # 1) Charger les labels vidéos
    print(f"Lecture des labels depuis : {LABELS_JSON}")
    df_videos = load_labels()
    print(f"Nombre de vidéos dans le JSON : {len(df_videos)}")

    # 2) Construire les folds multi-label
    df_videos, classes_sorted = make_video_folds(df_videos, n_splits=5, random_state=42)
    print("Classes triées :", classes_sorted)
    print("Répartition du nombre de vidéos par fold :")
    print(df_videos["fold"].value_counts().sort_index())

    # 3) Tableau répartition classes × folds (pour le rapport)
    print("\nTableau de répartition des classes par fold :\n")
    df_class_dist = compute_class_distribution_per_fold(df_videos, classes_sorted)
    print(df_class_dist)

    # Sauvegarde du tableau dans reports/tables/preprocessing/
    dist_csv = REPORTS_DIR / "class_distribution_per_fold.csv"
    df_class_dist.to_csv(dist_csv)
    print(f"\n📄 Tableau classes/folds sauvegardé dans :\n{dist_csv}\n")

    # 4) Sauvegarder le CSV vidéo (idéal pour debug, stats, etc.)
    OUT_VIDEO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_videos.to_csv(OUT_VIDEO_CSV, index=False)
    print(f"CSV vidéo avec folds sauvegardé dans : {OUT_VIDEO_CSV}")

    # 5) Mapping frames ↔ labels ↔ fold
    df_frames = build_frames_mapping(df_videos)
    OUT_FRAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_frames.to_csv(OUT_FRAMES_CSV, index=False)
    print(f"CSV frames sauvegardé dans : {OUT_FRAMES_CSV}")


if __name__ == "__main__":
    main()
