from pathlib import Path
import pandas as pd

# Nombre de frames attendu pour une super-image
SUPERIMAGE_N = 9

# Dossier contenant les frames nettoyées pour l'étape 3
FRAMES_DIR = Path("data/frames/9_forstep3")

# Dossier + fichier de sortie
OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "repartition_superimage.csv"

def main():
    # Récupère toutes les images (adaptable si tu as du .png, etc.)
    frame_paths = sorted(
        list(FRAMES_DIR.glob("*.jpg")) +
        list(FRAMES_DIR.glob("*.jpeg")) +
        list(FRAMES_DIR.glob("*.png"))
    )

    if not frame_paths:
        print(f"Aucune image trouvée dans {FRAMES_DIR.resolve()}")
        return

    records = []
    for p in frame_paths:
        # Exemple de nom : d16427_f00.jpg -> video_stem = "d16427"
        stem = p.stem
        video_stem = stem.split("_f")[0]
        records.append(
            {
                "video_stem": video_stem,
                "frame_path": str(p).replace("\\", "/"),
            }
        )

    df_frames = pd.DataFrame(records)

    # Comptage du nombre de frames restantes par vidéo
    df_counts = (
        df_frames
        .groupby("video_stem")
        .size()
        .reset_index(name="n_frames")
    )

    # Nombre de duplications nécessaires pour arriver à 9
    df_counts["duplications_needed"] = (SUPERIMAGE_N - df_counts["n_frames"]).clip(lower=0)

    # Description textuelle du cas (0 = super-image complète,
    # 1 = 1 duplication, ..., 8 = 9 fois la même photo)
    def case_label(k):
        if k == 0:
            return "0_duplications (super-image complète)"
        elif k == 1:
            return "1_duplication"
        elif k == SUPERIMAGE_N - 1:
            # n_frames = 1 -> 8 duplications = 9 fois la même photo
            return f"{k}_duplications (9x la même photo)"
        else:
            return f"{k}_duplications"

    df_counts["case"] = df_counts["duplications_needed"].apply(case_label)

    # Sauvegarde du CSV
    df_counts.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("Fichier sauvegardé dans :", OUT_PATH.resolve())

    # Petit résumé en console
    print("\nRépartition des vidéos par nombre de duplications nécessaires :")
    print(df_counts["duplications_needed"].value_counts().sort_index())


if __name__ == "__main__":
    main()
