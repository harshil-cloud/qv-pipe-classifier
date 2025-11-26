"""
############# ETAPE1-1 ############

Script d'extraction de 5 frames uniformément espacées par vidéo.

Entrée :
    data/raw/videos/*.mp4

Sortie :
    data/frames/5_forstep1and2/nomvideo_f00.jpg ... nomvideo_f04.jpg
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# --- Paramètres principaux ---
N_FRAMES = 5  # nombre de frames à extraire par vidéo

# Dossier contenant les vidéos d'entrée
VIDEO_DIR = Path("data/raw_videos/videos")

# Dossier de sortie pour les frames extraites
OUTPUT_DIR = Path("data/frames/5_forstep1and2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_uniform_frames_for_video(video_path: Path, n_frames: int = N_FRAMES):
    """
    Extrait n_frames régulièrement espacées d'une vidéo
    et les sauvegarde dans OUTPUT_DIR.

    Retourne une liste de tuples (frame_index, frame_path).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Impossible d'ouvrir la vidéo : {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"[WARN] Frame count = 0 pour : {video_path}")
        cap.release()
        return []

    # Indices uniformément répartis sur [0, frame_count - 1]
    indices = np.linspace(0, frame_count - 1, num=n_frames, dtype=int)

    saved = []
    for i, idx in enumerate(indices):
        # Positionner la vidéo sur la frame idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Lecture impossible à la frame {idx} ({video_path.name})")
            continue

        # Nom du fichier de sortie : nomvideo_fXX.jpg
        out_name = f"{video_path.stem}_f{i:02d}.jpg"
        out_path = OUTPUT_DIR / out_name

        # Sauvegarde de l'image
        success = cv2.imwrite(str(out_path), frame)
        if not success:
            print(f"[WARN] Échec de l'écriture de l'image : {out_path}")
            continue

        saved.append((int(idx), out_path))

    cap.release()
    return saved

def main():
    # On récupère toutes les vidéos mp4 (minuscules ou majuscules)
    video_paths = sorted(
        list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.MP4"))
    )

    if not video_paths:
        print(f"Aucune vidéo .mp4 trouvée dans {VIDEO_DIR.resolve()}")
        return

    print(f"Nombre de vidéos trouvées : {len(video_paths)}")
    print(f"Extraction de {N_FRAMES} frames par vidéo...")
    print(f"Sortie : {OUTPUT_DIR.resolve()}")

    for vp in tqdm(video_paths, desc="Extraction frames (5 par vidéo)"):
        _ = extract_uniform_frames_for_video(vp, N_FRAMES)

    print("Extraction terminée.")

if __name__ == "__main__":
    main()