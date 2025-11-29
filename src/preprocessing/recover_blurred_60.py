from pathlib import Path
import shutil
import cv2
from multiprocessing import Pool, cpu_count

FRAMES_DIR = Path("data/frames/9_forstep3")
BLURRED_DIR = Path("data/frames/9_forstep3_removed/blurred")
NEW_BLUR_THRESHOLD = 60.0

def test_one_image(img_path_str):
    """ Fonction appelée en parallèle """
    img_path = Path(img_path_str)
    img = cv2.imread(str(img_path))

    if img is None:
        return (img_path_str, False)  # laisse dans blurred

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # True si doit être récupérée
    return (img_path_str, lap_var >= NEW_BLUR_THRESHOLD)


def main():
    blurred_paths = list(BLURRED_DIR.glob("*.jpg")) + list(BLURRED_DIR.glob("*.jpeg")) + list(BLURRED_DIR.glob("*.png"))
    blurred_paths = [str(p) for p in blurred_paths]

    print(f"{len(blurred_paths)} images floues à re-tester avec seuil {NEW_BLUR_THRESHOLD}")
    print(f"Utilisation de {cpu_count()} coeurs CPU")

    with Pool(cpu_count()) as pool:
        results = pool.map(test_one_image, blurred_paths)

    recovered = 0
    kept = 0

    for img_path_str, should_recover in results:
        img_path = Path(img_path_str)
        if should_recover:
            target = FRAMES_DIR / img_path.name
            shutil.move(str(img_path), str(target))
            recovered += 1
        else:
            kept += 1

    print("===================================")
    print(f"Images récupérées : {recovered}")
    print(f"Images conservées dans blurred/ : {kept}")
    print("===================================")

if __name__ == "__main__":
    main()
