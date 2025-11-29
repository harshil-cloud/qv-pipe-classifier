import os
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import average_precision_score

import timm


# ============================================================
#  Utils
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(num_classes: int, model_name: str = "convnext_base", pretrained: bool = True):
    """
    Crée un modèle de classification multi-label basé sur timm.
    La dernière couche est remplacée par un linear(num_classes).
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# ============================================================
#  Dataset
# ============================================================

class SuperImageDataset(Dataset):
    """
    Dataset pour les super-images 3x3.

    CSV attendu avec les colonnes :
    - video_stem       : identifiant de la vidéo (pas forcément utilisé pour le training)
    - superimage_path  : chemin vers l'image 3x3 (png/jpg)
    - labels_str       : chaîne "0 1 0 0 1 ..." ou "0,1,0,0,1,..."
    - fold             : entier du fold (0..4 ou autre)
    """

    def __init__(self, csv_path: str, split: str, fold: int, transform=None):
        """
        split : 'train' ou 'val'
        fold  : numéro du fold courant
        """
        df = pd.read_csv(csv_path)

        if "fold" not in df.columns:
            raise ValueError("La colonne 'fold' est absente du CSV.")

        if "superimage_path" not in df.columns:
            raise ValueError("La colonne 'superimage_path' est absente du CSV.")

        if "labels_str" not in df.columns:
            raise ValueError("La colonne 'labels_str' est absente du CSV.")

        # Split train / val basé uniquement sur la colonne 'fold'
        if split == "train":
            df = df[df["fold"] != fold]
        elif split == "val":
            df = df[df["fold"] == fold]
        else:
            raise ValueError(f"split doit être 'train' ou 'val', pas '{split}'")

        if len(df) == 0:
            raise ValueError(f"Dataset {split} vide pour fold={fold} – vérifie le CSV.")

        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Déduit le nombre de classes à partir de la première ligne
        example_labels = self._parse_labels(self.df.loc[0, "labels_str"])
        self.num_classes = len(example_labels)

    @staticmethod
    def _parse_labels(labels_str: str) -> torch.Tensor:
        """
        Convertit labels_str en vecteur tensor float [0,1,...].
        Gère séparateur espace ou virgule.
        """
        if labels_str is None or (isinstance(labels_str, float) and np.isnan(labels_str)):
            raise ValueError("labels_str est vide ou NaN")

        # Remplace les virgules par des espaces, split, cast en float
        parts = str(labels_str).replace(",", " ").split()
        labels = [float(x) for x in parts]
        return torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row["superimage_path"]
        labels_str = row["labels_str"]

        # Chargement image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = self._parse_labels(labels_str)

        return img, target


# ============================================================
#  Training & Evaluation
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_targets = []
    all_logits = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    y_true = np.concatenate(all_targets, axis=0)
    y_scores = np.concatenate(all_logits, axis=0)

    # mAP (mean Average Precision) multi-label
    try:
        ap_per_class = average_precision_score(y_true, y_scores, average=None)
        ap_per_class = np.array(ap_per_class, dtype=float)
        map_macro = np.nanmean(ap_per_class)
    except Exception as e:
        print(f"[WARN] Erreur calcul mAP: {e}")
        map_macro = float("nan")

    return epoch_loss, map_macro


# ============================================================
#  Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training super-image 3x3 baseline (multi-label)"
    )

    parser.add_argument(
        "--splits_csv",
        type=str,
        required=True,
        help="Chemin vers super_images_3x3_folds.csv",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Dossier de sortie pour les modèles entraînés",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold courant utilisé pour la validation (train = fold != this)",
    )

    parser.add_argument("--model_name", type=str, default="convnext_base",
                        help="Backbone timm (ex: convnext_base, tf_efficientnet_b4_ns, nfnet_f0, tresnet_xl, ...)")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Normalisation ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # Datasets & DataLoaders
    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("[INFO] Loading datasets...")
    train_dataset = SuperImageDataset(
        csv_path=args.splits_csv,
        split="train",
        fold=args.fold,
        transform=train_tfms,
    )
    val_dataset = SuperImageDataset(
        csv_path=args.splits_csv,
        split="val",
        fold=args.fold,
        transform=val_tfms,
    )

    num_classes = train_dataset.num_classes
    print(f"[INFO] Num train samples: {len(train_dataset)}")
    print(f"[INFO] Num val samples  : {len(val_dataset)}")
    print(f"[INFO] Num classes      : {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Modèle, loss, optim
    print(f"[INFO] Creating model: {args.model_name}")
    model = create_model(num_classes=num_classes, model_name=args.model_name, pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Dossier des checkpoints pour ce fold
    fold_dir = os.path.join(args.models_dir, f"fold_{args.fold}")
    os.makedirs(fold_dir, exist_ok=True)
    best_model_path = os.path.join(fold_dir, "best.pth")

    # Boucle d'entraînement
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} (fold {args.fold}) =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_map = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val mAP: {val_map:.4f}")

        # Sauvegarde du meilleur modèle (selon mAP)
        improved = False
        if not np.isnan(val_map) and val_map > best_map:
            best_map = val_map
            improved = True
            torch.save(model.state_dict(), best_model_path)

        if improved:
            print(f"[INFO] New best mAP: {best_map:.4f} -> model saved to {best_model_path}")

    print(f"\n[INFO] Training finished for fold {args.fold}. Best mAP = {best_map:.4f}")


if __name__ == "__main__":
    main()
