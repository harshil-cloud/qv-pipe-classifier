"""
src/train/framewise_baseline.py

Baseline frame-wise training script for QV Pipe (Step 2).

- One sample = one video (list of up to K=5 frames)
- Backbone: timm (ResNet-18 by default)
- Loss: BCEWithLogitsLoss (multi-label)
- Optimizer: AdamW
- Scheduler: OneCycleLR
- AMP: torch.cuda.amp (mixed precision)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

import timm
import torchvision.transforms as T


# ----------------------------
# Utils
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline frame-wise training (Step 2)")

    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Root directory that contains 'data/frames/5_forstep1and2/...'.",
    )
    parser.add_argument(
        "--splits_csv",
        type=str,
        required=True,
        help="CSV file: frames_5_forstep1and2_folds.csv.",
    )
    parser.add_argument(
        "--labels_json",
        type=str,
        required=False,
        help="track1-qv_pipe_train.json (optional here, labels_str from CSV is enough).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory where checkpoints and history will be saved.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold used for validation (0-4). Train = all other folds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (number of videos per batch).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Max learning rate for OneCycleLR.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="Backbone name in timm (e.g., resnet18, tresnet_m, etc.).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size (H=W).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=5,
        help="Maximum number of frames per video.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, run a single batch through the model and exit.",
    )

    args = parser.parse_args()
    return args


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_labels_str(labels_str: str) -> List[int]:
    """
    Parse labels_str from CSV (e.g. '3 12' or '3,12' or '[3, 12]') into a list of ints.
    """
    if labels_str is None or (isinstance(labels_str, float) and np.isnan(labels_str)):
        return []
    s = str(labels_str).strip()
    if not s:
        return []
    s = s.replace("[", "").replace("]", "")
    tokens = [t for t in s.replace(",", " ").split() if t.strip() != ""]
    return [int(t) for t in tokens]


def build_video_table(
    df_frames: pd.DataFrame,
    max_frames_per_video: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    From frames-level CSV, build a video-level structure:
      - video_stem
      - fold
      - labels (list[int])
      - frame_paths (list[str])

    Also infer num_classes from all labels.
    """
    df_valid = df_frames.copy()
    df_valid = df_valid[~df_valid["labels_str"].isna()]
    df_valid = df_valid[~df_valid["fold"].isna()]

    df_valid["fold"] = df_valid["fold"].astype(int)
    df_valid["labels_list"] = df_valid["labels_str"].apply(parse_labels_str)

    all_labels: List[int] = []
    for lbls in df_valid["labels_list"].tolist():
        all_labels.extend(lbls)
    num_classes = max(all_labels) + 1 if len(all_labels) > 0 else 0

    videos: List[Dict[str, Any]] = []

    for video_stem, df_vid in df_valid.groupby("video_stem"):
        labels_list = df_vid["labels_list"].iloc[0]
        fold = int(df_vid["fold"].iloc[0])

        frame_paths = df_vid["frame_path"].tolist()
        if len(frame_paths) > max_frames_per_video:
            idxs = np.linspace(0, len(frame_paths) - 1, max_frames_per_video).astype(int)
            frame_paths = [frame_paths[i] for i in idxs]

        videos.append(
            {
                "video_stem": video_stem,
                "fold": fold,
                "labels": labels_list,
                "frame_paths": frame_paths,
            }
        )

    return videos, num_classes


# ----------------------------
# Dataset
# ----------------------------

class FramewiseVideoDataset(Dataset):
    """
    Dataset niveau vidéo :
      - sample = une vidéo (1 à max_frames images)
      - labels = vecteur multi-hot
    """

    def __init__(
        self,
        videos: List[Dict[str, Any]],
        frames_dir: Path,
        num_classes: int,
        max_frames: int = 5,
        image_size: int = 224,
    ):
        self.videos = videos
        self.frames_dir = frames_dir
        self.num_classes = num_classes
        self.max_frames = max_frames

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.videos)

    def _resolve_frame_path(self, frame_path_str: str) -> Path:
        """
        Chemin dans le CSV = 'data/frames/5_forstep1and2/1002_f00.jpg'
        --> frames_dir = racine 'QV Pipe'
        --> full path = frames_dir / frame_path_str
        """
        p = Path(frame_path_str)
        if p.is_absolute():
            return p
        return self.frames_dir / p  # frames_dir = DRIVE_ROOT

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.videos[idx]
        labels = item["labels"]
        frame_paths = item["frame_paths"][: self.max_frames]

        images = []
        for fp in frame_paths:
            full_path = self._resolve_frame_path(fp)
            with Image.open(full_path).convert("RGB") as img:
                img = self.transform(img)
            images.append(img)

        frames_tensor = torch.stack(images, dim=0)  # (K, C, H, W)

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in labels:
            if 0 <= c < self.num_classes:
                y[c] = 1.0

        return {
            "frames": frames_tensor,
            "labels": y,
            "video_stem": item["video_stem"],
            "fold": item["fold"],
        }


# ----------------------------
# Model
# ----------------------------

class FramewiseBaselineModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=num_classes,
            global_pool="avg",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, K, C, H, W)
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        logits = self.backbone(x)              # (B*K, num_classes)
        logits = logits.view(B, K, self.num_classes)
        video_logits = logits.mean(dim=1)      # (B, num_classes)
        return video_logits


# ----------------------------
# Train / Val loops
# ----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    running_samples = 0
    last_lr = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        frames = batch["frames"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_samples += bs

    avg_loss = running_loss / max(running_samples, 1)
    return avg_loss, last_lr


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            frames = batch["frames"].to(device)
            labels = batch["labels"].to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_samples += bs

            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / max(running_samples, 1)

    if len(all_targets) > 0:
        y_true = np.concatenate(all_targets, axis=0)
        y_scores = np.concatenate(all_probs, axis=0)
        try:
            map_score = average_precision_score(y_true, y_scores, average="macro")
        except Exception:
            map_score = 0.0
    else:
        map_score = 0.0

    return avg_loss, map_score


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    set_seed(42)

    frames_dir = Path(args.frames_dir)
    splits_csv = Path(args.splits_csv)
    models_dir = Path(args.models_dir)
    ensure_dir(models_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading splits CSV from {splits_csv}")
    df_frames = pd.read_csv(splits_csv)

    required_cols = {"frame_path", "video_stem", "labels_str", "fold"}
    missing = required_cols - set(df_frames.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    videos_all, num_classes = build_video_table(
        df_frames, max_frames_per_video=args.max_frames
    )
    print(f"Found {len(videos_all)} videos with labels.")
    print(f"Inferred num_classes = {num_classes}")

    if num_classes == 0:
        raise ValueError("num_classes == 0, check labels_str in CSV.")

    fold_val = args.fold
    videos_train = [v for v in videos_all if v["fold"] != fold_val]
    videos_val = [v for v in videos_all if v["fold"] == fold_val]

    print(f"Fold {fold_val}: train videos = {len(videos_train)}, val videos = {len(videos_val)}")
    if len(videos_train) == 0 or len(videos_val) == 0:
        raise ValueError("Train or val set is empty. Check fold value or CSV content.")

    train_dataset = FramewiseVideoDataset(
        videos=videos_train,
        frames_dir=frames_dir,
        num_classes=num_classes,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )
    val_dataset = FramewiseVideoDataset(
        videos=videos_val,
        frames_dir=frames_dir,
        num_classes=num_classes,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )

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

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print(f"Creating backbone: {args.backbone}")
    model = FramewiseBaselineModel(args.backbone, num_classes=num_classes).to(device)

    # DRY RUN
    if args.dry_run:
        print("Running DRY RUN (one batch through the model)...")
        batch = next(iter(train_loader))
        frames = batch["frames"].to(device)
        labels = batch["labels"].to(device)
        print(f"Batch frames shape: {frames.shape}")
        print(f"Batch labels shape: {labels.shape}")
        with torch.cuda.amp.autocast():
            logits = model(frames)
        print(f"Logits shape: {logits.shape}")
        print("Dry run OK. Exiting.")
        return

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler()

    history = {"train_loss": [], "val_loss": [], "val_map": [], "lr": []}
    best_map = -1.0
    run_dir = models_dir / f"{args.backbone}_fold{args.fold}"
    ensure_dir(run_dir)
    best_ckpt_path = run_dir / "best_model.pth"
    history_path = run_dir / "history.json"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, last_lr = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_loss, val_map = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_mAP={val_map:.4f}, "
            f"lr={last_lr:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)
        history["lr"].append(last_lr)

        if val_map > best_map:
            best_map = val_map
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"New best mAP = {best_map:.4f} → checkpoint saved at {best_ckpt_path}")

        with open(history_path, "w") as f:
            json.dump(history, f)

    print("\nTraining finished.")
    print(f"Best mAP on fold {args.fold} = {best_map:.4f}")
    print(f"Best model saved at: {best_ckpt_path}")
    print(f"History saved at:    {history_path}")


if __name__ == "__main__":
    main()
