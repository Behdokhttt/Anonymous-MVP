"""
Step 4: Assemble Dataset
========================
Combines per-video features from Steps 1-3 into unified .pt files.
Handles labeled vs unlabeled data, computes normalization statistics,
and creates train/val/test splits.

Output per video:
  mfcc_stats     (130,)      — MFCC summary statistics
  prosodic       (6,)        — prosodic features
  openface       (T_v, 17)   — OpenFace AU intensity features
  openface_length int
  text_emb       (768,)      — DeBERTa mean-pooled embedding
  label          int or None — emotion class (None if unlabeled)
  video_id       str
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PathConfig, AudioConfig, VisualConfig, TextConfig, TrainConfig


def load_labels(labels_csv: str) -> Dict[str, int]:
    """Load video_id → label mapping from CSV. Supports header row."""
    labels = {}
    path = Path(labels_csv)
    if not path.exists():
        print(f"[WARN] Labels file not found: {labels_csv}")
        return labels

    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            video_id = parts[0].strip()
            try:
                label = int(parts[1].strip())
                labels[video_id] = label
            except ValueError:
                if i == 0:
                    continue  # skip header
                print(f"  [WARN] Could not parse label: {line}")
    return labels


class DatasetAssembler:
    """Combines all modality features into unified samples."""

    def __init__(
        self,
        path_cfg: PathConfig = None,
        audio_cfg: AudioConfig = None,
        visual_cfg: VisualConfig = None,
        text_cfg: TextConfig = None,
    ):
        self.path_cfg = path_cfg or PathConfig()
        self.audio_cfg = audio_cfg or AudioConfig()
        self.visual_cfg = visual_cfg or VisualConfig()
        self.text_cfg = text_cfg or TextConfig()

    def get_all_video_ids(self) -> List[str]:
        """Discover all video IDs that have at least one modality extracted."""
        ids = set()
        for d in [self.path_cfg.audio_feature_dir, self.path_cfg.visual_feature_dir, self.path_cfg.text_feature_dir]:
            p = Path(d)
            if p.exists():
                for f in p.glob("*.pt"):
                    ids.add(f.stem)
        return sorted(ids)

    def load_audio(self, video_id: str) -> Optional[dict]:
        pt = Path(self.path_cfg.audio_feature_dir) / f"{video_id}.pt"
        if not pt.exists():
            return None
        return torch.load(pt, weights_only=False)

    def load_visual(self, video_id: str) -> Optional[dict]:
        pt = Path(self.path_cfg.visual_feature_dir) / f"{video_id}.pt"
        if not pt.exists():
            return None
        return torch.load(pt, weights_only=False)

    def load_text(self, video_id: str) -> Optional[dict]:
        pt = Path(self.path_cfg.text_feature_dir) / f"{video_id}.pt"
        if not pt.exists():
            return None
        return torch.load(pt, weights_only=False)

    def assemble_sample(self, video_id: str, label: Optional[int] = None) -> Optional[dict]:
        """Load and combine all modalities for one video."""
        audio = self.load_audio(video_id)
        visual = self.load_visual(video_id)
        text = self.load_text(video_id)

        if audio is None or visual is None or text is None:
            missing = []
            if audio is None:
                missing.append("audio")
            if visual is None:
                missing.append("visual")
            if text is None:
                missing.append("text")
            print(f"  [SKIP] {video_id} — missing: {', '.join(missing)}")
            return None

        sample = {
            "video_id": video_id,
            "mfcc_stats": torch.tensor(audio["mfcc_stats"], dtype=torch.float32),
            "prosodic": torch.tensor(audio["prosodic"], dtype=torch.float32),
            "openface": torch.tensor(visual["openface"], dtype=torch.float32),
            "openface_length": visual["openface_length"],
            "text_emb": torch.tensor(text["text_emb"], dtype=torch.float32),
        }

        if label is not None:
            sample["label"] = torch.tensor(label, dtype=torch.long)
        else:
            sample["label"] = None

        return sample

    def compute_normalization_stats(self, samples: List[dict]) -> dict:
        """Compute mean/std for fixed-length features across all samples."""
        mfcc_all = torch.stack([s["mfcc_stats"] for s in samples])
        prosodic_all = torch.stack([s["prosodic"] for s in samples])
        text_all = torch.stack([s["text_emb"] for s in samples])

        stats = {
            "mfcc_mean": mfcc_all.mean(dim=0),
            "mfcc_std": mfcc_all.std(dim=0).clamp(min=1e-8),
            "prosodic_mean": prosodic_all.mean(dim=0),
            "prosodic_std": prosodic_all.std(dim=0).clamp(min=1e-8),
            "text_mean": text_all.mean(dim=0),
            "text_std": text_all.std(dim=0).clamp(min=1e-8),
        }

        # OpenFace: collect all frames
        of_frames = torch.cat([s["openface"] for s in samples], dim=0)
        stats["openface_mean"] = of_frames.mean(dim=0)
        stats["openface_std"] = of_frames.std(dim=0).clamp(min=1e-8)

        return stats

    def normalize_sample(self, sample: dict, stats: dict) -> dict:
        """Apply z-score normalization to a sample."""
        sample["mfcc_stats"] = (sample["mfcc_stats"] - stats["mfcc_mean"]) / stats["mfcc_std"]
        sample["prosodic"] = (sample["prosodic"] - stats["prosodic_mean"]) / stats["prosodic_std"]
        sample["text_emb"] = (sample["text_emb"] - stats["text_mean"]) / stats["text_std"]
        sample["openface"] = (sample["openface"] - stats["openface_mean"]) / stats["openface_std"]
        return sample


def create_splits(
    labeled_ids: List[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Create stratified-like train/val/test splits."""
    rng = np.random.RandomState(seed)
    ids = labeled_ids.copy()
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def run_assembly(
    path_cfg: PathConfig = None,
    train_cfg: TrainConfig = None,
):
    """
    Main entry point: assemble, normalize, split, and save.
    """
    path_cfg = path_cfg or PathConfig()
    train_cfg = train_cfg or TrainConfig()

    out_dir = Path(path_cfg.assembled_dir)
    splits_dir = Path(path_cfg.splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = load_labels(path_cfg.labels_csv)
    print(f"Loaded {len(labels)} labels")

    assembler = DatasetAssembler(path_cfg)
    all_video_ids = assembler.get_all_video_ids()
    print(f"Found {len(all_video_ids)} videos with extracted features")

    # Assemble all samples
    all_samples = []
    labeled_ids = []
    unlabeled_ids = []

    for vid in all_video_ids:
        label = labels.get(vid, None)
        sample = assembler.assemble_sample(vid, label)
        if sample is None:
            continue
        all_samples.append(sample)
        if label is not None:
            labeled_ids.append(vid)
        else:
            unlabeled_ids.append(vid)

    print(f"\nAssembled: {len(all_samples)} total | {len(labeled_ids)} labeled | {len(unlabeled_ids)} unlabeled")

    if not all_samples:
        print("No samples assembled. Check feature extraction outputs.")
        return

    # Compute normalization stats from ALL data
    print("Computing normalization statistics...")
    norm_stats = assembler.compute_normalization_stats(all_samples)
    torch.save(norm_stats, out_dir / "norm_stats.pt")

    # Normalize and save each sample
    print("Normalizing and saving samples...")
    for sample in all_samples:
        sample = assembler.normalize_sample(sample, norm_stats)
        torch.save(sample, out_dir / f"{sample['video_id']}.pt")

    # Create splits for labeled data
    if labeled_ids:
        train_ids, val_ids, test_ids = create_splits(
            labeled_ids,
            val_ratio=train_cfg.val_ratio,
            test_ratio=train_cfg.test_ratio,
            seed=train_cfg.seed,
        )

        # Label distribution
        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            dist = Counter(labels[vid] for vid in ids)
            print(f"  {name}: {len(ids)} samples | distribution: {dict(sorted(dist.items()))}")

        splits = {
            "train": [f"{vid}.pt" for vid in train_ids],
            "val": [f"{vid}.pt" for vid in val_ids],
            "test": [f"{vid}.pt" for vid in test_ids],
            "unlabeled": [f"{vid}.pt" for vid in unlabeled_ids],
            "all": [f"{s['video_id']}.pt" for s in all_samples],
        }

        with open(splits_dir / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)
        print(f"\nSplits saved to {splits_dir / 'splits.json'}")

    print(f"Assembled dataset saved to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    run_assembly()
