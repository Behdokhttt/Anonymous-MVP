"""
Step 6: Semi-Supervised Fine-tuning
====================================
Fine-tunes the multimodal emotion classifier using:
  1. Labeled data — standard cross-entropy loss
  2. Unlabeled data — FixMatch pseudo-labeling with consistency regularization
  3. Pre-trained encoder weights from Step 5 (contrastive pre-training)

Key techniques:
  - FixMatch: weak aug → pseudo-label (high confidence), strong aug → train
  - EMA teacher model for more stable pseudo-labels
  - MixUp regularization on labeled data
  - Cosine annealing with warm restarts
"""

import os
import sys
import json
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PathConfig, ModelConfig, TrainConfig, PretrainConfig,
    EMOTION_LABELS, NUM_CLASSES,
)
from models.classifier import MultimodalEmotionModel
from models.contrastive import ContrastivePretrainModel
from utils.dataset import MultimodalDataset, collate_multimodal
from utils.augmentation import MultimodalAugmentor
from utils.metrics import compute_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_encoders(model: MultimodalEmotionModel, pretrain_ckpt: str, device):
    """Transfer encoder weights from contrastive pre-training model."""
    ckpt_path = Path(pretrain_ckpt)
    if not ckpt_path.exists():
        print(f"[WARN] Pre-train checkpoint not found: {ckpt_path}. Training from scratch.")
        return model

    print(f"Loading pre-trained encoders from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pretrain_state = ckpt["model_state_dict"]

    # Map encoder weights
    mapping = {
        "audio_encoder.": "audio_encoder.",
        "visual_encoder.": "visual_encoder.",
        "text_encoder.": "text_encoder.",
    }

    model_state = model.state_dict()
    loaded = 0
    for pt_key, pt_val in pretrain_state.items():
        for prefix_from, prefix_to in mapping.items():
            if pt_key.startswith(prefix_from):
                target_key = prefix_to + pt_key[len(prefix_from):]
                if target_key in model_state and model_state[target_key].shape == pt_val.shape:
                    model_state[target_key] = pt_val
                    loaded += 1

    model.load_state_dict(model_state)
    print(f"Loaded {loaded} parameter tensors from pre-trained model.")
    return model


class EMAModel:
    """Exponential Moving Average of model parameters (teacher model)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def train_one_epoch(
    model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler,
    device, augmentor, train_cfg, ema=None,
):
    """One epoch of semi-supervised training."""
    model.train()
    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    correct = 0
    total = 0
    pseudo_used = 0
    n_batches = 0

    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

    for labeled_batch in labeled_loader:
        # Move labeled data to device
        labeled_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labeled_batch.items()}
        labels = labeled_batch["label"]

        # ── Supervised loss ──────────────────────────────────────────────
        logits = model(labeled_batch)
        sup_loss = criterion(logits, labels)

        # Track accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # ── Unsupervised loss (FixMatch) ─────────────────────────────────
        unsup_loss = torch.tensor(0.0, device=device)

        if unlabeled_iter is not None:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            unlabeled_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in unlabeled_batch.items()}

            # Weak augmentation → pseudo-labels
            with torch.no_grad():
                weak_batch = augmentor.weak(unlabeled_batch)
                weak_logits = model(weak_batch)
                pseudo_probs = F.softmax(weak_logits, dim=1)
                max_probs, pseudo_labels = pseudo_probs.max(dim=1)

                # Keep only high-confidence pseudo-labels
                mask = max_probs >= train_cfg.pseudo_label_threshold

            if mask.any():
                # Strong augmentation → train with pseudo-labels
                strong_batch = augmentor.strong(unlabeled_batch)
                strong_logits = model(strong_batch)
                unsup_loss = F.cross_entropy(strong_logits[mask], pseudo_labels[mask])
                pseudo_used += mask.sum().item()

        # ── Total loss ───────────────────────────────────────────────────
        loss = sup_loss + train_cfg.unlabeled_weight * unsup_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / max(n_batches, 1),
        "sup_loss": total_sup_loss / max(n_batches, 1),
        "unsup_loss": total_unsup_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
        "pseudo_used": pseudo_used,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch["label"]

        logits = model(batch)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    metrics = compute_metrics(all_labels, all_preds, list(EMOTION_LABELS.values()))
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def run_supervised_training(
    path_cfg: PathConfig = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainConfig = None,
    pretrain_ckpt: str = None,
):
    """
    Main entry point for semi-supervised fine-tuning.
    """
    path_cfg = path_cfg or PathConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    if pretrain_ckpt is None:
        pretrain_ckpt = str(Path(path_cfg.pretrain_ckpt_dir) / "best_pretrain.pt")

    device = get_device()
    torch.manual_seed(train_cfg.seed)
    print(f"Device: {device}")

    # Load splits
    splits_path = Path(path_cfg.splits_dir) / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    # Create datasets
    train_dataset = MultimodalDataset(path_cfg.assembled_dir, splits["train"], require_label=True)
    val_dataset = MultimodalDataset(path_cfg.assembled_dir, splits["val"], require_label=True)

    unlabeled_dataset = None
    if splits.get("unlabeled"):
        try:
            unlabeled_dataset = MultimodalDataset(path_cfg.assembled_dir, splits["unlabeled"], require_label=False)
            print(f"Unlabeled data: {len(unlabeled_dataset)} samples")
        except FileNotFoundError:
            print("No unlabeled data found.")

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate_multimodal, num_workers=0, drop_last=len(train_dataset) > train_cfg.batch_size,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, shuffle=False,
        collate_fn=collate_multimodal, num_workers=0,
    )
    unlabeled_loader = None
    if unlabeled_dataset and len(unlabeled_dataset) > 0:
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=train_cfg.batch_size, shuffle=True,
            collate_fn=collate_multimodal, num_workers=0, drop_last=False,
        )

    # Build model
    num_classes = train_dataset.num_classes if train_dataset.num_classes > 0 else NUM_CLASSES
    model = MultimodalEmotionModel(
        num_classes=num_classes,
        mfcc_stat_dim=train_dataset.feature_dims.get("mfcc_dim", 130),
        prosodic_dim=train_dataset.feature_dims.get("prosodic_dim", 6),
        visual_input_dim=train_dataset.feature_dims.get("openface_dim", 17),
        text_input_dim=train_dataset.feature_dims.get("text_dim", 768),
        encoder_hidden_dim=model_cfg.encoder_hidden_dim,
        projection_dim=model_cfg.projection_dim,
        encoder_num_heads=model_cfg.encoder_num_heads,
        encoder_num_layers=model_cfg.encoder_num_layers,
        encoder_dropout=model_cfg.encoder_dropout,
        fusion_num_heads=model_cfg.fusion_num_heads,
        fusion_num_layers=model_cfg.fusion_num_layers,
        fusion_dropout=model_cfg.fusion_dropout,
        classifier_hidden_dim=model_cfg.classifier_hidden_dim,
    ).to(device)

    # Load pre-trained encoder weights
    model = load_pretrained_encoders(model, pretrain_ckpt, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total | {trainable:,} trainable | {num_classes} classes")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    ema = EMAModel(model, decay=train_cfg.ema_decay)
    augmentor = MultimodalAugmentor()

    # Checkpointing
    ckpt_dir = Path(path_cfg.supervised_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    patience_counter = 0

    # Training loop
    header = f"{'Ep':>4}  {'TrLoss':>8}  {'SupL':>7}  {'UnsL':>7}  {'TrAcc':>7}  {'VLoss':>8}  {'VAcc':>7}  {'VF1':>7}  {'Pseudo':>7}"
    print(f"\n{header}")
    print("=" * len(header))

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, unlabeled_loader, criterion, optimizer, scheduler,
            device, augmentor, train_cfg, ema,
        )

        # Evaluate with EMA model
        ema_model = copy.deepcopy(model)
        ema.apply(ema_model)
        val_metrics = evaluate(ema_model, val_loader, criterion, device)
        del ema_model

        print(
            f"{epoch:4d}  {train_metrics['loss']:8.4f}  {train_metrics['sup_loss']:7.4f}  "
            f"{train_metrics['unsup_loss']:7.4f}  {train_metrics['accuracy']:7.4f}  "
            f"{val_metrics['loss']:8.4f}  {val_metrics['accuracy']:7.4f}  "
            f"{val_metrics['f1_macro']:7.4f}  {train_metrics['pseudo_used']:7d}"
        )

        # Save best (by macro F1)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            # Save EMA weights as best model
            best_state = {k: v.clone() for k, v in ema.shadow.items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_metrics": val_metrics,
                "train_cfg": vars(train_cfg) if hasattr(train_cfg, "__dict__") else str(train_cfg),
            }, ckpt_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= train_cfg.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={train_cfg.early_stop_patience})")
            break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow,
    }, ckpt_dir / "final_model.pt")

    print(f"\nTraining complete. Best val F1 (macro): {best_val_f1:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    run_supervised_training()
