"""
Step 5: Self-Supervised Contrastive Pre-training
=================================================
Trains modality encoders on ALL data (labeled + unlabeled) without labels.
Uses cross-modal contrastive learning (NT-Xent) to align audio, visual,
and text representations in a shared embedding space.

Also includes masked modality reconstruction as auxiliary objective.

This step is critical for leveraging the thousands of unlabeled videos.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PathConfig, ModelConfig, PretrainConfig, AudioConfig, VisualConfig, TextConfig
from models.contrastive import ContrastivePretrainModel
from utils.dataset import MultimodalDataset, collate_multimodal
from utils.augmentation import MultimodalAugmentor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_one_epoch(model, loader, optimizer, scheduler, device, augmentor=None):
    """Run one pre-training epoch."""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_recon = 0.0
    n_batches = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Apply augmentation if available (for contrastive diversity)
        if augmentor is not None:
            batch = augmentor.weak(batch)

        optimizer.zero_grad()
        losses = model(batch)
        loss = losses["total"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_contrastive += losses["contrastive_total"].item()
        total_recon += losses["reconstruction_total"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "contrastive": total_contrastive / max(n_batches, 1),
        "reconstruction": total_recon / max(n_batches, 1),
    }


def run_pretraining(
    path_cfg: PathConfig = None,
    model_cfg: ModelConfig = None,
    pretrain_cfg: PretrainConfig = None,
):
    """
    Main entry point for contrastive pre-training.
    Uses ALL assembled data (no labels needed).
    """
    path_cfg = path_cfg or PathConfig()
    model_cfg = model_cfg or ModelConfig()
    pretrain_cfg = pretrain_cfg or PretrainConfig()

    device = get_device()
    print(f"Device: {device}")

    # Load splits to get all data
    splits_path = Path(path_cfg.splits_dir) / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        all_files = splits.get("all", None)
    else:
        all_files = None

    # Create dataset with ALL data (no label requirement)
    dataset = MultimodalDataset(
        data_dir=path_cfg.assembled_dir,
        file_list=all_files,
        require_label=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=pretrain_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=0,
        drop_last=len(dataset) > pretrain_cfg.batch_size,  # Only drop if enough data
    )

    # Build model
    model = ContrastivePretrainModel(
        mfcc_stat_dim=dataset.feature_dims.get("mfcc_dim", 130),
        prosodic_dim=dataset.feature_dims.get("prosodic_dim", 6),
        visual_input_dim=dataset.feature_dims.get("openface_dim", 17),
        text_input_dim=dataset.feature_dims.get("text_dim", 768),
        encoder_hidden_dim=model_cfg.encoder_hidden_dim,
        projection_dim=model_cfg.projection_dim,
        encoder_num_heads=model_cfg.encoder_num_heads,
        encoder_num_layers=model_cfg.encoder_num_layers,
        encoder_dropout=model_cfg.encoder_dropout,
        temperature=pretrain_cfg.temperature,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Contrastive model: {total_params:,} parameters")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pretrain_cfg.lr,
        weight_decay=pretrain_cfg.weight_decay,
    )
    total_steps = pretrain_cfg.epochs * len(loader)
    warmup_steps = pretrain_cfg.warmup_epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    augmentor = MultimodalAugmentor()

    # Checkpoint dir
    ckpt_dir = Path(path_cfg.pretrain_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    # Training loop
    print(f"\n{'Epoch':>5}  {'Loss':>10}  {'Contrastive':>12}  {'Recon':>10}  {'LR':>10}  {'Time':>8}")
    print("=" * 65)

    for epoch in range(1, pretrain_cfg.epochs + 1):
        t0 = time.time()
        metrics = pretrain_one_epoch(model, loader, optimizer, scheduler, device, augmentor)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {metrics['loss']:10.4f}  {metrics['contrastive']:12.4f}  "
              f"{metrics['reconstruction']:10.4f}  {lr:10.6f}  {elapsed:7.1f}s")

        # Save best
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, ckpt_dir / "best_pretrain.pt")

        # Periodic save
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": metrics["loss"],
            }, ckpt_dir / f"pretrain_epoch_{epoch}.pt")

    # Save final
    torch.save({
        "epoch": pretrain_cfg.epochs,
        "model_state_dict": model.state_dict(),
        "loss": metrics["loss"],
    }, ckpt_dir / "final_pretrain.pt")

    print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    run_pretraining()
