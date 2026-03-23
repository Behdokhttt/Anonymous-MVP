"""
Step 7: Evaluation & Inference
===============================
Loads the best trained model and evaluates on the test set.
Produces:
  - Per-class precision, recall, F1
  - Confusion matrix (saved as image)
  - Inference function for new videos
  - t-SNE visualization of learned embeddings
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PathConfig, ModelConfig, TrainConfig, EMOTION_LABELS, NUM_CLASSES, VIDEO_EXTENSIONS
from models.classifier import MultimodalEmotionModel
from utils.dataset import MultimodalDataset, collate_multimodal
from utils.metrics import compute_metrics, print_classification_report, plot_confusion_matrix


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, model_cfg: ModelConfig, feature_dims: dict, device) -> MultimodalEmotionModel:
    """Load trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes = NUM_CLASSES
    model = MultimodalEmotionModel(
        num_classes=num_classes,
        mfcc_stat_dim=feature_dims.get("mfcc_dim", 130),
        prosodic_dim=feature_dims.get("prosodic_dim", 6),
        visual_input_dim=feature_dims.get("openface_dim", 17),
        text_input_dim=feature_dims.get("text_dim", 768),
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

    state_dict = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded model from epoch {epoch}")
    return model


@torch.no_grad()
def run_evaluation(
    path_cfg: PathConfig = None,
    model_cfg: ModelConfig = None,
    ckpt_path: str = None,
):
    """
    Evaluate the best model on the test set.
    """
    path_cfg = path_cfg or PathConfig()
    model_cfg = model_cfg or ModelConfig()

    if ckpt_path is None:
        ckpt_path = str(Path(path_cfg.supervised_ckpt_dir) / "best_model.pt")

    device = get_device()
    print(f"Device: {device}")

    # Load splits
    splits_path = Path(path_cfg.splits_dir) / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    # Test dataset
    test_dataset = MultimodalDataset(path_cfg.assembled_dir, splits["test"], require_label=True)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        collate_fn=collate_multimodal, num_workers=0,
    )

    # Load model
    model = load_model(ckpt_path, model_cfg, test_dataset.feature_dims, device)

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = {"audio": [], "visual": [], "text": [], "fused": []}
    all_video_ids = []

    for batch in test_loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Get embeddings and logits
        outputs = model.get_embeddings(batch_dev)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
        all_probs.extend(probs.cpu().numpy())

        for key in ["audio_emb", "visual_emb", "text_emb", "fused"]:
            short_key = key.replace("_emb", "")
            all_embeddings[short_key].extend(outputs[key].cpu().numpy())

        if "video_id" in batch:
            all_video_ids.extend(batch["video_id"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    max_idx = max(int(all_labels.max()), int(all_preds.max()), NUM_CLASSES - 1)
    label_names = [EMOTION_LABELS.get(i, f"class_{i}") for i in range(max_idx + 1)]

    # Print classification report
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print_classification_report(all_labels, all_preds, label_names)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, label_names)
    print(f"\nAccuracy:       {metrics['accuracy']:.4f}")
    print(f"F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):  {metrics['f1_weighted']:.4f}")

    # Save confusion matrix
    results_dir = Path(path_cfg.supervised_ckpt_dir) / "results"
    results_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(
        all_labels, all_preds, label_names,
        save_path=str(results_dir / "confusion_matrix.png"),
    )

    # Save metrics
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Save per-sample predictions
    predictions = []
    for i in range(len(all_preds)):
        pred_entry = {
            "video_id": all_video_ids[i] if i < len(all_video_ids) else f"sample_{i}",
            "true_label": int(all_labels[i]),
            "true_emotion": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_emotion": label_names[int(all_preds[i])],
            "confidence": float(all_probs[i].max()),
            "correct": bool(all_preds[i] == all_labels[i]),
        }
        predictions.append(pred_entry)

    with open(results_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    csv_path = results_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "pred_label"])
        for i in range(len(all_preds)):
            video_id = all_video_ids[i] if i < len(all_video_ids) else f"sample_{i}"
            writer.writerow([video_id, int(all_preds[i])])

    # t-SNE visualization
    try:
        _plot_tsne(all_embeddings["fused"], all_labels, label_names, results_dir / "tsne_fused.png")
        print(f"\nt-SNE plot saved to {results_dir / 'tsne_fused.png'}")
    except Exception as e:
        print(f"[WARN] t-SNE visualization failed: {e}")

    print(f"\nAll results saved to {results_dir}")
    return metrics


def _plot_tsne(embeddings_list, labels, label_names, save_path):
    """Generate t-SNE visualization of fused embeddings."""
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings = np.array(embeddings_list)
    if len(embeddings) < 2:
        print("[WARN] Need at least 2 samples for t-SNE")
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(1, len(embeddings) - 1)))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_names[lbl] if lbl < len(label_names) else f"class_{lbl}"
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name, alpha=0.7, s=40)

    ax.legend()
    ax.set_title("t-SNE of Fused Multimodal Embeddings")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _extract_single_sample(
    video_path: str,
    path_cfg: PathConfig,
    dev: torch.device,
    audio_ext=None,
    text_ext=None,
    of_ext=None,
) -> dict:
    from s1_extract_audio import AudioFeatureExtractor
    from s2_extract_visual import OpenFaceExtractor
    from s3_extract_text import TextFeatureExtractor

    if audio_ext is None:
        audio_ext = AudioFeatureExtractor(device=str(dev))
    if text_ext is None:
        text_ext = TextFeatureExtractor(device=str(dev))

    audio_feat = audio_ext.extract_all(video_path)
    text_feat = text_ext.extract_all(video_path)

    try:
        if of_ext is None:
            from s2_extract_visual import OpenFaceExtractor
            of_ext = OpenFaceExtractor(path_cfg.openface_bin)
        import tempfile
        raw_dir = Path(tempfile.mkdtemp())
        csv_path = of_ext.run_on_video(Path(video_path), raw_dir)
        visual_feat = of_ext.parse_csv(csv_path)
    except Exception as e:
        print(f"[WARN] OpenFace extraction failed: {e}")
        visual_feat = None

    if visual_feat is None:
        visual_feat = np.zeros((1, 17), dtype=np.float32)

    sample = {
        "mfcc_stats": torch.tensor(audio_feat["mfcc_stats"], dtype=torch.float32).unsqueeze(0),
        "prosodic": torch.tensor(audio_feat["prosodic"], dtype=torch.float32).unsqueeze(0),
        "openface": torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0),
        "openface_lengths": torch.tensor([visual_feat.shape[0]], dtype=torch.long),
        "text_emb": torch.tensor(text_feat["text_emb"], dtype=torch.float32).unsqueeze(0),
        "transcript": text_feat.get("transcript", ""),
    }

    norm_path = Path(path_cfg.assembled_dir) / "norm_stats.pt"
    if norm_path.exists():
        stats = torch.load(norm_path, weights_only=False)
        sample["mfcc_stats"] = (sample["mfcc_stats"] - stats["mfcc_mean"]) / stats["mfcc_std"]
        sample["prosodic"] = (sample["prosodic"] - stats["prosodic_mean"]) / stats["prosodic_std"]
        sample["text_emb"] = (sample["text_emb"] - stats["text_mean"]) / stats["text_std"]
        sample["openface"] = (sample["openface"] - stats["openface_mean"]) / stats["openface_std"]

    return sample


def predict_single(
    video_path: str,
    path_cfg: PathConfig = None,
    model_cfg: ModelConfig = None,
    ckpt_path: str = None,
    device: str = None,
) -> dict:
    """
    Run full inference pipeline on a single video.
    Extracts features → assembles → normalizes → predicts.
    """
    path_cfg = path_cfg or PathConfig()
    model_cfg = model_cfg or ModelConfig()

    if ckpt_path is None:
        ckpt_path = str(Path(path_cfg.supervised_ckpt_dir) / "best_model.pt")

    dev = torch.device(device) if device else get_device()

    sample = _extract_single_sample(video_path, path_cfg, dev)

    feature_dims = {"mfcc_dim": 130, "prosodic_dim": 6, "openface_dim": 17, "text_dim": 768}
    model = load_model(ckpt_path, model_cfg, feature_dims, dev)

    sample_dev = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
    with torch.no_grad():
        logits = model(sample_dev)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    result = {
        "predicted_emotion": EMOTION_LABELS.get(pred_idx, f"class_{pred_idx}"),
        "predicted_label": pred_idx,
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {EMOTION_LABELS.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)},
        "transcript": sample.get("transcript", ""),
    }

    results_dir = Path(path_cfg.supervised_ckpt_dir) / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "prediction_single.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "pred_label"])
        writer.writerow([Path(video_path).stem, pred_idx])

    print(f"\nPrediction: {result['predicted_emotion']} ({result['confidence']:.2%})")
    for emo, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {emo:>10s}: {prob:.4f} {bar}")

    return result


def predict_folder(
    folder_path: str,
    path_cfg: PathConfig = None,
    model_cfg: ModelConfig = None,
    ckpt_path: str = None,
    device: str = None,
) -> Path:
    """Predict all videos in a folder and save a CSV with video_id and predicted label."""
    path_cfg = path_cfg or PathConfig()
    model_cfg = model_cfg or ModelConfig()

    if ckpt_path is None:
        ckpt_path = str(Path(path_cfg.supervised_ckpt_dir) / "best_model.pt")

    dev = torch.device(device) if device else get_device()
    feature_dims = {"mfcc_dim": 130, "prosodic_dim": 6, "openface_dim": 17, "text_dim": 768}
    model = load_model(ckpt_path, model_cfg, feature_dims, dev)

    folder = Path(folder_path)
    video_files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        raise FileNotFoundError(f"No videos found in {folder}")

    # Pre-load extractors once for the whole folder
    from s1_extract_audio import AudioFeatureExtractor
    from s2_extract_visual import OpenFaceExtractor
    from s3_extract_text import TextFeatureExtractor

    audio_ext = AudioFeatureExtractor(device=str(dev))
    text_ext = TextFeatureExtractor(device=str(dev))
    try:
        of_ext = OpenFaceExtractor(path_cfg.openface_bin)
    except Exception:
        of_ext = None

    results_dir = Path(path_cfg.supervised_ckpt_dir) / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "predictions_folder.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "pred_label"])
        for video_path in video_files:
            sample = _extract_single_sample(str(video_path), path_cfg, dev, audio_ext, text_ext, of_ext)
            sample_dev = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            with torch.no_grad():
                logits = model(sample_dev)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())
            writer.writerow([video_path.stem, pred_idx])
            print(f"[OK] {video_path.stem}: {EMOTION_LABELS.get(pred_idx, pred_idx)} ({probs[pred_idx]:.2%})")

    print(f"\nFolder predictions saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "predict"], default="evaluate")
    parser.add_argument("--video", type=str, default=None, help="Path to video or folder for prediction")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    if args.mode == "evaluate":
        run_evaluation(ckpt_path=args.ckpt)
    elif args.mode == "predict":
        if args.video is None:
            print("Please provide --video path")
        else:
            video_path = Path(args.video)
            if video_path.is_dir():
                predict_folder(str(video_path), ckpt_path=args.ckpt)
            else:
                predict_single(str(video_path), ckpt_path=args.ckpt)
