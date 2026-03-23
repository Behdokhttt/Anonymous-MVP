"""
Step 3: Text Feature Extraction
================================
For each video:
  1. Transcribe speech using OpenAI Whisper
  2. Extract DeBERTa-v3 token embeddings → mean-pooled (768-dim)
  3. Extract sentiment & linguistic features as auxiliary signals

Saves one .pt file per video in text_feature_dir.
"""

import os
import sys
import warnings
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PathConfig, TextConfig, VIDEO_EXTENSIONS

warnings.filterwarnings("ignore", category=UserWarning)


class TextFeatureExtractor:
    """Transcribes speech and extracts text embeddings."""

    def __init__(self, text_cfg: TextConfig = None, device: str = None):
        self.cfg = text_cfg or TextConfig()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.whisper_model = None
        self.tokenizer = None
        self.text_model = None

    def _load_whisper(self):
        if self.whisper_model is not None:
            return
        import whisper
        print(f"Loading Whisper ({self.cfg.whisper_model_size}) ...")
        self.whisper_model = whisper.load_model(self.cfg.whisper_model_size, device=str(self.device))
        print("Whisper loaded.")

    def _load_text_model(self):
        if self.text_model is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading text model: {self.cfg.text_model_name} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model_name, use_fast=False)
            self.text_model = AutoModel.from_pretrained(self.cfg.text_model_name).to(self.device)
        except Exception:
            # Fallback to BERT
            print("DeBERTa failed, falling back to bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.text_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.text_model.eval()
        print("Text model loaded.")

    def transcribe(self, video_path: str) -> str:
        """Transcribe speech from video using Whisper."""
        self._load_whisper()
        result = self.whisper_model.transcribe(
            str(video_path),
            language="en",
            fp16=(self.device.type == "cuda"),
        )
        return result.get("text", "").strip()

    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract mean-pooled DeBERTa/BERT embedding (768-dim)."""
        self._load_text_model()

        if not text.strip():
            return np.zeros(self.cfg.text_embed_dim, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.cfg.max_token_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Mean pool over all tokens (excluding padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            token_embs = outputs.last_hidden_state
            pooled = (token_embs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            embedding = pooled.squeeze(0).cpu().numpy()

        return embedding.astype(np.float32)

    def extract_sentiment_features(self, text: str) -> np.ndarray:
        """
        Extract basic sentiment/linguistic features as auxiliary signal.
        Returns 5-dim: [polarity, subjectivity, word_count, exclamation_ratio, question_ratio]
        """
        if not text.strip():
            return np.zeros(5, dtype=np.float32)

        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except ImportError:
            polarity = 0.0
            subjectivity = 0.0

        words = text.split()
        word_count = min(len(words) / 50.0, 1.0)  # normalized
        char_count = max(len(text), 1)
        exclamation_ratio = text.count("!") / char_count
        question_ratio = text.count("?") / char_count

        return np.array([polarity, subjectivity, word_count, exclamation_ratio, question_ratio], dtype=np.float32)

    def extract_all(self, video_path: str) -> Dict[str, object]:
        """Extract all text features from a video."""
        transcript = self.transcribe(video_path)
        text_emb = self.extract_text_embedding(transcript)
        sentiment = self.extract_sentiment_features(transcript)

        return {
            "transcript": transcript,
            "text_emb": text_emb,              # (768,)
            "sentiment_features": sentiment,   # (5,)
        }


def run_text_extraction(
    path_cfg: PathConfig = None,
    text_cfg: TextConfig = None,
    device: str = None,
):
    """
    Main entry point: extract text features for all videos.
    """
    path_cfg = path_cfg or PathConfig()
    text_cfg = text_cfg or TextConfig()

    input_dir = Path(path_cfg.input_video_dir)
    output_dir = Path(path_cfg.text_feature_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Extracting text features...")
    extractor = TextFeatureExtractor(text_cfg, device=device)

    success, fail = 0, 0
    for vf in video_files:
        stem = vf.stem
        out_path = output_dir / f"{stem}.pt"
        if out_path.exists():
            print(f"  [SKIP] {stem} (already exists)")
            success += 1
            continue

        try:
            features = extractor.extract_all(str(vf))
            # Save transcript as text too
            txt_path = output_dir / f"{stem}_transcript.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(features["transcript"])

            torch.save({
                "text_emb": features["text_emb"],
                "sentiment_features": features["sentiment_features"],
                "transcript": features["transcript"],
            }, out_path)

            print(f"  [OK] {stem}  transcript_len={len(features['transcript'])}  emb={features['text_emb'].shape}")
            success += 1

        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            fail += 1

    print(f"\nDone. Success: {success} | Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    run_text_extraction()
