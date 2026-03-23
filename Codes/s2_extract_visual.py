"""
Step 2: Visual Feature Extraction (OpenFace)
=============================================
Runs OpenFace FeatureExtraction on each video to extract per-frame
Action Unit (AU) intensities based on the Facial Action Coding System (FACS).
  - 17 AU intensities (AU01_r .. AU45_r)
Total: 17 features per frame, saved as (T, 17) tensor.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PathConfig, VisualConfig, VIDEO_EXTENSIONS


def build_output_stem(video_path: Path, input_dir: Path) -> str:
    """Build a unique, flattened output stem from the video's relative path."""
    relative_path = video_path.relative_to(input_dir).with_suffix("")
    return "__".join(relative_path.parts)


class OpenFaceExtractor:
    """Extract visual features using OpenFace 2.0 binary."""

    def __init__(self, openface_bin: str, visual_cfg: VisualConfig = None):
        self.bin_path = Path(openface_bin)
        self.cfg = visual_cfg or VisualConfig()
        self.haar_path = self._resolve_haar_path()

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"OpenFace binary not found: {self.bin_path}\n"
                "Set OPENFACE_FEATURE_EXTRACTION env var or install OpenFace."
            )

    def _resolve_haar_path(self) -> Optional[Path]:
        env_path = os.getenv("OPENFACE_HAAR")
        if env_path:
            path = Path(env_path)
            return path if path.exists() else None

        default_path = self.bin_path.parent / "model" / "haarcascade_frontalface_alt2.xml"
        return default_path if default_path.exists() else None

    def run_on_video(self, video_path: Path, raw_out_dir: Path, output_stem: str) -> Path:
        """Run FeatureExtraction and return path to output CSV."""
        out_dir = raw_out_dir / output_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.bin_path),
            "-f", str(video_path),
            "-out_dir", str(out_dir),
            "-of", output_stem,
            "-aus",
        ]

        if self.haar_path:
            cmd.extend(["-haar", str(self.haar_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"OpenFace failed on {output_stem}: {result.stderr[:500]}")

        csv_path = out_dir / f"{output_stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV not produced: {csv_path}")
        return csv_path

    def parse_csv(self, csv_path: Path) -> Optional[np.ndarray]:
        """Parse OpenFace CSV and return (T, 17) AU intensity array."""
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        missing = [c for c in self.cfg.all_cols if c not in df.columns]
        if missing:
            print(f"  [WARN] Missing columns: {missing}")
            return None

        features = df[self.cfg.all_cols].values.astype(np.float32)

        # Filter low-confidence frames
        if "confidence" in df.columns:
            mask = df["confidence"].values >= self.cfg.confidence_threshold
            features = features[mask]

        if len(features) == 0:
            return None

        return features


def run_visual_extraction(
    path_cfg: PathConfig = None,
    visual_cfg: VisualConfig = None,
):
    """
    Main entry point: extract AU intensity features for all videos using OpenFace.
    """
    path_cfg = path_cfg or PathConfig()
    visual_cfg = visual_cfg or VisualConfig()

    input_dir = Path(path_cfg.input_video_dir)
    output_dir = Path(path_cfg.visual_feature_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in sorted(input_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    raw_out_dir = output_dir / "openface_raw"
    extractor = OpenFaceExtractor(path_cfg.openface_bin, visual_cfg)
    print(f"Using OpenFace: {path_cfg.openface_bin}")
    print(f"Found {len(video_files)} videos. Extracting AU features...")

    success, fail = 0, 0
    for vf in video_files:
        stem = build_output_stem(vf, input_dir)
        out_path = output_dir / f"{stem}.pt"
        if out_path.exists():
            print(f"  [SKIP] {stem} (already exists)")
            success += 1
            continue

        try:
            csv_path = extractor.run_on_video(vf, raw_out_dir, stem)
            features = extractor.parse_csv(csv_path)

            if features is None:
                print(f"  [FAIL] {stem}: no face detected or low confidence")
                fail += 1
                continue

            data = {
                "openface": features.astype(np.float32),    # (T, 17)
                "openface_length": features.shape[0],
            }
            torch.save(data, out_path)
            print(f"  [OK] {stem}  frames={features.shape[0]}  dim={features.shape[1]}")
            success += 1
            print(f"\nDone. Success: {success} | Failed: {fail}")
        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            fail += 1

    print(f"\nDone. Success: {success} | Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    run_visual_extraction()
