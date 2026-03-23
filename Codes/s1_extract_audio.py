"""
Step 1: Audio Feature Extraction
=================================
Extracts two types of audio features from each video:
  1. MFCC statistics (130-dim) — captures vocal quality / spectral envelope
  2. Prosodic features (6-dim) — f0 stats, energy stats, speaking rate

Each video produces a single .pt file in the audio feature directory.
"""
import warnings
warnings.filterwarnings("ignore", message=".*librosa.core.audio.__audioread_load.*", category=FutureWarning)

import os
import sys
import warnings
import numpy as np
import torch
import librosa
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy import stats

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PathConfig, AudioConfig, VIDEO_EXTENSIONS

warnings.filterwarnings("ignore", category=UserWarning)


def build_output_stem(video_path: Path, input_dir: Path) -> str:
    """Build a unique, flattened output stem from the video's relative path."""
    relative_path = video_path.relative_to(input_dir).with_suffix("")
    return "__".join(relative_path.parts)


class AudioFeatureExtractor:
    """Extracts MFCC statistics and prosodic features."""

    def __init__(self, audio_cfg: AudioConfig = None, device: str = None):
        self.cfg = audio_cfg or AudioConfig()

    def load_audio_from_video(self, video_path: str) -> np.ndarray:
        """Extract audio waveform from video file."""
        try:
            audio, sr = librosa.load(video_path, sr=self.cfg.sample_rate, mono=True)
        except Exception:
            # Fallback: use moviepy to extract audio first
            from moviepy.editor import VideoFileClip
            import tempfile
            clip = VideoFileClip(str(video_path))
            tmp_wav = tempfile.mktemp(suffix=".wav")
            clip.audio.write_audiofile(tmp_wav, fps=self.cfg.sample_rate, logger=None)
            clip.close()
            audio, sr = librosa.load(tmp_wav, sr=self.cfg.sample_rate, mono=True)
            os.remove(tmp_wav)

        audio = librosa.util.normalize(audio)
        return audio

    def extract_mfcc_stats(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC summary statistics (130-dim)."""
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.cfg.sample_rate, n_mfcc=self.cfg.n_mfcc,
            hop_length=self.cfg.hop_length,
        )
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)

        parts = []
        for arr in [mfccs, delta, delta2]:
            parts.append(np.mean(arr, axis=1))
            parts.append(np.std(arr, axis=1))
        # Add skew and kurtosis for static MFCCs
        parts.append(stats.skew(mfccs, axis=1))
        parts.append(stats.kurtosis(mfccs, axis=1))
        # Add min/max for static MFCCs
        parts.append(np.min(mfccs, axis=1))
        parts.append(np.max(mfccs, axis=1))

        return np.concatenate(parts).astype(np.float32)  # (130,)

    def extract_prosodic(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract prosodic features (6-dim):
        [f0_mean, f0_std, energy_mean, energy_std, speaking_rate, voicing_ratio]
        """
        # F0 via pyin
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.cfg.sample_rate,
            hop_length=self.cfg.hop_length,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        f0_mean = float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_std = float(np.nanstd(f0_voiced)) if len(f0_voiced) > 0 else 0.0

        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.cfg.hop_length)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # Speaking rate (approximate via onset detection)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.cfg.sample_rate, hop_length=self.cfg.hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.cfg.sample_rate, hop_length=self.cfg.hop_length)
        duration = len(audio) / self.cfg.sample_rate
        speaking_rate = len(onsets) / max(duration, 0.1)

        # Voicing ratio
        voicing_ratio = float(np.sum(voiced_flag)) / max(len(voiced_flag), 1) if voiced_flag is not None else 0.0

        return np.array([f0_mean, f0_std, energy_mean, energy_std, speaking_rate, voicing_ratio], dtype=np.float32)

    def extract_all(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extract all audio features from a video file."""
        audio = self.load_audio_from_video(video_path)

        mfcc_stats = self.extract_mfcc_stats(audio)
        prosodic = self.extract_prosodic(audio)

        return {
            "mfcc_stats": mfcc_stats,           # (130,)
            "prosodic": prosodic,               # (6,)
        }


def run_audio_extraction(
    path_cfg: PathConfig = None,
    audio_cfg: AudioConfig = None,
    device: str = None,
):
    """
    Main entry point: extract audio features for all videos in input_video_dir.
    Saves one .pt file per video in audio_feature_dir.
    """
    path_cfg = path_cfg or PathConfig()
    audio_cfg = audio_cfg or AudioConfig()

    input_dir = Path(path_cfg.input_video_dir)
    output_dir = Path(path_cfg.audio_feature_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in sorted(input_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Extracting audio features...")
    extractor = AudioFeatureExtractor(audio_cfg, device=device)

    success, fail = 0, 0
    for vf in video_files:
        stem = build_output_stem(vf, input_dir)
        out_path = output_dir / f"{stem}.pt"
        if out_path.exists():
            print(f"  [SKIP] {stem} (already exists)")
            success += 1
            continue

        try:
            features = extractor.extract_all(str(vf))
            torch.save(features, out_path)
            print(f"  [OK] {stem}  mfcc={features['mfcc_stats'].shape}  prosodic={features['prosodic'].shape}")
            print(f"\nSuccess: {success} | Failed: {fail}")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            print(f"\nSuccess: {success} | Failed: {fail}")
            fail += 1

    print(f"\nDone. Success: {success} | Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    run_audio_extraction()
