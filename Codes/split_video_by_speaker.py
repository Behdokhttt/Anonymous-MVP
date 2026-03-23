"""
Video Splitting by Speaker Turn-Taking
======================================

This script processes a video with two people talking, identifies who is speaking 
at each moment using speaker diarization, and splits the video into sections 
based on turn-taking. Overlapping sections are created when both people talk 
simultaneously to ensure complete capture of each person's speech.
"""

import os
import glob
from pathlib import Path
import tempfile
import json
import numpy as np
from typing import List, Dict, Tuple
import torch
import whisper
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import cv2
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation


class SpeakerDiarizer:
    """
    Performs speaker diarization to identify who is speaking and when.
    """
    
    def __init__(self, num_speakers: int = 2):
        """
        Initialize the speaker diarizer.
        
        Args:
            num_speakers: Expected number of speakers (default: 2)
        """
        self.num_speakers = num_speakers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        print(f"Using device: {self.device}")
        
        # Initialize Whisper for speech activity detection
        self.whisper_model = whisper.load_model("base", device=self.device)
        
        # Initialize pyannote diarization pipeline
        try:
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("PYANNOTE_TOKEN")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if pipeline is None:
                raise RuntimeError("pyannote Pipeline.from_pretrained returned None")
            self.diarization_pipeline = pipeline.to(self.torch_device)
            print("Pyannote diarization pipeline loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pyannote pipeline: {e}")
            print(
                "Pyannote diarization is unavailable (model download/auth failed). "
                "For best audio-only diarization quality, set env var HUGGINGFACE_TOKEN (or HF_TOKEN/PYANNOTE_TOKEN) "
                "to a HuggingFace token and accept the model terms on HuggingFace."
            )
            self.diarization_pipeline = None
    
    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to temporary audio file
        """
        try:
            print(f"Extracting audio from: {video_path}")
            
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # Extract audio using moviepy
            with VideoFileClip(video_path) as clip:
                audio = clip.audio
                if audio is not None:
                    audio.write_audiofile(temp_audio.name, fps=16000, codec='pcm_s16le', 
                                        verbose=False, logger=None)
                    audio.close()

                    try:
                        audio_data, sr = librosa.load(temp_audio.name, sr=16000, mono=True)
                        if audio_data is None or len(audio_data) == 0:
                            raise ValueError("Empty audio")
                        peak = float(np.max(np.abs(audio_data))) if len(audio_data) else 0.0
                        if peak > 0:
                            audio_data = (audio_data / peak) * 0.99
                        sf.write(temp_audio.name, audio_data.astype(np.float32), 16000, subtype="PCM_16")
                    except Exception as e:
                        print(f"Warning: audio normalization failed: {e}")

                    return temp_audio.name
                else:
                    print(f"Warning: No audio found in {video_path}")
                    os.unlink(temp_audio.name)
                    return None
                    
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def detect_speech_activity(self, audio_path: str) -> List[Dict]:
        """
        Detect speech activity using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speech segments with timestamps
        """
        try:
            print("Detecting speech activity with Whisper...")
            
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                audio_data, 
                language="en",
                task="transcribe",
                word_timestamps=True
            )
            
            # Extract speech segments
            speech_segments = []
            for segment in result.get("segments", []):
                speech_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", [])
                })
            
            print(f"Detected {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            print(f"Error in speech activity detection: {e}")
            return []
    
    def perform_diarization(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization to identify speakers.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments with timestamps
        """
        if self.diarization_pipeline is None:
            # Fallback: simple energy-based speaker separation
            print("Using fallback diarization (energy-based speaker alternation)")
            return self._fallback_diarization(audio_path)
        
        try:
            print("Performing speaker diarization...")
            
            # Perform diarization
            diarization = self.diarization_pipeline(
                audio_path,
                min_speakers=self.num_speakers,
                max_speakers=self.num_speakers,
            )
            
            # Convert to list format
            speaker_segments = []
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                    "confidence": 1.0  # Pyannote doesn't provide confidence by default
                })
            
            # Sort by start time
            speaker_segments.sort(key=lambda x: x["start"])
            
            print(f"Diarization complete: {len(speaker_segments)} segments")
            return speaker_segments
            
        except Exception as e:
            print(f"Error in diarization: {e}")
            return self._fallback_diarization(audio_path)
    
    def _fallback_diarization(self, audio_path: str) -> List[Dict]:
        """
        Fallback diarization using energy and pitch characteristics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments (alternating between two speakers)
        """
        print("Using fallback diarization (energy-based speaker alternation)")
        
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate energy and pitch
            frame_length = 2048
            hop_length = 512
            
            # Energy
            energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, hop_length=hop_length)
            pitch = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch.append(pitches[index, t])
            pitch = np.array(pitch)
            
            # Simple voice activity detection
            vad_threshold = np.mean(energy) + 0.5 * np.std(energy)
            speech_frames = energy > vad_threshold
            
            # Create segments (alternating speakers every few seconds)
            segments = []
            current_speaker = "SPEAKER_00"
            segment_duration = 3.0  # seconds
            frame_time = hop_length / sr
            
            i = 0
            while i < len(speech_frames):
                if speech_frames[i]:
                    start_time = i * frame_time
                    end_time = min(start_time + segment_duration, len(audio_data) / sr)
                    
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "speaker": current_speaker,
                        "confidence": 0.7
                    })
                    
                    # Alternate speaker
                    current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                    
                    # Skip ahead
                    i += int(segment_duration / frame_time)
                else:
                    i += 1
            
            return segments
            
        except Exception as e:
            print(f"Error in fallback diarization: {e}")
            return []
    
    def combine_speech_and_speakers(self, speech_segments: List[Dict], 
                                  speaker_segments: List[Dict]) -> List[Dict]:
        """
        Combine speech activity detection with speaker diarization.
        
        Args:
            speech_segments: Speech activity segments
            speaker_segments: Speaker diarization segments
            
        Returns:
            Combined segments with speaker labels
        """
        combined_segments = []
        
        for speech_seg in speech_segments:
            # Find the dominant speaker for this speech segment
            speaker_scores = {}
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(speech_seg["start"], speaker_seg["start"])
                overlap_end = min(speech_seg["end"], speaker_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker = speaker_seg["speaker"]
                    if speaker not in speaker_scores:
                        speaker_scores[speaker] = 0
                    speaker_scores[speaker] += overlap_duration
            
            # Assign the dominant speaker
            if speaker_scores:
                dominant_speaker = max(speaker_scores, key=speaker_scores.get)
                confidence = speaker_scores[dominant_speaker] / (speech_seg["end"] - speech_seg["start"])
            else:
                dominant_speaker = "SPEAKER_00"
                confidence = 0.5
            
            combined_segments.append({
                "start": speech_seg["start"],
                "end": speech_seg["end"],
                "speaker": dominant_speaker,
                "text": speech_seg["text"],
                "confidence": confidence
            })
        
        return combined_segments


class VideoSplitter:
    """
    Splits video based on speaker turn-taking.
    """
    
    def __init__(self, input_dir: str = "data/input_videos", 
                 output_dir: str = "data/output/split_videos"):
        """
        Initialize the video splitter.
        
        Args:
            input_dir: Directory containing input video files
            output_dir: Directory to save split video sections
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize diarizer
        self.diarizer = SpeakerDiarizer()
        
        # Supported video formats
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    def _estimate_active_speaker_side(self, video_path: str, sample_fps: float = 15.0) -> List[Dict]:
        """Estimate per-timestamp mouth-motion activity for left vs right speaker.

        Primary method: face-detection based mouth-region motion (more robust).
        Fallback: fixed left/right ROIs if face detection fails.
        """ 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: could not open video for active-speaker estimation: {video_path}")
            return []

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0

        step = max(1, int(round(fps / max(1.0, sample_fps))))
        prev_gray = None
        activity: List[Dict] = []

        face_cascade = None
        try:
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                face_cascade = None
        except Exception:
            face_cascade = None

        last_faces = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(gray, prev_gray)
                _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                left_score = 0.0
                right_score = 0.0
                used_face_method = False

                if face_cascade is not None:
                    try:
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                        faces = sorted(list(faces), key=lambda r: int(r[2] * r[3]), reverse=True)
                        faces = faces[:2]
                        if len(faces) == 2:
                            last_faces = faces
                        elif last_faces is not None and len(last_faces) == 2:
                            faces = last_faces

                        if faces is not None and len(faces) == 2:
                            # Sort faces by x (left-to-right)
                            faces = sorted(list(faces), key=lambda r: r[0])
                            # Mouth/lower-face ROI inside each face box
                            def mouth_roi(rect):
                                x, y, fw, fh = [int(v) for v in rect]
                                mx1 = x + int(0.15 * fw)
                                mx2 = x + int(0.85 * fw)
                                my1 = y + int(0.55 * fh)
                                my2 = y + int(0.95 * fh)
                                mx1 = max(0, min(mx1, w - 1))
                                mx2 = max(0, min(mx2, w))
                                my1 = max(0, min(my1, h - 1))
                                my2 = max(0, min(my2, h))
                                return mx1, my1, mx2, my2

                            lx1, ly1, lx2, ly2 = mouth_roi(faces[0])
                            rx1, ry1, rx2, ry2 = mouth_roi(faces[1])

                            left_roi = diff[ly1:ly2, lx1:lx2]
                            right_roi = diff[ry1:ry2, rx1:rx2]

                            left_score = float(np.mean(left_roi)) if left_roi.size else 0.0
                            right_score = float(np.mean(right_roi)) if right_roi.size else 0.0
                            used_face_method = True
                    except Exception:
                        used_face_method = False

                if not used_face_method:
                    # Fallback: fixed split-screen ROIs
                    y1 = int(0.45 * h)
                    y2 = int(0.78 * h)

                    lx1 = int(0.05 * w)
                    lx2 = int(0.48 * w)
                    rx1 = int(0.52 * w)
                    rx2 = int(0.95 * w)

                    left_roi = diff[y1:y2, lx1:lx2]
                    right_roi = diff[y1:y2, rx1:rx2]

                    left_score = float(np.mean(left_roi)) if left_roi.size else 0.0
                    right_score = float(np.mean(right_roi)) if right_roi.size else 0.0

                t = frame_idx / fps
                activity.append({"t": t, "left": left_score, "right": right_score})

            prev_gray = gray
            frame_idx += 1

        cap.release()
        return activity

    def _relabel_segments_with_active_speaker(
        self,
        video_path: str,
        segments: List[Dict],
        sample_fps: float = 15.0,
    ) -> List[Dict]:
        """Relabel diarization speaker IDs to LEFT/RIGHT using video mouth-motion activity."""
        if not segments:
            return segments

        activity = self._estimate_active_speaker_side(video_path, sample_fps=sample_fps)
        if not activity:
            return segments

        times = np.array([a["t"] for a in activity], dtype=np.float32)
        left_vals = np.array([a["left"] for a in activity], dtype=np.float32)
        right_vals = np.array([a["right"] for a in activity], dtype=np.float32)

        speaker_to_votes: Dict[str, Dict[str, float]] = {}

        for seg in segments:
            spk = seg.get("speaker")
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if spk is None or end <= start:
                continue

            mask = (times >= start) & (times <= end)
            if not np.any(mask):
                continue

            l = float(np.mean(left_vals[mask]))
            r = float(np.mean(right_vals[mask]))
            side = "LEFT" if l >= r else "RIGHT"
            weight = end - start

            if spk not in speaker_to_votes:
                speaker_to_votes[spk] = {"LEFT": 0.0, "RIGHT": 0.0}
            speaker_to_votes[spk][side] += weight

        if not speaker_to_votes:
            return segments

        speaker_to_side: Dict[str, str] = {}
        used_sides: set = set()
        for spk, votes in speaker_to_votes.items():
            side = "LEFT" if votes["LEFT"] >= votes["RIGHT"] else "RIGHT"
            speaker_to_side[spk] = side
            used_sides.add(side)

        # If both diarization speakers collapse to the same side, do not relabel (heuristic likely failed)
        if len(used_sides) == 1 and len(speaker_to_side) >= 2:
            print("Warning: active-speaker heuristic mapped multiple speakers to one side; keeping original labels")
            return segments

        out: List[Dict] = []
        for seg in segments:
            spk = seg.get("speaker")
            if spk in speaker_to_side:
                new_seg = dict(seg)
                new_seg["speaker"] = speaker_to_side[spk]
                out.append(new_seg)
            else:
                out.append(seg)
        return out

    def _label_speech_segments_with_active_speaker(
        self,
        video_path: str,
        speech_segments: List[Dict],
        sample_fps: float = 15.0,
    ) -> List[Dict]:
        """Assign LEFT/RIGHT labels directly to Whisper speech segments using video mouth-motion activity."""
        if not speech_segments:
            return []

        activity = self._estimate_active_speaker_side(video_path, sample_fps=sample_fps)
        if not activity:
            return []

        times = np.array([a["t"] for a in activity], dtype=np.float32)
        left_vals = np.array([a["left"] for a in activity], dtype=np.float32)
        right_vals = np.array([a["right"] for a in activity], dtype=np.float32)

        labeled: List[Dict] = []
        for seg in speech_segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end <= start:
                continue
            mask = (times >= start) & (times <= end)
            if not np.any(mask):
                continue

            l = float(np.mean(left_vals[mask]))
            r = float(np.mean(right_vals[mask]))
            speaker = "LEFT" if l >= r else "RIGHT"
            labeled.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": seg.get("text", ""),
                "confidence": 1.0,
            })

        return labeled
    
    def create_turn_sections(self, segments: List[Dict], overlap_duration: float = 0.0) -> List[Dict]:
        """
        Create video sections based on turn-taking with overlap for simultaneous speech.
        
        Args:
            segments: Speaker segments with timestamps
            overlap_duration: Duration of overlap when speakers change (seconds)
            
        Returns:
            List of video sections to extract
        """
        if not segments:
            return []
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        sections = []
        current_speaker = segments[0]["speaker"]
        section_start = segments[0]["start"]
        section_end = segments[0]["end"]
        
        for i in range(1, len(segments)):
            segment = segments[i]
            
            if segment["speaker"] != current_speaker:
                # Speaker change detected - create section with no overlap
                transition_time = max(0.0, float(segment["start"]))
                prev_end = min(section_end, transition_time)
                
                if prev_end > section_start:
                    sections.append({
                        "start": section_start,
                        "end": prev_end,
                        "speaker": current_speaker,
                        "section_id": len(sections)
                    })
                
                # Start new section at the transition time (no overlap)
                current_speaker = segment["speaker"]
                section_start = transition_time
                section_end = segment["end"]
            else:
                # Same speaker - extend current section
                section_end = max(section_end, segment["end"])
        
        # Add final section
        sections.append({
            "start": section_start,
            "end": section_end,
            "speaker": current_speaker,
            "section_id": len(sections)
        })
        
        return sections

    def create_utterance_sections(self, segments: List[Dict], max_gap: float = 0.25) -> List[Dict]:
        """Create sections from speech segments (one clip per utterance).

        Segments are expected to already have a `speaker` label (e.g., LEFT/RIGHT).
        Adjacent segments from the same speaker separated by <= `max_gap` seconds
        are merged into a single section.
        """
        if not segments:
            return []

        segs = sorted(segments, key=lambda x: float(x.get("start", 0.0)))
        sections: List[Dict] = []

        cur = dict(segs[0])
        cur_start = float(cur.get("start", 0.0))
        cur_end = float(cur.get("end", 0.0))
        cur_speaker = cur.get("speaker", "UNKNOWN")

        for seg in segs[1:]:
            s_start = float(seg.get("start", 0.0))
            s_end = float(seg.get("end", 0.0))
            s_speaker = seg.get("speaker", "UNKNOWN")

            if s_speaker == cur_speaker and s_start - cur_end <= max_gap:
                cur_end = max(cur_end, s_end)
            else:
                if cur_end > cur_start:
                    sections.append({
                        "start": cur_start,
                        "end": cur_end,
                        "speaker": cur_speaker,
                        "section_id": len(sections),
                    })
                cur_start = s_start
                cur_end = s_end
                cur_speaker = s_speaker

        if cur_end > cur_start:
            sections.append({
                "start": cur_start,
                "end": cur_end,
                "speaker": cur_speaker,
                "section_id": len(sections),
            })

        return sections
    
    def split_video_section(self, video_path: str, start_time: float, 
                           end_time: float, output_path: str) -> bool:
        """
        Extract a section from the video.
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save output section
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Extracting section: {start_time:.2f}s - {end_time:.2f}s")
            
            # Load video
            with VideoFileClip(video_path) as clip:
                # Extract subclip
                subclip = clip.subclip(start_time, end_time)
                
                # Write to file
                subclip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                subclip.close()
            
            print(f"Saved section: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error extracting section: {e}")
            return False
    
    def process_single_video(
        self,
        video_path: str,
        overlap_duration: float = 0.0,
        use_visual_active_speaker: bool = True,
        utterance_sections: bool = True,
        overlap_padding: float = 0.3,
    ) -> bool:
        """
        Process a single video and split by speaker turns.
        
        Args:
            video_path: Path to video file
            overlap_duration: Duration of overlap when speakers change (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing video: {video_path}")
            
            # Extract audio
            temp_audio_path = self.diarizer.extract_audio(video_path)
            if temp_audio_path is None:
                return False
            
            # Perform speech activity detection with Whisper to refine boundaries
            speech_segments = self.diarizer.detect_speech_activity(temp_audio_path)
            
            # Perform speaker diarization
            speaker_segments = self.diarizer.perform_diarization(temp_audio_path)
            
            unique_diar_speakers = sorted({s.get("speaker") for s in speaker_segments if s.get("speaker") is not None})
            print(f"Diarization speakers: {unique_diar_speakers}")
            
            # Combine speech and speaker information to refine boundaries
            combined_segments = self.diarizer.combine_speech_and_speakers(
                speech_segments, speaker_segments
            )
            
            unique_combined = sorted({s.get("speaker") for s in combined_segments if s.get("speaker") is not None})
            print(f"Combined segment speakers: {unique_combined}")
            
            # Fallback: if only one speaker detected, assign alternating LEFT/RIGHT for split-screen videos
            if len(unique_combined) < 2:
                print("Warning: only one speaker detected by diarization; assigning alternating LEFT/RIGHT labels")
                for i, seg in enumerate(combined_segments):
                    seg["speaker"] = "LEFT" if i % 2 == 0 else "RIGHT"
                unique_combined = ["LEFT", "RIGHT"]
                print(f"Fallback segment speakers: {unique_combined}")
            
            # Clean up temporary audio file
            os.unlink(temp_audio_path)
            
            if not combined_segments:
                print("No speech segments detected")
                return False
            
            if utterance_sections:
                sections = self.create_utterance_sections(combined_segments)
            else:
                # Create turn sections
                sections = self.create_turn_sections(combined_segments, overlap_duration)
            
            # Generate output filename base
            video_name = Path(video_path).stem

            # Get video duration once for safe padding
            try:
                with VideoFileClip(video_path) as _clip:
                    video_duration = float(_clip.duration)
            except Exception:
                video_duration = None
            
            # Save segment information
            segments_file = self.output_dir / f"{video_name}_segments.json"
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "video": Path(video_path).name,
                    "segments": combined_segments,
                    "sections": sections
                }, f, indent=2)
            
            print(f"Found {len(sections)} speaker turns")

            # Extract video sections
            successful = 0
            for section in sections:
                output_path = self.output_dir / f"{video_name}_section_{section['section_id']:03d}_{section['speaker']}.mp4"

                start_time = float(section["start"])
                end_time = float(section["end"])
                if overlap_padding and overlap_padding > 0:
                    start_time = max(0.0, start_time - float(overlap_padding))
                    end_time = end_time + float(overlap_padding)
                    if video_duration is not None:
                        end_time = min(end_time, video_duration)

                if self.split_video_section(
                    video_path,
                    start_time,
                    end_time,
                    str(output_path)
                ):
                    successful += 1

            print(f"Successfully extracted {successful}/{len(sections)} sections")
            print(f"Segments saved to: {segments_file}")

            return successful > 0
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return False
    
    def process_all_videos(
        self,
        overlap_duration: float = 0.0,
        use_visual_active_speaker: bool = True,
        utterance_sections: bool = True,
        overlap_padding: float = 0.3,
    ) -> None:
        """
        Process all video files in the input directory.
        
        Args:
            overlap_duration: Duration of overlap when speakers change (seconds)
        """
        print(f"Starting video splitting by speaker turns...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all video files
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(glob.glob(str(self.input_dir / f"*{ext}")))
        
        if not video_files:
            print("No video files found in input directory!")
            return
        
        print(f"Found {len(video_files)} video files")
        print("-" * 50)
        
        # Process each video
        successful = 0
        failed = 0
        
        for video_path in video_files:
            if self.process_single_video(
                video_path,
                overlap_duration,
                use_visual_active_speaker=use_visual_active_speaker,
                utterance_sections=utterance_sections,
                overlap_padding=overlap_padding,
            ):
                successful += 1
            else:
                failed += 1
        
        print("-" * 50)
        print(f"Video splitting complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_dir}")


def main():
    """
    Main function to split videos by speaker turns.
    """
    # Initialize video splitter
    splitter = VideoSplitter()
    
    # Process all videos
    splitter.process_all_videos(
        overlap_duration=0.0,
        use_visual_active_speaker=True,
        utterance_sections=True,
        overlap_padding=0.3,
    )


if __name__ == "__main__":
    main()
