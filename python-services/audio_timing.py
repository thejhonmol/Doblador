"""Sample-based dialogue fitting, independent of the GPU model services."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
import math
import subprocess

import numpy as np
from scipy.signal import resample


@dataclass
class TimingSlot:
    segment_index: int
    speaker: str
    original_start_ms: float
    original_end_ms: float
    start_sample: int
    end_sample: int


def plan_timing_slots(segments: list[dict], sr: int, total_samples: int) -> list[TimingSlot]:
    """Keep source timing; bound each clip by the next take of the same speaker.

    Different speakers retain source overlaps. Duplicate starts for one speaker
    share their common window sequentially, without shifting later dialogue.
    """
    if sr <= 0 or total_samples <= 0:
        raise ValueError("Sample rate and track length must be positive")
    speakers = defaultdict(list)
    for index, segment in enumerate(segments):
        start_ms = float(segment.get("start_ms", 0) or 0)
        end_ms = float(segment.get("end_ms", start_ms + 500))
        if not math.isfinite(start_ms) or not math.isfinite(end_ms) or end_ms <= start_ms:
            raise ValueError(f"segment {index} has invalid timestamps")
        start = max(0, round(start_ms * sr / 1000))
        end = min(total_samples, round(end_ms * sr / 1000))
        if end <= start:
            raise ValueError(f"segment {index} has no time inside the video")
        speaker = str(segment.get("speaker_label") or segment.get("speaker") or "SPEAKER_00").strip()
        speakers[speaker].append(TimingSlot(index, speaker, start_ms, end_ms, start, end))

    slots = []
    for speaker_slots in speakers.values():
        speaker_slots.sort(key=lambda slot: (slot.start_sample, slot.segment_index))
        groups = [list(group) for _, group in groupby(speaker_slots, key=lambda slot: slot.start_sample)]
        for index, group in enumerate(groups):
            start = group[0].start_sample
            next_start = groups[index + 1][0].start_sample if index + 1 < len(groups) else total_samples
            end = min(max(slot.end_sample for slot in group), next_start)
            room = end - start
            if room < len(group):
                raise ValueError(f"segment {group[0].segment_index} has no usable timing window")
            for position, slot in enumerate(group):
                slot.start_sample = start + room * position // len(group)
                slot.end_sample = start + room * (position + 1) // len(group)
                slots.append(slot)
    return sorted(slots, key=lambda slot: (slot.start_sample, slot.segment_index))


def atempo_filter(speed: float) -> str:
    """Chain factors of at most 2x so acceleration has no application ceiling."""
    if not math.isfinite(speed) or speed < 1:
        raise ValueError("Speed must be finite and at least 1")
    factors = []
    while speed > 2:
        factors.append("atempo=2")
        speed /= 2
    factors.append(f"atempo={speed:.12g}")
    return ",".join(factors)


def fit_audio_to_samples(audio: np.ndarray, sr: int, available_samples: int) -> tuple[np.ndarray, float, bool]:
    """Accelerate a whole clip to its window; never crop speech to make it fit.

    Short takes keep their original speed and leave a pause. FFmpeg atempo keeps
    pitch for long takes. Sample-exact resampling handles residual overshoot and
    very short windows that atempo cannot represent (possibly changing pitch).
    Returns audio, effective speed and whether resampling was needed.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.all(np.isfinite(audio)):
        raise ValueError("Audio must be a nonempty, finite mono signal")
    if sr <= 0 or available_samples <= 0:
        raise ValueError("Sample rate and available samples must be positive")
    if len(audio) <= available_samples:
        return audio, 1.0, False

    speed = len(audio) / available_samples
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-filter:a", atempo_filter(speed),
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1",
        ],
        input=audio.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(30.0, len(audio) / sr * 4.0),
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"FFmpeg atempo failed: {detail}")
    fitted = np.frombuffer(result.stdout, dtype=np.float32).copy()
    resampled = False
    if len(fitted) < available_samples * 0.75:
        # At extreme ratios / tiny windows atempo can emit little or no audio.
        # Compress the complete original signal instead of losing the dialogue.
        # Linear interpolation avoids a Fourier low-pass removing virtually all
        # speech at extreme ratios. This last resort can change pitch / alias.
        fitted = np.interp(
            np.linspace(0, len(audio) - 1, available_samples),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        resampled = True
    elif len(fitted) > available_samples:
        fitted = resample(fitted, available_samples).astype(np.float32)
        resampled = True
    return fitted, len(audio) / len(fitted), resampled
