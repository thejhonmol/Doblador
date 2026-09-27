from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import sys
import math
import tempfile
import subprocess
import shutil

# Add NVIDIA CUDA DLL paths to PATH so ctranslate2 can find cublas64_12.dll, cudnn, etc.
_site_packages = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages')
_nvidia_base = os.path.join(os.path.abspath(_site_packages), 'nvidia')
if os.path.isdir(_nvidia_base):
    for lib_dir in os.listdir(_nvidia_base):
        bin_path = os.path.join(_nvidia_base, lib_dir, 'bin')
        if os.path.isdir(bin_path):
            os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
            os.add_dll_directory(bin_path)

import json
import pyloudnorm as pyln
import numpy as np
import soundfile as sf
import scipy
import scipy.signal
from faster_whisper import WhisperModel
import transcribe_cpp as tc
from huggingface_hub import hf_hub_download

def measure_audio_lufs(audio_data: np.ndarray, sr: int) -> float:
    """Measures integrated LUFS using pyloudnorm (ITU-R BS.1770-4), handling short audio safely."""
    try:
        meter = pyln.Meter(sr)
        min_samples = int(0.45 * sr)
        if len(audio_data) < min_samples:
            repeats = int(np.ceil(min_samples / max(1, len(audio_data))))
            padded = np.tile(audio_data, repeats)[:min_samples]
            lufs = meter.integrated_loudness(padded)
        else:
            lufs = meter.integrated_loudness(audio_data)
        if np.isfinite(lufs):
            return float(lufs)
    except Exception as e:
        print(f"LUFS measure warning: {e}")
    return -18.0


def safe_path(raw_path: str, must_exist: bool = False) -> str:
    """
    Normalises a client-supplied path and rejects traversal.

    Several endpoints take absolute paths straight from the caller (`output_path`,
    `output_background_path`, `tts_audio_url`, `vocals_path`), which is arbitrary file
    read/write for anyone who can reach the service. Binding to 127.0.0.1 already
    keeps it off the network; this closes the traversal case.
    """
    if not raw_path or not str(raw_path).strip():
        raise HTTPException(status_code=400, detail="Empty path.")
    candidate = os.path.abspath(str(raw_path))
    if ".." in str(raw_path).split(os.sep) or not os.path.isabs(str(raw_path)):
        raise HTTPException(status_code=400, detail=f"Path must be absolute and free of '..': {raw_path}")
    if must_exist and not os.path.exists(candidate):
        raise HTTPException(status_code=404, detail=f"File not found: {candidate}")
    return candidate

app = FastAPI(title="Doblador MVP - Python Services")

# Initialize Whisper model.
# Tries CUDA first, falls back to CPU if CUDA runtime libs are missing.
print("Loading Whisper model (base)...")
try:
    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    whisper_model.transcribe(test_audio, beam_size=1)
    print("Whisper model loaded and verified on CUDA!")
except Exception as e:
    print(f"CUDA not usable ({e}), loading on CPU...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=8)
    print("Whisper model loaded on CPU (int8).")

# Initialize Sortformer Diarization model (GGUF, GPU accelerated via Vulkan/GGML)
print("Loading Sortformer Diarization model (GGUF)...")
diar_model = None
try:
    diar_model_path = hf_hub_download(
        repo_id="handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf",
        filename="diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf"
    )
    diar_model = tc.Model(diar_model_path)
    print(f"Sortformer Diarization model loaded successfully from {diar_model_path}!")
except Exception as e:
    print(f"Warning: Could not load Sortformer Diarization model ({e}). Diarization will be disabled.")

# Initialize MOSS-Transcribe-Diarize model (Audio-LLM 0.9B, GGUF, GPU accelerated via Vulkan/GGML)
print("Loading MOSS-Transcribe-Diarize model (GGUF)...")
moss_model = None
try:
    moss_model_path = hf_hub_download(
        repo_id="handy-computer/moss-transcribe-diarize-gguf",
        filename="MOSS-Transcribe-Diarize-Q5_K_M.gguf"
    )
    moss_model = tc.Model(moss_model_path)
    print(f"MOSS-Transcribe-Diarize model loaded successfully from {moss_model_path}!")
except Exception as e:
    print(f"Warning: Could not load MOSS model ({e}).")

# Initialize audEERING Age & Gender Classifier (wav2vec2)
print("Loading Age/Gender Classifier model...")
import torch
# Disable cuDNN to avoid 'Could not load symbol cudnnGetLibConfig. Error code 127' aborts on Windows
torch.backends.cudnn.enabled = False
import torch.nn as nn
from transformers import AutoProcessor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    all_tied_weights_keys = {}
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

age_gender_model = None
age_gender_processor = None
try:
    ag_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    age_gender_processor = AutoProcessor.from_pretrained(ag_name)
    age_gender_model = AgeGenderModel.from_pretrained(ag_name)
    age_gender_model.eval()
    print("Age/Gender Classifier model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load Age/Gender model ({e}).")

# Initialize Speech Emotion Recognition (SER) model (Whisper-Large-v3 fine-tuned for SER)
print("Loading Speech Emotion Recognition model (Whisper-Large-v3 SER)...")
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

emotion_model = None
emotion_feature_extractor = None
emotion_device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_dtype = torch.float16 if emotion_device == "cuda" else torch.float32

try:
    ser_repo = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(ser_repo)
    emotion_model = AutoModelForAudioClassification.from_pretrained(
        ser_repo,
        dtype=emotion_dtype
    ).to(emotion_device)
    emotion_model.eval()
    print(f"Speech Emotion Recognition model loaded successfully on {emotion_device} ({emotion_dtype})!")
except Exception as e:
    print(f"Warning: Could not load Speech Emotion Recognition model ({e}).")

GENDER_LABELS = {0: "female", 1: "male", 2: "child"}


def _parse_age_years(raw_value: float) -> int:
    """
    Normalises the age-head output to a plausible year count.

    The audEERING ``*-ft-age-gender`` checkpoints regress age directly in years, but
    0-1 normalised variants also circulate. The previous hardcoded ``* 100`` turned a
    45 year old into 4500, which pushed every speaker into the ``*_mature`` voice pools
    and made the young and child pools unreachable. Sniff the range instead of assuming.
    """
    value = float(raw_value)
    if not np.isfinite(value):
        return 0
    if 0.0 <= value <= 1.5:
        value *= 100.0
    return int(round(min(100.0, max(0.0, value))))

def _classify_speakers(audio_16k, segments):
    """
    Extracts a representative audio slice for each speaker and predicts gender and age.
    """
    if age_gender_model is None or age_gender_processor is None or len(audio_16k) == 0:
        return {}

    metadata = {}
    unique_speakers = sorted(list({s["speaker_label"] for s in segments}))

    for spk in unique_speakers:
        spk_segs = [s for s in segments if s["speaker_label"] == spk]
        spk_segs.sort(key=lambda s: s["end_ms"] - s["start_ms"], reverse=True)
        if not spk_segs:
            continue

        best_seg = spk_segs[0]
        start_idx = int((best_seg["start_ms"] / 1000.0) * 16000)
        max_duration_samples = int(4.0 * 16000)
        end_idx = min(start_idx + max_duration_samples, int((best_seg["end_ms"] / 1000.0) * 16000), len(audio_16k))
        
        slice_audio = audio_16k[start_idx:end_idx]
        if len(slice_audio) < 16000 * 0.5:
            continue

        try:
            inputs = age_gender_processor(slice_audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                _, age_out, gender_out = age_gender_model(inputs["input_values"])
            age = _parse_age_years(age_out.item())
            gender_idx = torch.argmax(gender_out, dim=1).item()
            gender = GENDER_LABELS.get(gender_idx, "unknown")
            confidence = round(gender_out[0][gender_idx].item(), 3)
            metadata[spk] = {
                "gender": gender,
                "age": age,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Age/Gender classification error for {spk}: {e}")

    return metadata


def _classify_emotions(audio_16k, segments):
    """
    Classifies the emotion of each audio segment using the Whisper-Large-v3 SER model
    (firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3).
    Labels: angry, disgust, fearful, happy, neutral, sad, surprised.
    """
    if emotion_model is None or emotion_feature_extractor is None or len(audio_16k) == 0:
        for s in segments:
            s.setdefault("emotion", "neutral")
            s.setdefault("emotion_confidence", 0.5)
        return segments

    total_samples = len(audio_16k)
    sr = 16000
    max_chunk = 30 * sr  # Whisper max window is 30 seconds

    for seg in segments:
        try:
            start_ms = seg.get("start_ms", 0) or 0
            end_ms = seg.get("end_ms", 0) or 0
            start_idx = max(0, int((start_ms / 1000.0) * sr))
            end_idx = min(total_samples, int((end_ms / 1000.0) * sr))

            if end_idx <= start_idx:
                seg["emotion"] = "neutral"
                seg["emotion_confidence"] = 0.5
                continue

            slice_audio = audio_16k[start_idx:end_idx]
            # Minimum audio length for meaningful classification: 0.25 seconds
            if len(slice_audio) < int(sr * 0.25):
                seg["emotion"] = "neutral"
                seg["emotion_confidence"] = 0.5
                continue

            if len(slice_audio) > max_chunk:
                slice_audio = slice_audio[:max_chunk]

            inputs = emotion_feature_extractor(slice_audio, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(emotion_device, dtype=emotion_dtype)

            with torch.no_grad():
                outputs = emotion_model(input_features)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = float(probs[0][pred_idx].item())
                label = emotion_model.config.id2label.get(pred_idx, "neutral")

            seg["emotion"] = label
            seg["emotion_confidence"] = round(confidence, 3)
        except Exception as e:
            print(f"[SER] Emotion classification error for segment {seg.get('start_ms')}-{seg.get('end_ms')}: {e}")
            seg["emotion"] = "neutral"
            seg["emotion_confidence"] = 0.5

    return segments


def _aggregate_speaker_emotions(segments, speakers_metadata):
    """
    Computes primary emotion and emotion distribution per speaker.
    """
    from collections import Counter
    speaker_emotions = {}
    for seg in segments:
        spk = seg.get("speaker_label", "SPEAKER_00")
        em = seg.get("emotion", "neutral")
        if spk not in speaker_emotions:
            speaker_emotions[spk] = Counter()
        speaker_emotions[spk][em] += 1

    for spk, counter in speaker_emotions.items():
        if spk in speakers_metadata:
            most_common = counter.most_common(1)
            speakers_metadata[spk]["primary_emotion"] = most_common[0][0] if most_common else "neutral"
            speakers_metadata[spk]["emotion_distribution"] = dict(counter)


DEFAULT_ENGINE = os.environ.get("TRANSCRIBE_ENGINE", "moss")


def _assign_speakers(whisper_segments, diar_segments):
    """
    Assigns each Whisper segment to the speaker with the maximum temporal overlap.
    Falls back to the most recent speaker or SPEAKER_01 if no overlap is detected.
    """
    last_speaker = "SPEAKER_01"
    assigned = []
    
    for seg in whisper_segments:
        start_ms = seg["start_ms"]
        end_ms = seg["end_ms"]
        
        overlap_per_speaker = {}
        for ds in diar_segments:
            ov = max(0, min(end_ms, ds.t1_ms) - max(start_ms, ds.t0_ms))
            if ov > 0:
                overlap_per_speaker[ds.speaker_id] = overlap_per_speaker.get(ds.speaker_id, 0) + ov
                
        if overlap_per_speaker:
            best_spk = max(overlap_per_speaker.items(), key=lambda x: x[1])[0]
            speaker_label = f"SPEAKER_{best_spk:02d}"
            last_speaker = speaker_label
        else:
            speaker_label = last_speaker
            
        assigned.append({
            **seg,
            "speaker_label": speaker_label
        })
        
    return assigned


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Python microservices are running",
        "default_engine": DEFAULT_ENGINE,
        "moss_enabled": moss_model is not None,
        "diarization_enabled": diar_model is not None,
        "age_gender_enabled": age_gender_model is not None,
        "emotion_recognition_enabled": emotion_model is not None,
        "emotion_device": str(emotion_device) if emotion_model is not None else None,
        "whisper_device": getattr(whisper_model, "device", "unknown")
    }

@app.post("/separate")
async def separate_audio(
    file: UploadFile = File(...),
    output_background_path: str = Form(None),
    output_vocals_path: str = Form(None),
    timeout_sec: float = Form(None)
):
    """
    Separates audio using Demucs (GPU/CUDA mode if available, else CPU).
    Returns paths to separated stems (vocals and background / no_vocals),
    plus the integrated loudness (LUFS) of the isolated vocals using pyloudnorm.

    The temp directory is always removed: it used to be leaked on every call, and a
    Demucs run writes a full extra copy of both stems into it.
    """
    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, os.path.basename(file.filename))

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        out_dir = os.path.join(temp_dir, "demucs_out")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,
            "--shifts", "1",
            "--segment", "7",
            "-o", out_dir,
            input_path
        ]

        # Demucs can outrun any fixed budget on a long file; the caller knows the
        # video length and passes a budget, and we fail loudly instead of hanging.
        effective_timeout = float(timeout_sec) if timeout_sec and timeout_sec > 0 else 3600.0

        print(f"Running Demucs ({device}, timeout {effective_timeout:.0f}s): {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail=f"Demucs exceeded {effective_timeout:.0f}s and was terminated."
            )

        if process.returncode != 0:
            print("Demucs error:", process.stderr)
            raise HTTPException(status_code=500, detail=f"Demucs separation failed: {process.stderr[:300]}")

        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        model_out_dir = os.path.join(out_dir, "htdemucs")
        subdirs = [os.path.join(model_out_dir, d) for d in os.listdir(model_out_dir) if os.path.isdir(os.path.join(model_out_dir, d))]
        track_dir = subdirs[0] if subdirs else os.path.join(model_out_dir, base_name)

        vocals_path = os.path.join(track_dir, "vocals.wav")
        background_path = os.path.join(track_dir, "no_vocals.wav")

        if not os.path.exists(vocals_path) or not os.path.exists(background_path):
            raise HTTPException(status_code=500, detail="Demucs output files not found.")

        if output_background_path:
            out_bg = safe_path(output_background_path)
            os.makedirs(os.path.dirname(out_bg), exist_ok=True)
            shutil.copyfile(background_path, out_bg)
            background_path = out_bg
            print(f"Copied separated background stem to: {out_bg}")

        if output_vocals_path:
            out_voc = safe_path(output_vocals_path)
            os.makedirs(os.path.dirname(out_voc), exist_ok=True)
            shutil.copyfile(vocals_path, out_voc)
            vocals_path = out_voc
            print(f"Copied separated vocals stem to: {out_voc}")

        # Measure original isolated vocal track loudness with pyloudnorm
        vocal_data, vocal_sr = sf.read(vocals_path)
        vocals_lufs = measure_audio_lufs(vocal_data, vocal_sr)
        print(f"[pyloudnorm] Original isolated vocal track: {vocals_lufs:.2f} LUFS")
            
        return {
            "vocals_path": vocals_path,
            "background_path": background_path,
            "vocals_lufs": round(vocals_lufs, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        print("Demucs exception:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Was leaked on every call, including failures: a Demucs run writes a full
        # extra copy of both stems here.
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/measure-speakers-loudness")
async def measure_speakers_loudness(
    vocals_path: str = Form(...),
    segments_json: str = Form(...)
):
    """
    Measures the original voice loudness (LUFS) for each speaker individually
    using the separated vocals.wav and the diarization timestamps.
    """
    try:
        segments = json.loads(segments_json)
        resolved_vocals = safe_path(vocals_path, must_exist=True)

        vocal_data, sr = sf.read(resolved_vocals)
        overall_lufs = measure_audio_lufs(vocal_data, sr)

        speaker_audio = {}
        for seg in segments:
            spk = seg.get("speaker_label", "SPEAKER_01")
            start_sec = (seg.get("start_ms", 0)) / 1000.0
            end_sec = (seg.get("end_ms", 0)) / 1000.0
            start_idx = max(0, int(start_sec * sr))
            end_idx = min(len(vocal_data), int(end_sec * sr))
            if end_idx > start_idx:
                slice_data = vocal_data[start_idx:end_idx]
                if spk not in speaker_audio:
                    speaker_audio[spk] = []
                speaker_audio[spk].append(slice_data)

        speakers_lufs = {}
        for spk, slices in speaker_audio.items():
            concat_data = np.concatenate(slices)
            spk_lufs = measure_audio_lufs(concat_data, sr)
            speakers_lufs[spk] = round(spk_lufs, 2)
            print(f"[pyloudnorm] Original voice volume for {spk}: {spk_lufs:.2f} LUFS")

        return {
            "overall_vocals_lufs": round(overall_lufs, 2),
            "speakers_lufs": speakers_lufs
        }
    except HTTPException:
        raise
    except Exception as e:
        print("Error measuring speakers loudness:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/normalize-tts")
async def normalize_tts(
    output_dir: str = Form(...),
    target_lufs: float = Form(-18.0),
    speakers_lufs_json: str = Form("{}"),
    segments_json: str = Form("[]")
):
    """
    Adjusts the loudness of each generated TTS audio segment in output_dir using pyloudnorm
    so it matches the original voice loudness (either per-speaker or overall).
    """
    try:
        resolved_dir = safe_path(output_dir)
        if not os.path.isdir(resolved_dir):
            raise HTTPException(status_code=404, detail=f"Output dir not found: {resolved_dir}")

        speakers_lufs = json.loads(speakers_lufs_json) if speakers_lufs_json else {}
        segments = json.loads(segments_json) if segments_json else []
        seg_speaker_map = {f"seg_{i}": s.get("speaker_label") for i, s in enumerate(segments)}

        files = [f for f in os.listdir(resolved_dir) if f.endswith(('.mp3', '.wav'))]
        adjusted_count = 0
        details = []
        # Measured duration of every file, so the caller can record what TTS actually
        # produced instead of the slot it was asked to fill. Used to be faked as
        # `generated_ms = end_ms - start_ms`, which made an over-long clip look
        # compliant in the database.
        durations: dict[str, float] = {}

        for fname in sorted(files):
            fpath = os.path.join(resolved_dir, fname)
            base_name = os.path.splitext(fname)[0]
            spk = seg_speaker_map.get(base_name)

            # Determine target LUFS: specific speaker loudness or overall vocals loudness
            spk_target = speakers_lufs.get(spk) if spk else None
            desired_lufs = float(spk_target) if spk_target is not None else float(target_lufs)

            data, sr = sf.read(fpath)
            durations[fname] = round(len(data) / float(sr), 3)
            cur_lufs = measure_audio_lufs(data, sr)
            if not np.isfinite(cur_lufs) or cur_lufs < -70:
                continue

            gain_db = desired_lufs - cur_lufs

            try:
                norm_audio = pyln.normalize.loudness(data, cur_lufs, desired_lufs)
                peak = np.max(np.abs(norm_audio))
                if peak > 0.98:
                    norm_audio = norm_audio * (0.98 / peak)

                sf.write(fpath, norm_audio, sr)
                adjusted_count += 1
                details.append({
                    "file": fname,
                    "speaker": spk,
                    "before_lufs": round(cur_lufs, 2),
                    "after_lufs": round(desired_lufs, 2),
                    "gain_db": round(gain_db, 2)
                })
                print(f"[pyloudnorm] {fname} ({spk or 'all'}): {cur_lufs:.2f} LUFS -> {desired_lufs:.2f} LUFS (Gain: {gain_db:+.2f} dB)")
            except Exception as norm_err:
                print(f"[pyloudnorm] Warning on {fname}: {norm_err}")

        return {
            "status": "success",
            "adjusted_count": adjusted_count,
            "target_lufs": target_lufs,
            "durations_sec": durations,
            "details": details[:10]
        }
    except HTTPException:
        raise
    except Exception as e:
        print("Normalize TTS error:", e)
        raise HTTPException(status_code=500, detail=str(e))


def _trim_silence_dynamic(audio: np.ndarray, sr: int, top_db: float = 25.0) -> np.ndarray:
    """
    Layer 1: Dynamic silence trimming using energy-based VAD (librosa).
    Removes leading/trailing silence adaptively instead of fixed ms cutoffs.
    top_db=25 is aggressive enough to catch Fish Audio padding without clipping speech.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=1024, hop_length=256)
    # Safety: never return empty audio
    if len(trimmed) < int(0.05 * sr):
        return audio
    return trimmed


def _apply_atempo(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """
    Layer 2: Time-stretch via phase vocoder (scipy-based, no extra deps).
    Preserves pitch. Only applied when speed > 1.03 to avoid unnecessary processing.
    Speed range: 1.0-1.35x (beyond 1.35 sounds artificial).
    """
    if speed <= 1.03 or speed > 2.0:
        return audio
    # Use scipy phase vocoder approach via librosa
    import librosa
    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
    return stretched


# ── Stitch-TTS Configuration ──────────────────────────────────────────────
STITCH_SR = 44100
MIN_SAME_SPEAKER_GAP_MS = 70.0   # Physical breath/gap between phrases of the SAME speaker. ZERO overlap allowed!
MAX_CROSS_SPEAKER_OVERLAP_MS = 250.0 # Natural conversational overlap allowed between DIFFERENT speakers
MAX_DRIFT_MS = 900.0             # Soft ceiling for drift alert
DRIFT_RECOVERY_START_MS = 150.0  # From here, boost atempo to win back time before next sentence
MAX_ATEMPO_SOFT = 1.18           # Preferred ceiling — natural sounding time-stretch
MAX_ATEMPO_HARD = 1.35           # Absolute ceiling — still acceptable quality
TRIM_TOP_DB = 25.0               # Layer 1: librosa trim aggressiveness
SOFT_LIMIT_CEILING = 0.98
SOFT_LIMIT_KNEE = 0.80


def _soft_limit(
    audio: np.ndarray,
    ceiling: float = SOFT_LIMIT_CEILING,
    knee: float = SOFT_LIMIT_KNEE,
) -> np.ndarray:
    """
    Peak protection that leaves everything below the knee bit-identical and only
    compresses the peaks.
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling:
        return audio

    out = audio.copy()
    hot = np.abs(out) > knee
    if not np.any(hot):
        return out

    span = max(1e-6, ceiling - knee)
    excess = (np.abs(out[hot]) - knee) / span
    out[hot] = np.sign(out[hot]) * (knee + span * np.tanh(excess))
    return out


@app.post("/stitch-tts")
async def stitch_tts(
    segments_json: str = Form(...),
    total_duration_sec: float = Form(...),
    output_path: str = Form(...)
):
    """
    Stitches individual TTS segments into a single continuous audio track with
    strict per-speaker non-collision and anti-drift processing:

    1. Dynamic silence trim (librosa VAD) to remove TTS padding.
    2. Per-Speaker Tracking: A single speaker has only one mouth; a speaker can
       NEVER start a new phrase until their previous phrase is done (+70ms breath gap).
    3. Cross-Speaker Natural Mixing: Different speakers can interrupt or cross-talk
       naturally, summing on the bus.
    4. Forward-Only Drift Recovery: Drift is recovered by compressing clip durations
       (atempo), NEVER by snapping start times backward in time onto playing audio.
    """
    try:
        segments = json.loads(segments_json)
        sr = STITCH_SR
        total_samples = int(np.ceil(total_duration_sec * sr))
        total_ms = float(total_duration_sec) * 1000.0
        full_track = np.zeros(total_samples, dtype=np.float32)

        # Sort by original start time
        indexed_segments = sorted(
            ((i, seg) for i, seg in enumerate(segments)),
            key=lambda x: float(x[1].get("start_ms", 0) or 0),
        )

        # Track the actual end timestamp for EACH speaker independently
        speaker_last_end_ms: dict[str, float] = {}
        stitch_log = []
        max_start_drift_ms = 0.0

        for pos, (idx, seg) in enumerate(indexed_segments):
            audio_url = seg.get("tts_audio_url")

            if not audio_url:
                continue
            try:
                audio_url = safe_path(str(audio_url), must_exist=True)
            except HTTPException as e:
                print(f"[stitch-tts] Skipping segment {idx}: {e.detail}")
                continue

            spk = str(seg.get("speaker_label") or seg.get("speaker") or "SPEAKER_00").strip()
            original_start_ms = float(seg.get("start_ms", 0) or 0)
            original_end_ms = float(seg.get("end_ms", original_start_ms + 500) or 0)
            original_duration_ms = max(100.0, original_end_ms - original_start_ms)

            # Available slot: find when this same speaker speaks next in the script
            next_same_speaker_start_ms = None
            for _, future_seg in indexed_segments[pos + 1:]:
                future_spk = str(future_seg.get("speaker_label") or future_seg.get("speaker") or "SPEAKER_00").strip()
                if future_spk == spk:
                    cand = float(future_seg.get("start_ms", 0) or 0)
                    if cand > original_start_ms:
                        next_same_speaker_start_ms = cand
                        break

            slot_end_ms = min(original_end_ms, next_same_speaker_start_ms) if next_same_speaker_start_ms is not None else original_end_ms
            available_ms = max(200.0, slot_end_ms - original_start_ms)

            try:
                data, in_sr = sf.read(audio_url)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # Resample with anti-aliasing prefilter
                if in_sr != sr:
                    divisor = math.gcd(int(in_sr), int(sr))
                    data = scipy.signal.resample_poly(data, sr // divisor, in_sr // divisor)

                data = data.astype(np.float32)

                # ── Layer 1: Dynamic silence trim (librosa VAD) ──
                data = _trim_silence_dynamic(data, sr, top_db=TRIM_TOP_DB)
                clip_duration_ms = (len(data) / sr) * 1000.0

                actions: list[str] = []

                # ── Layer 2: Baseline speed to fit slot ──
                base_speed = clip_duration_ms / available_ms

                # ── Layer 3: Absolute Per-Speaker Non-Collision ──
                # A single speaker has only one vocal tract. They CANNOT articulate phrase B
                # until phrase A has completely finished + MIN_SAME_SPEAKER_GAP_MS breath gap.
                last_end_this_spk = speaker_last_end_ms.get(spk, 0.0)
                min_start_this_spk = (last_end_this_spk + MIN_SAME_SPEAKER_GAP_MS) if last_end_this_spk > 0 else 0.0

                # Start time is strictly after this speaker's last word, never overlapping!
                actual_start_ms = max(original_start_ms, min_start_this_spk)
                drift_ms = actual_start_ms - original_start_ms
                max_start_drift_ms = max(max_start_drift_ms, drift_ms)

                # ── Layer 4: Speed Calculation & Progressive Drift Recovery ──
                speed = 1.0
                if base_speed > 1.03:
                    speed = min(base_speed, MAX_ATEMPO_SOFT)
                    actions.append("FIT")

                # If the speaker is starting late due to previous phrase length,
                # speed up THIS phrase so it finishes earlier and eliminates the delay.
                if drift_ms > DRIFT_RECOVERY_START_MS:
                    remaining_ms = max(1000.0, total_ms - actual_start_ms)
                    # Smooth proportional catch-up
                    catchup_mult = 1.0 + min(0.30, (drift_ms / 2000.0))
                    boosted = min(speed * catchup_mult, MAX_ATEMPO_HARD)
                    if boosted > speed + 0.01:
                        speed = boosted
                        actions.append("RECOVER")

                # Apply time-stretch if needed
                if speed > 1.03:
                    data = _apply_atempo(data, sr, speed)

                final_clip_ms = (len(data) / sr) * 1000.0

                # ── Placement onto bus ──
                start_idx = max(0, int(actual_start_ms * sr / 1000.0))
                end_idx = min(total_samples, start_idx + len(data))
                if end_idx > start_idx:
                    full_track[start_idx:end_idx] += data[: end_idx - start_idx]

                # Update THIS speaker's end time
                actual_end_ms = actual_start_ms + final_clip_ms
                speaker_last_end_ms[spk] = actual_end_ms

                stitch_log.append({
                    "seg": idx,
                    "spk": spk,
                    "orig_start": f"{original_start_ms:.0f}ms",
                    "actual_start": f"{actual_start_ms:.0f}ms",
                    "drift": f"{drift_ms:.0f}ms",
                    "speed": f"{speed:.2f}x",
                    "reason": "+".join(actions) if actions else "NATURAL",
                    "clip_dur": f"{final_clip_ms:.0f}ms",
                })

            except Exception as e:
                print(f"[stitch-tts] Error processing segment {idx} ({audio_url}): {e}")

        # Peak protection without touching the level of everything below the knee
        full_track = _soft_limit(full_track)

        out_abs = safe_path(output_path)
        out_dir = os.path.dirname(out_abs)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(out_abs, full_track, sr)

        original_end_total = (
            float(indexed_segments[-1][1].get("end_ms") or 0) if indexed_segments else 0.0
        )
        max_actual_end = max(speaker_last_end_ms.values()) if speaker_last_end_ms else 0.0
        tail_delta_ms = max_actual_end - original_end_total
        overrun_ms = max(0.0, max_actual_end - total_ms)
        recover_count = sum(1 for l in stitch_log if "RECOVER" in l.get("reason", ""))
        fit_count = sum(1 for l in stitch_log if "FIT" in l.get("reason", ""))
        natural_count = sum(1 for l in stitch_log if l.get("reason") == "NATURAL")

        print(f"[stitch-tts] == Summary ==============================")
        print(f"[stitch-tts] Segments: {len(stitch_log)} processed | Speakers: {len(speaker_last_end_ms)}")
        print(f"[stitch-tts] Natural: {natural_count} | Compressed: {fit_count} | Recovered: {recover_count}")
        print(f"[stitch-tts] Max start drift: {max_start_drift_ms:.0f}ms (ceiling {MAX_DRIFT_MS}ms)")
        print(f"[stitch-tts] Same-speaker min gap: {MIN_SAME_SPEAKER_GAP_MS}ms (Hard zero overlap)")
        print(f"[stitch-tts] Output: {out_abs} ({total_duration_sec:.2f}s)")
        print(f"[stitch-tts] Last clip runs {tail_delta_ms:.0f}ms past its own slot")
        print(f"[stitch-tts] Overrun past buffer end: {overrun_ms:.0f}ms")
        if overrun_ms > 0:
            print(
                f"[stitch-tts] WARNING: the last clip ends {overrun_ms:.0f}ms past the buffer. "
                "The caller passed a total_duration_sec shorter than the dialogue; "
                "raise it to the source video duration."
            )

        for entry in stitch_log[:10]:
            print(f"[stitch-tts]   seg {entry['seg']}: {entry['orig_start']} -> {entry['actual_start']} | drift {entry['drift']} | {entry['speed']} ({entry['reason']}) | dur {entry['clip_dur']}")
        if len(stitch_log) > 10:
            print(f"[stitch-tts]   ... and {len(stitch_log) - 10} more segments")

        return {
            "output_path": out_abs,
            "duration_sec": total_duration_sec,
            "segments_processed": len(stitch_log),
            "natural_count": natural_count,
            "compressed_count": fit_count,
            "resynced_count": recover_count,
            "max_start_drift_ms": round(max_start_drift_ms, 1),
            "tail_delta_ms": round(tail_delta_ms, 1),
            "overrun_ms": round(overrun_ms, 1)
        }
    except HTTPException:
        # Path validation raises 400; the blanket handler below would turn it into a
        # 500 and hide a client error as a server fault.
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Stitch TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    engine: str = None,
    age_gender_audio_path: str = Form(None)
):
    """
    Transcribes audio and performs speaker diarization.
    Supports engines:
      - 'moss': MOSS-Transcribe-Diarize (Audio-LLM 0.9B, single-pass STT + Diarization)
      - 'whisper': faster-whisper (CUDA) + Sortformer (GPU Vulkan)

    `age_gender_audio_path` optionally points at the Demucs vocals stem. When present
    the speaker profiling reads the isolated voice instead of the full mix: the
    audEERING classifier is noticeably less accurate with music under the speaker, and
    the whole point of profiling is to pick the right voice.
    """
    active_engine = engine or DEFAULT_ENGINE
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, os.path.basename(file.filename or "audio.wav"))

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    profiling_audio = None
    if age_gender_audio_path:
        try:
            resolved = safe_path(age_gender_audio_path, must_exist=True)
            profiling_audio, profiling_sr = sf.read(resolved)
            if len(profiling_audio.shape) > 1:
                profiling_audio = profiling_audio.mean(axis=1)
            if profiling_sr != 16000:
                divisor = math.gcd(int(profiling_sr), 16000)
                profiling_audio = scipy.signal.resample_poly(profiling_audio, 16000 // divisor, profiling_sr // divisor)
            profiling_audio = profiling_audio.astype(np.float32)
            print(f"[transcribe] Speaker profiling will use the isolated vocals: {resolved}")
        except HTTPException as e:
            print(f"[transcribe] Falling back to the full mix for speaker profiling: {e.detail}")
            profiling_audio = None

    try:
        print(f"Transcribing audio file with engine '{active_engine}': {file.filename} ({os.path.getsize(input_path)} bytes)")
        
        # 1. Try MOSS if requested
        if active_engine == "moss" and moss_model is not None:
            try:
                print("Running MOSS-Transcribe-Diarize...")
                audio_data, sr = sf.read(input_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if sr != 16000:
                    num_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples)
                audio_data = audio_data.astype(np.float32)

                moss_session = tc.Session(moss_model)
                moss_res = moss_session.run(audio_data, diarize='on')

                final_segments = []
                for seg in moss_res.segments:
                    final_segments.append({
                        "start_ms": seg.t0_ms,
                        "end_ms": seg.t1_ms,
                        "text": seg.text.strip(),
                        "speaker_label": f"SPEAKER_{seg.speaker_id:02d}"
                    })

                detected_lang = getattr(moss_res, 'language', 'en') or 'en'
                unique_speakers = sorted(list({s["speaker_label"] for s in final_segments}))
                print(f"MOSS complete: {len(final_segments)} segments, speakers: {unique_speakers}")

                audio_for_profiling = profiling_audio if profiling_audio is not None else audio_data
                speakers_metadata = _classify_speakers(
                    audio_for_profiling,
                    final_segments,
                )
                _classify_emotions(audio_for_profiling, final_segments)
                _aggregate_speaker_emotions(final_segments, speakers_metadata)
                for seg in final_segments:
                    meta = speakers_metadata.get(seg["speaker_label"], {})
                    seg["speaker_gender"] = meta.get("gender")
                    seg["speaker_age"] = meta.get("age")
                    
                print(f"Speaker Profiles Classified: {speakers_metadata}")

                return {
                    "engine": "moss-transcribe-diarize",
                    "language": detected_lang,
                    "speakers": unique_speakers,
                    "speakers_metadata": speakers_metadata,
                    "segments": final_segments
                }
            except Exception as me:
                print(f"MOSS inference error ({me}), falling back to Whisper + Sortformer...")

        # 2. Whisper + Sortformer pipeline
        if whisper_model is None:
            raise HTTPException(status_code=500, detail="Whisper model failed to load.")

        print("Running faster-whisper + Sortformer pipeline...")
        segments_gen, info = whisper_model.transcribe(input_path, beam_size=5, word_timestamps=False)
        detected_lang = info.language
        raw_segments = []
        for segment in segments_gen:
            raw_segments.append({
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "text": segment.text.strip(),
                "speaker_label": "SPEAKER_00"
            })
            
        audio_data, sr = sf.read(input_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != 16000:
            num_samples = int(len(audio_data) * 16000 / sr)
            audio_data = scipy.signal.resample(audio_data, num_samples)
        audio_data = audio_data.astype(np.float32)

        if diar_model is not None and len(raw_segments) > 0:
            try:
                print("Running Sortformer speaker diarization...")
                diar_session = tc.Session(diar_model)
                diar_res = diar_session.run(audio_data)
                diar_segments = diar_res.speaker_segments
                
                print(f"Diarization detected {len(diar_segments)} speaker segments.")
                final_segments = _assign_speakers(raw_segments, diar_segments)
            except Exception as de:
                print(f"Diarization error ({de}), falling back to default speaker.")
                final_segments = raw_segments
        else:
            final_segments = raw_segments

        unique_speakers = sorted(list({s["speaker_label"] for s in final_segments}))
        audio_for_profiling = profiling_audio if profiling_audio is not None else audio_data
        speakers_metadata = _classify_speakers(
            audio_for_profiling,
            final_segments,
        )
        _classify_emotions(audio_for_profiling, final_segments)
        _aggregate_speaker_emotions(final_segments, speakers_metadata)
        for seg in final_segments:
            meta = speakers_metadata.get(seg["speaker_label"], {})
            seg["speaker_gender"] = meta.get("gender")
            seg["speaker_age"] = meta.get("age")

        print(f"Whisper finished: {len(final_segments)} segments, lang: {detected_lang}, speakers: {unique_speakers}, metadata: {speakers_metadata}")
        
        return {
            "engine": "whisper-sortformer",
            "language": detected_lang,
            "speakers": unique_speakers,
            "speakers_metadata": speakers_metadata,
            "segments": final_segments
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # rmtree, not rmdir: any leftover file made rmdir fail and the whole temp
        # directory was silently left behind.
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/classify-emotions")
async def classify_emotions_endpoint(
    audio_path: str = Form(...),
    segments_json: str = Form(...)
):
    """
    Standalone endpoint to classify emotions on arbitrary segments using Whisper-Large-v3 SER.
    `audio_path` must be an absolute path to audio (e.g. isolated vocals).
    `segments_json` is a JSON array of segment dicts with start_ms and end_ms.
    """
    resolved_path = safe_path(audio_path, must_exist=True)
    audio_data, sr = sf.read(resolved_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(int(sr), 16000)
        audio_data = scipy.signal.resample_poly(audio_data, 16000 // divisor, sr // divisor)
    audio_data = audio_data.astype(np.float32)

    try:
        segments = json.loads(segments_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid segments_json: {e}")

    classified = _classify_emotions(audio_data, segments)
    return {"segments": classified}


if __name__ == "__main__":
    import uvicorn
    # Loopback only. The service takes absolute paths from its caller and has no
    # authentication, so binding 0.0.0.0 exposed arbitrary file read/write to the
    # whole LAN. Override with PY_BIND_HOST only if you understand that.
    bind_host = os.environ.get("PY_BIND_HOST", "127.0.0.1")
    uvicorn.run(app, host=bind_host, port=8000)

