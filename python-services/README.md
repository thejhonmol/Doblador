# 🐍 Doblador - Python AI & DSP Microservices

High-performance service built with **FastAPI** hosting GPU-accelerated deep learning models and Digital Signal Processing (DSP) algorithms for the dubbing pipeline.

## 🧠 Loaded Models
1. **MOSS-Transcribe-Diarize** (`MOSS-Transcribe-Diarize-Q5_K_M.gguf`): Single-pass STT and speaker diarization via Audio-LLM (0.9B) accelerated on GPU (Vulkan/GGML).
2. **Sortformer Diarization** (`diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf`): Fallback multi-speaker diarization for up to 4 speakers.
3. **faster-whisper** (`base` on CUDA float16): Secondary speech recognition and fallback transcription.
4. **Wav2Vec2 Age & Gender** (`audeering/wav2vec2-large-robust-24-ft-age-gender`): Acoustic demographic classification for dynamic voice allocation.
5. **Demucs** (`htdemucs` with `--two-stems vocals`): Deep source separation isolating speech from music, Foley, and ambient background tracks.

## 🎛️ Digital Signal Processing (DSP)
- **`pyloudnorm`**: ITU-R BS.1770-4 loudness measurement and calibration per speaker.
- **`librosa`**: Energy-based dynamic VAD trimming (`top_db=25`) and pitch-preserving phase vocoder time-stretching (`atempo`).
- **`scipy` / `numpy`**: Anti-aliased multi-rate resampling, soft-knee bus limiting, and strict per-speaker non-collision stitching (`/stitch-tts`).

## 🚀 Manual Execution
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Launch FastAPI server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
API root endpoint: `http://127.0.0.1:8000/`.
