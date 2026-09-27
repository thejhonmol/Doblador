# 🌐 Doblador - Web Client

Modern, responsive user interface built with **React 19**, **TypeScript**, and **Vite**, engineered for real-time monitoring and playback of the autonomous dubbing pipeline.

## 🚀 Features
- **Video Upload**: Fast multi-format video upload (MP4, MKV, MOV, WebM) with instant client-side preview.
- **Real-Time Stage Tracking**: Visual status timeline tracking each stage of the orchestrator:
  - `extract`: Stem separation (Demucs), transcription (MOSS), and speaker diarization.
  - `translate`: Full-context script adaptation and textual expansion with Gemini Flash.
  - `tts`: Parallel multi-key synthesis, LUFS calibration, and 4-layer DSP stitching.
  - `assemble`: Clean background stem mixing with FFmpeg and duration preservation.
- **Speaker Assignment Summary**: Live breakdown of detected speakers, assigned voices, and translated segment counts.
- **Embedded Player**: Direct playback and one-click download of the final dubbed MP4 file.

## 🛠️ Local Development
```bash
npm install
npm run dev
```
Available by default at `http://localhost:5173/`.
