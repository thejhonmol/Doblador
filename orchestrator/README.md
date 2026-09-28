# ⚙️ Doblador - Node.js & BullMQ Orchestrator

Central pipeline coordination engine for automated audiovisual dubbing. Manages job queues with **BullMQ** and **Redis**, interfaces with the **PostgreSQL** database, and orchestrates the sequential execution of workers.

## 🔄 Queue Pipeline (Workers)
1. **`extract`** ([`src/workers/extract.ts`](file:///c:/Users/USER/Documents/Doblador/orchestrator/src/workers/extract.ts)): Probes video duration, extracts audio with FFmpeg, calls Demucs for voice/instrumental separation, and runs transcription + diarization via the Python microservice.
2. **`translate`** ([`src/workers/translate.ts`](file:///c:/Users/USER/Documents/Doblador/orchestrator/src/workers/translate.ts)): Splits dialogues into 35-segment batches with full video context, explicit syllable budgets, DNT rules, and an automatic Gemini fallback chain (`gemini-3.7-flash` -> `3.5` -> `3.8` -> `3.1` -> `latest`).
3. **`tts`** ([`src/workers/tts.ts`](file:///c:/Users/USER/Documents/Doblador/orchestrator/src/workers/tts.ts)): Allocates voices, runs up to six Fish Audio synthesis slots, measures every clip, rewrites/regenerates duration outliers once, normalizes LUFS, applies strict drift/overlap gates, and persists the synchronization report.
4. **`assemble`** ([`src/workers/assemble.ts`](file:///c:/Users/USER/Documents/Doblador/orchestrator/src/workers/assemble.ts)): Muxes the original video with the pre-stitched continuous speech track and clean Demucs background music using FFmpeg (`apad,atrim` bounds ensuring exact video length).

## 🚀 Build and Run
```bash
# Build TypeScript
npm run build

# Start production server
npm start

# Development mode (watch/rebuild)
npm run dev
```
The HTTP API server listens on `http://127.0.0.1:3000`.
