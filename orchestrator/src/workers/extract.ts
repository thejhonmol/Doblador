import { Worker, Job } from 'bullmq';
import { redisConnection, PYTHON_SERVICES_URL } from '../config';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import axios from 'axios';
import fs from 'fs';
import FormData from 'form-data';

const TRANSCRIBE_MAX_ATTEMPTS = 3;
const SEPARATE_MAX_ATTEMPTS = 2;

/**
 * Demucs is roughly real-time-ish on GPU but degrades badly on long inputs, and the
 * fixed 180 s timeout aborted the axios call while the Python subprocess kept running:
 * the caller saw a failure, `backgroundAudioPath` stayed null, and the job shipped
 * with no background music. Budget generously and scale with the source length.
 */
function separateTimeoutMs(durationSec: number | null): number {
    const base = 10 * 60 * 1000;
    if (!durationSec || durationSec <= 0) return base;
    return Math.min(90 * 60 * 1000, Math.max(base, Math.round(durationSec * 2000)));
}

// Reads the container duration. This is the single source of truth for the final
// video length: TTS stages must never be allowed to define it (that truncated the
// muxed output whenever the speech track was shorter than the source video).
export function probeDurationSec(filePath: string): Promise<number | null> {
    return new Promise((resolve) => {
        ffmpeg.ffprobe(filePath, (err, data) => {
            const d = Number(data?.format?.duration);
            if (err || !Number.isFinite(d) || d <= 0) {
                console.warn(`[Extract] Could not probe duration of ${filePath}: ${err?.message || d}`);
                return resolve(null);
            }
            resolve(d);
        });
    });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const extractWorker = new Worker('extract', async (job: Job) => {
    const { videoPath, jobId, targetLang, voiceMode } = job.data;
    console.log(`[Extract] Starting extraction for job ${jobId} (voiceMode: ${voiceMode || 'clone'})`);

    // Probed before anything else so it travels with job.data through every stage.
    const videoDurationSec = await probeDurationSec(videoPath);
    console.log(`[Extract] Source video duration: ${videoDurationSec !== null ? `${videoDurationSec.toFixed(2)}s` : 'unknown'}`);

    const audioOutputPath = path.join(path.dirname(videoPath), `${jobId}_extracted.wav`);
    const backgroundAudioPath = path.join(path.dirname(videoPath), `${jobId}_background.wav`);
    const vocalsAudioPath = path.join(path.dirname(videoPath), `${jobId}_vocals.wav`);

    // Extract audio
    await new Promise((resolve, reject) => {
        ffmpeg(videoPath)
            .noVideo()
            .audioCodec('pcm_s16le')
            .audioFrequency(16000)
            .audioChannels(1)
            .save(audioOutputPath)
            .on('end', resolve)
            .on('error', reject);
    });

    console.log(`[Extract] Extracted to ${audioOutputPath}`);

    // Ensure Python services are ready (allows up to 30s for AI models to finish booting)
    let serviceReady = false;
    for (let attempt = 0; attempt < 15; attempt++) {
        try {
            await axios.get(`${PYTHON_SERVICES_URL}/`, { timeout: 2000 });
            serviceReady = true;
            break;
        } catch {
            console.log(`[Extract] Waiting for Python AI services at ${PYTHON_SERVICES_URL} to become ready... (attempt ${attempt + 1}/15)`);
            await sleep(2000);
        }
    }
    if (!serviceReady) {
        throw new Error(`Python services at ${PYTHON_SERVICES_URL} are not reachable. Please make sure python-services is running.`);
    }

    // ── Demucs separation, with retries ──
    // Previously transcription and separation ran concurrently. Both need the GPU,
    // and the parent process already holds Whisper + MOSS + Sortformer + Wav2Vec2 in
    // VRAM while the Demucs subprocess imports torch again, so running them together
    // on a 4 GB card is an OOM waiting to happen. Serialising also lets the speaker
    // profiling run on the isolated vocals instead of on top of the music.
    console.log(`[Extract] Separating vocals with Demucs...`);
    const separateTimeout = separateTimeoutMs(videoDurationSec);
    let separateResponse: any = null;
    let separateAttempts = 0;
    for (let attempt = 1; attempt <= SEPARATE_MAX_ATTEMPTS; attempt++) {
        separateAttempts = attempt;
        try {
            const form = new FormData();
            form.append('file', fs.createReadStream(audioOutputPath));
            form.append('output_background_path', backgroundAudioPath);
            form.append('output_vocals_path', vocalsAudioPath);
            const res = await axios.post(`${PYTHON_SERVICES_URL}/separate`, form, {
                headers: { ...form.getHeaders() },
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
                timeout: separateTimeout
            });
            separateResponse = res;
            break;
        } catch (err: any) {
            console.warn(`[Extract] Demucs attempt ${attempt}/${SEPARATE_MAX_ATTEMPTS} failed: ${err.message}`);
            if (attempt < SEPARATE_MAX_ATTEMPTS) {
                await sleep(10000);
                // A partial output would poison the next attempt.
                for (const p of [backgroundAudioPath, vocalsAudioPath]) {
                    try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch { /* ignore */ }
                }
            }
        }
    }

    let finalBackgroundPath: string | null = null;
    if (fs.existsSync(backgroundAudioPath)) {
        finalBackgroundPath = backgroundAudioPath;
        console.log(`[Extract] Clean background audio ready at: ${backgroundAudioPath}`);
    } else if (separateResponse?.data?.background_path && fs.existsSync(separateResponse.data.background_path)) {
        fs.copyFileSync(separateResponse.data.background_path, backgroundAudioPath);
        finalBackgroundPath = backgroundAudioPath;
        console.log(`[Extract] Clean background audio copied to: ${backgroundAudioPath}`);
    }

    const vocalsLufs = separateResponse?.data?.vocals_lufs || -18.0;
    const hasVocals = fs.existsSync(vocalsAudioPath);
    console.log(`[Extract] Original isolated vocals loudness: ${vocalsLufs} LUFS (vocals stem: ${hasVocals ? 'yes' : 'no'})`);
    if (!finalBackgroundPath) {
        console.warn(
            `[Extract] No background stem after ${separateAttempts} attempt(s). The dubbed video will carry ` +
            'only the dubbed voice: the original music and effects are NOT preserved, because the source ' +
            'audio track is never mapped into the output.'
        );
    }

    // ── Transcription + diarization ──
    console.log(`[Extract] Transcribing and diarizing...`);
    let transcribeResponse: any = null;
    let lastTranscribeError: any = null;
    for (let attempt = 1; attempt <= TRANSCRIBE_MAX_ATTEMPTS; attempt++) {
        try {
            // A fresh FormData per attempt: a form-data stream is single-use, so
            // reusing one after a failed send uploads an empty body on the retry.
            const form = new FormData();
            form.append('file', fs.createReadStream(audioOutputPath));
            // Age/gender classification runs on the isolated vocals when they exist:
            // the audEERING model is far less accurate with music under the voice, and
            // the whole point of profiling is to pick the right voice.
            if (hasVocals) {
                form.append('age_gender_audio_path', vocalsAudioPath);
            }
            const res = await axios.post(`${PYTHON_SERVICES_URL}/transcribe`, form, {
                headers: { ...form.getHeaders() },
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
                timeout: 60 * 60 * 1000
            });
            transcribeResponse = res;
            break;
        } catch (err: any) {
            lastTranscribeError = err;
            console.warn(`[Extract] Transcription attempt ${attempt}/${TRANSCRIBE_MAX_ATTEMPTS} failed: ${err.message}`);
            if (attempt < TRANSCRIBE_MAX_ATTEMPTS) await sleep(5000);
        }
    }
    if (!transcribeResponse) {
        throw new Error(`Transcription failed after ${TRANSCRIBE_MAX_ATTEMPTS} attempts: ${lastTranscribeError?.message}`);
    }

    const segments = transcribeResponse.data.segments;
    const speakersMetadata = transcribeResponse.data.speakers_metadata || {};
    console.log(`[Extract] Transcription complete. Got ${segments.length} segments.`);
    if (Object.keys(speakersMetadata).length > 0) {
        console.log(`[Extract] Speakers detected:`, JSON.stringify(speakersMetadata));
    }

    let speakersLoudness: Record<string, number> = {};
    if (hasVocals && segments.length > 0) {
        try {
            const measureFormData = new FormData();
            measureFormData.append('vocals_path', vocalsAudioPath);
            measureFormData.append('segments_json', JSON.stringify(segments));
            const measureRes = await axios.post(`${PYTHON_SERVICES_URL}/measure-speakers-loudness`, measureFormData, {
                headers: { ...measureFormData.getHeaders() },
                timeout: 5 * 60 * 1000
            });
            speakersLoudness = measureRes.data.speakers_lufs || {};
            console.log(`[Extract] Original loudness per speaker:`, JSON.stringify(speakersLoudness));
        } catch (mErr: any) {
            console.warn(`[Extract] Warning measuring speaker loudness: ${mErr.message}`);
        }
    }

    return {
        audioPath: audioOutputPath,
        backgroundAudioPath: finalBackgroundPath,
        vocalsAudioPath: hasVocals ? vocalsAudioPath : null,
        vocalsLufs,
        speakersLoudness,
        segments,
        speakersMetadata,
        videoPath,
        videoDurationSec,
        targetLang: targetLang || 'Spanish',
        voiceMode: voiceMode || 'clone',
        jobId
    };
}, { connection: redisConnection });
