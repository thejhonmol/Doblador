import { Worker, Job } from 'bullmq';
import { redisConnection, FISH_AUDIO_API_KEY, FISH_AUDIO_API_KEYS, PYTHON_SERVICES_URL } from '../config';
import { stitchSegments } from './stitch';
import { persistSpeakers, persistSegments, syncVoicesCatalog } from '../persist';
import axios from 'axios';
import fs from 'fs';
import path from 'path';
import FormData from 'form-data';

// Default reference voice — change this to match your preferred voice from Fish Audio catalog
const DEFAULT_REFERENCE_ID = '7033e0e6d35e404d81a100701ceca41b';

/**
 * Worker slots are independent of the number of API keys.
 *
 * Concurrency used to be `min(activeKeys.length, segments.length)`, so a single-key
 * setup ran strictly one request at a time and a 3-key setup capped at 3 even though
 * Fish Audio tolerates far more. Keys are now spread across a fixed pool of slots;
 * each slot takes its next key round-robin and backs off independently on 429.
 */
const MAX_CONCURRENCY = 6;
const MAX_ATTEMPTS_PER_SEGMENT = 4;
const BASE_BACKOFF_MS = 2000;
const MAX_BACKOFF_MS = 30000;
const RATE_LIMIT_STATUS = new Set([429, 500, 502, 503, 504]);

// Parse comma-separated voice IDs from environment variable
function parseVoicePool(envVal?: string, fallback: string[] = []): string[] {
    if (!envVal) return fallback;
    const list = envVal.split(',').map(s => s.trim()).filter(s => isValidVoiceId(s));
    return list.length > 0 ? list : fallback;
}

interface PoolVoice {
    id: string;
    label: string;
}

// Voice Pools by AI Profile (Gender & Age)
export const VOICE_POOLS: Record<string, PoolVoice[]> = {
    male_young: parseVoicePool(process.env.FISH_VOICE_POOL_MALE_YOUNG, [
        '7033e0e6d35e404d81a100701ceca41b', // Voz natural hombre
        'dfa5b230c8054f429e434f4a6e9bbdec', // Farid Dieck
        '35199d5438854f5d9157c500479ab684', // Narrador v2
    ]).map((id, i) => ({ id, label: ['Voz natural hombre', 'Farid Dieck', 'Narrador v2'][i] || id })),
    male_mature: parseVoicePool(process.env.FISH_VOICE_POOL_MALE_MATURE, [
        '47a7c0605b6a4a658acc1fb85df19444', // Waldemaro Martínez
        '8d2c17a9b26d4d83888ea67a1ee565b2', // Valentino
        'bcdc2d4b044d4a6992d9260ce16715eb', // Loquendo Documental
    ]).map((id, i) => ({ id, label: ['Waldemaro Martínez', 'Valentino', 'Loquendo Documental'][i] || id })),
    female_young: parseVoicePool(process.env.FISH_VOICE_POOL_FEMALE_YOUNG, [
        '8d75acf7b75b470ba5d975191bebf5f3', // Voz natural eleven
        '35929683c49c4ec0bf779dc07d22620b', // Chica
        'bfed5c0810a347dbb62e8ccce7f59c48', // Voz Femenina Español
    ]).map((id, i) => ({ id, label: ['Voz natural eleven', 'Chica', 'Voz Femenina Español'][i] || id })),
    female_mature: parseVoicePool(process.env.FISH_VOICE_POOL_FEMALE_MATURE, [
        '26ff45fab722431c85eea2536e5c5197', // Idea Vilariño
        '692eb1e1023242219dc8caae8c56fb12', // Verity
    ]).map((id, i) => ({ id, label: ['Idea Vilariño', 'Verity'][i] || id })),
    child: parseVoicePool(process.env.FISH_VOICE_POOL_CHILD, [
        '541dad35db004c03a11e33c439fad693', // Niño
        'a00a2848fef646fdacfc21830af27d2c', // Madoka
    ]).map((id, i) => ({ id, label: ['Niño', 'Madoka'][i] || id })),
    default: [{ id: DEFAULT_REFERENCE_ID, label: 'Default' }],
};

function isValidVoiceId(id?: string): id is string {
    if (!id) return false;
    if (id.includes('tu_') || id.includes('aqui') || id.includes('your_') || id.length < 10) return false;
    return true;
}

// Allocates a distinct, consistent voice ID for each speaker in the video
function allocateSpeakerVoices(uniqueSpeakers: string[], speakersMetadata: any = {}): Record<string, string> {
    const speakerVoiceMap: Record<string, string> = {};
    const categoryUsage: Record<string, number> = {};

    for (const label of uniqueSpeakers) {
        // 1. Determine demographic category
        const meta = speakersMetadata[label] || {};
        const gender = (meta.gender || '').toLowerCase();
        const age = typeof meta.age === 'number' ? meta.age : undefined;

        let category = 'male_young';
        if (gender === 'child' || (age !== undefined && age < 14)) {
            category = 'child';
        } else if (gender === 'female') {
            category = (age !== undefined && age >= 45) ? 'female_mature' : 'female_young';
        } else if (gender === 'male') {
            category = (age !== undefined && age >= 45) ? 'male_mature' : 'male_young';
        } else {
            category = 'male_young';
        }

        // 2. Select next voice from the category's pool
        const pool = VOICE_POOLS[category] || VOICE_POOLS.default;
        const currentIdx = categoryUsage[category] || 0;
        const selectedVoice = pool[currentIdx % pool.length].id;
        categoryUsage[category] = currentIdx + 1;

        speakerVoiceMap[label] = selectedVoice;
        console.log(`[TTS Voice Allocation] ${label} (${gender || 'unknown'}, ~${age || '?'}y -> ${category}) allocated pool voice ${selectedVoice} (slot ${(currentIdx % pool.length) + 1}/${pool.length})`);
    }

    return speakerVoiceMap;
}

// Generate a silent WAV as fallback if Fish Audio fails for a segment
function createSilentWav(filePath: string, durationSec: number) {
    const sampleRate = 16000;
    const numChannels = 1;
    const bitsPerSample = 16;
    const numSamples = Math.floor(sampleRate * durationSec);
    const dataSize = numSamples * numChannels * (bitsPerSample / 8);
    const buffer = Buffer.alloc(44 + dataSize);

    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + dataSize, 4);
    buffer.write('WAVE', 8);
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16);
    buffer.writeUInt16LE(1, 20);
    buffer.writeUInt16LE(numChannels, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(sampleRate * numChannels * (bitsPerSample / 8), 28);
    buffer.writeUInt16LE(numChannels * (bitsPerSample / 8), 32);
    buffer.writeUInt16LE(bitsPerSample, 34);
    buffer.write('data', 36);
    buffer.writeUInt32LE(dataSize, 40);

    fs.writeFileSync(filePath, buffer);
}

class HttpError extends Error {
    constructor(message: string, readonly status?: number) {
        super(message);
    }
}

async function generateSpeech(text: string, referenceId: string, outputPath: string, apiKey: string): Promise<void> {
    let response;
    try {
        response = await axios.post('https://api.fish.audio/v1/tts', {
            text,
            reference_id: referenceId,
            format: 'mp3',
        }, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'model': 's2.1-pro-free',
            },
            responseType: 'arraybuffer',
            timeout: 60000,
        });
    } catch (err: any) {
        throw new HttpError(err.response?.data?.detail || err.message, err.response?.status);
    }

    const body = Buffer.from(response.data);
    // A 200 with an empty or near-empty body is a silent failure: the file would be
    // written and treated as a success, leaving a hole in the dubbed track.
    if (body.length < 512) {
        throw new HttpError(`Empty audio payload (${body.length} bytes)`, 200);
    }
    fs.writeFileSync(outputPath, body);
}

/**
 * Tries every configured key, in rotation, with exponential backoff.
 *
 * The previous failover loop did `if (k === workerSlot % keys.length) continue`,
 * which skipped every key when only one was configured — i.e. the most common setup
 * had literally zero retries. It also had no backoff, so a 429 storm was hammered
 * immediately and every key failed together.
 */
async function generateWithFailover(
    text: string,
    referenceId: string,
    outputPath: string,
    activeKeys: string[],
    startSlot: number,
    segLabel: string
): Promise<boolean> {
    for (let attempt = 1; attempt <= MAX_ATTEMPTS_PER_SEGMENT; attempt++) {
        // Rotate the starting key so load spreads across all of them.
        const keyIndex = (startSlot + attempt - 1) % activeKeys.length;
        const key = activeKeys[keyIndex];
        try {
            await generateSpeech(text, referenceId, outputPath, key);
            return true;
        } catch (err: any) {
            const status = err instanceof HttpError ? err.status : undefined;
            const lastAttempt = attempt === MAX_ATTEMPTS_PER_SEGMENT;
            const retryable = status === undefined || RATE_LIMIT_STATUS.has(status);
            console.warn(
                `[TTS] ${segLabel} attempt ${attempt}/${MAX_ATTEMPTS_PER_SEGMENT} ` +
                `failed on key #${keyIndex + 1}${status ? ` (HTTP ${status})` : ''}: ${err.message}` +
                (retryable ? '' : ' (not retryable, moving to the next key)')
            );
            if (lastAttempt || !retryable) break;
            // Jitter so parallel slots do not retry in lockstep.
            const backoff = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * Math.pow(2, attempt - 1));
            await new Promise((r) => setTimeout(r, backoff + Math.floor(Math.random() * 500)));
        }
    }
    return false;
}

export const ttsWorker = new Worker('tts', async (job: Job) => {
    const { segments, jobId, videoPath, speakersMetadata } = job.data;
    console.log(`[TTS] Generating audio for ${segments.length} segments for job ${jobId}`);

    const activeKeys = FISH_AUDIO_API_KEYS.length > 0 ? FISH_AUDIO_API_KEYS : (FISH_AUDIO_API_KEY ? [FISH_AUDIO_API_KEY] : []);
    if (activeKeys.length === 0) {
        throw new Error('No valid FISH_AUDIO_API_KEY configured. Cannot generate TTS audio.');
    }

    const outputDir = path.join(path.dirname(videoPath), `${jobId}_tts_output`);
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    // Determine unique speakers and allocate distinct consistent voices from the pools
    const uniqueSpeakers: string[] = Array.from(new Set(segments.map((s: any) => s.speaker_label || 'SPEAKER_00')));
    console.log(`[TTS] Unique speakers detected in job ${jobId}: ${uniqueSpeakers.join(', ')}`);
    const speakerVoiceMap = allocateSpeakerVoices(uniqueSpeakers, speakersMetadata);

    // Keep the DB catalogue in sync with the pools this worker actually uses.
    await syncVoicesCatalog(VOICE_POOLS).catch((e: any) =>
        console.warn(`[TTS] Warning: could not sync voices_catalog: ${e.message}`)
    );

    const concurrency = Math.max(1, Math.min(MAX_CONCURRENCY, segments.length || 1));
    console.log(`[TTS] Synthesizing ${segments.length} segments with ${concurrency} slot(s) over ${activeKeys.length} API key(s)...`);

    const finalSegments = new Array(segments.length);
    let nextIndex = 0;
    let silentSegments = 0;

    async function workerTask(workerSlot: number) {
        while (true) {
            const i = nextIndex++;
            if (i >= segments.length) break;

            const seg = segments[i];
            const label = seg.speaker_label || 'SPEAKER_00';
            const referenceId = isValidVoiceId(seg.fish_reference_id)
                ? seg.fish_reference_id
                : (speakerVoiceMap[label] || DEFAULT_REFERENCE_ID);
            const rawDurationMs = seg.end_ms - seg.start_ms;
            const targetMs = Math.max(200, rawDurationMs || 500);
            const segLabel = `Seg ${i + 1}/${segments.length}`;

            // An empty translation means the batch lost it, or the model returned a
            // blank for real speech. Falling back to `seg.text` dubbed the ORIGINAL
            // language with a Spanish voice — audible, and it looked like a lip-sync
            // problem rather than a data problem. Silence plus a loud log is honest.
            const translated = typeof seg.translated_text === 'string' ? seg.translated_text.trim() : '';
            if (!translated) {
                console.error(
                    `[TTS] ${segLabel} has no translation (source: "${String(seg.text || '').slice(0, 60)}"). ` +
                    'Emitting silence instead of dubbing the source language.'
                );
                const fallbackPath = path.join(outputDir, `seg_${i}.wav`);
                createSilentWav(fallbackPath, targetMs / 1000);
                finalSegments[i] = {
                    ...seg,
                    tts_audio_url: fallbackPath,
                    generated_ms: targetMs,
                    speed_used: 1.0,
                    tts_failed: true,
                    fish_reference_id: referenceId,
                };
                silentSegments++;
                continue;
            }

            const audioPath = path.join(outputDir, `seg_${i}.mp3`);

            const generated = await generateWithFailover(
                translated,
                referenceId,
                audioPath,
                activeKeys,
                workerSlot,
                segLabel
            );

            if (generated) {
                finalSegments[i] = {
                    ...seg,
                    translated_text: translated,
                    tts_audio_url: audioPath,
                    speed_used: 1.0,
                    tts_failed: false,
                    fish_reference_id: referenceId,
                };
            } else {
                console.error(`[TTS] ${segLabel} failed on all ${activeKeys.length} key(s) after ${MAX_ATTEMPTS_PER_SEGMENT} attempts. Using silence.`);
                const fallbackPath = path.join(outputDir, `seg_${i}.wav`);
                createSilentWav(fallbackPath, targetMs / 1000);
                finalSegments[i] = {
                    ...seg,
                    tts_audio_url: fallbackPath,
                    speed_used: 1.0,
                    tts_failed: true,
                    fish_reference_id: referenceId,
                };
                silentSegments++;
            }

            // Small delay between calls on this slot to reduce rate throttling.
            await new Promise(r => setTimeout(r, 120));
        }
    }

    // Launch workers in parallel
    const workers = [];
    for (let w = 0; w < concurrency; w++) {
        workers.push(workerTask(w));
    }
    await Promise.all(workers);

    console.log(`[TTS] Synthesis complete for job ${jobId}. ${finalSegments.length - silentSegments}/${finalSegments.length} segments voiced, ${silentSegments} silent.`);
    if (silentSegments > 0) {
        console.warn(`[TTS] ${silentSegments} segment(s) have no usable audio. They will be gaps in the dubbed track.`);
    }

    // Persist the speaker->voice decision so it is auditable and reusable.
    await persistSpeakers(jobId, speakerVoiceMap, job.data.targetLang || 'Spanish')
        .catch((e: any) => console.warn(`[TTS] Warning: could not persist speakers: ${e.message}`));

    // Match loudness to original vocals using pyloudnorm
    const targetLufs = job.data.vocalsLufs || -18.0;
    const speakersLufs = job.data.speakersLoudness || {};
    let generatedDurations: Record<string, number> = {};
    try {
        console.log(`[TTS] Normalizing generated speech with pyloudnorm to match original volume (${targetLufs} LUFS)...`);
        const normFormData = new FormData();
        normFormData.append('output_dir', outputDir);
        normFormData.append('target_lufs', targetLufs.toString());
        normFormData.append('speakers_lufs_json', JSON.stringify(speakersLufs));
        normFormData.append('segments_json', JSON.stringify(finalSegments));

        const normRes = await axios.post(`${PYTHON_SERVICES_URL}/normalize-tts`, normFormData, {
            headers: { ...normFormData.getHeaders() },
            timeout: 10 * 60 * 1000
        });
        generatedDurations = normRes.data.durations_sec || {};
        console.log(`[TTS] pyloudnorm successfully calibrated ${normRes.data.adjusted_count} segments to original voice level!`);
    } catch (normErr: any) {
        console.warn(`[TTS] Warning: pyloudnorm normalization error: ${normErr.message}. Continuing with raw audio.`);
    }

    // Record the real measured duration per segment instead of the target slot, so
    // `generated_ms` in the DB reflects what was actually produced.
    const withDurations = finalSegments.map((seg: any, i: number) => {
        const base = seg ? path.basename(seg.tts_audio_url) : null;
        const measured = base ? generatedDurations[base] : undefined;
        return seg && measured !== undefined
            ? { ...seg, generated_ms: Math.round(measured * 1000) }
            : seg;
    });

    await persistSegments(jobId, withDurations)
        .catch((e: any) => console.warn(`[TTS] Warning: could not persist segments: ${e.message}`));

    // Pre-stitch all segments into a single continuous audio track to avoid FFmpeg
    // command-line limits on Windows. The buffer length comes from the source video
    // duration, never from the last segment: sizing it from the speech alone made
    // the muxed video shorter than the source whenever the tail had no dialogue.
    const fullTtsAudioPath = await stitchSegments(
        withDurations,
        videoPath,
        jobId,
        job.data.videoDurationSec,
        '[TTS]'
    );

    return { ...job.data, segments: withDurations, fullTtsAudioPath };
}, { connection: redisConnection });
