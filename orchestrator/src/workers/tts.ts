import { Worker, Job } from 'bullmq';
import { redisConnection, FISH_AUDIO_API_KEY, FISH_AUDIO_API_KEYS, PYTHON_SERVICES_URL } from '../config';
import { stitchSegments } from './stitch';
import { persistSpeakers, persistSegments, syncVoicesCatalog } from '../persist';
import axios from 'axios';
import fs from 'fs';
import path from 'path';
import FormData from 'form-data';
import ffmpeg from 'fluent-ffmpeg';
import { encode } from '@msgpack/msgpack';
import { assertSpeechSynthesisComplete, ttsFailureAction } from '../pipeline-policy';

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

interface VoiceTarget {
    mode: 'clone' | 'preset';
    referenceId?: string;
    referenceAudio?: Buffer;
    referenceText?: string;
}

interface SpeakerCloneRef {
    audio: Buffer;
    text: string;
    durationSec: number;
}

/**
 * Extracts a clean speech slice of a speaker from the isolated vocals track
 * to use as an in-flight reference for zero-shot cloning in Fish Audio.
 */
async function extractSpeakerReference(
    vocalsPath: string,
    startMs: number,
    endMs: number,
    outPath: string
): Promise<Buffer | null> {
    const startSec = (startMs / 1000).toFixed(3);
    const durSec = Math.max(0.5, (endMs - startMs) / 1000).toFixed(3);

    return new Promise((resolve) => {
        ffmpeg(vocalsPath)
            .setStartTime(startSec)
            .setDuration(durSec)
            .audioFrequency(44100)
            .audioChannels(1)
            .audioBitrate('128k')
            .format('mp3')
            .on('end', () => {
                try {
                    if (fs.existsSync(outPath) && fs.statSync(outPath).size > 1000) {
                        resolve(fs.readFileSync(outPath));
                    } else {
                        resolve(null);
                    }
                } catch {
                    resolve(null);
                }
            })
            .on('error', (err) => {
                console.warn(`[TTS Clone] Failed extracting reference audio: ${err.message}`);
                resolve(null);
            })
            .save(outPath);
    });
}

/**
 * Finds the cleanest and most representative speech segment (between 3.5s and 10s)
 * for each speaker in the video and extracts its audio slice for inline cloning.
 */
async function prepareSpeakerCloneReferences(
    uniqueSpeakers: string[],
    segments: any[],
    vocalsAudioPath: string | null,
    tempDir: string
): Promise<Record<string, SpeakerCloneRef>> {
    const cloneRefs: Record<string, SpeakerCloneRef> = {};
    if (!vocalsAudioPath || !fs.existsSync(vocalsAudioPath)) {
        console.warn('[TTS Clone] Vocals audio path not available. Inline cloning will fallback to catalog voices.');
        return cloneRefs;
    }

    for (const spk of uniqueSpeakers) {
        const spkSegs = segments.filter(
            (s) => (s.speaker_label || 'SPEAKER_00') === spk &&
                   typeof s.text === 'string' &&
                   s.text.trim().length >= 8 &&
                   s.end_ms > s.start_ms
        );

        if (spkSegs.length === 0) continue;

        // Prefer segments between 3.5s and 10s (optimal for zero-shot prompt conditioning)
        let bestSeg = spkSegs.find((s) => {
            const dur = (s.end_ms - s.start_ms) / 1000;
            return dur >= 3.5 && dur <= 10.0;
        });

        // Otherwise pick the longest available segment >= 1.5s
        if (!bestSeg) {
            const sortedByDur = [...spkSegs].sort(
                (a, b) => (b.end_ms - b.start_ms) - (a.end_ms - a.start_ms)
            );
            if (sortedByDur.length > 0 && (sortedByDur[0].end_ms - sortedByDur[0].start_ms) >= 1500) {
                bestSeg = sortedByDur[0];
            }
        }

        if (!bestSeg) {
            console.warn(`[TTS Clone] Speaker ${spk} has no segments >= 1.5s. Will fallback to preset catalog voice.`);
            continue;
        }

        const outRefPath = path.join(tempDir, `ref_${spk}.mp3`);
        const audioBuf = await extractSpeakerReference(
            vocalsAudioPath,
            bestSeg.start_ms,
            bestSeg.end_ms,
            outRefPath
        );

        if (audioBuf && audioBuf.length > 1000) {
            const dur = ((bestSeg.end_ms - bestSeg.start_ms) / 1000).toFixed(1);
            console.log(`[TTS Clone] Extracted reference audio for ${spk} (${dur}s, ${audioBuf.length} bytes, text: "${bestSeg.text.slice(0, 50)}...")`);
            cloneRefs[spk] = {
                audio: audioBuf,
                text: bestSeg.text,
                durationSec: Number(dur),
            };
        }
    }

    return cloneRefs;
}

async function generateSpeech(
    text: string,
    target: VoiceTarget,
    outputPath: string,
    apiKey: string
): Promise<void> {
    let response;
    try {
        if (target.mode === 'clone' && target.referenceAudio && target.referenceAudio.length > 0) {
            const payload = {
                text,
                model: 's2.1-pro-free',
                format: 'mp3',
                references: [
                    {
                        audio: target.referenceAudio,
                        text: target.referenceText || '',
                    }
                ]
            };
            const packed = Buffer.from(encode(payload));
            response = await axios.post('https://api.fish.audio/v1/tts', packed, {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/msgpack',
                },
                responseType: 'arraybuffer',
                timeout: 60000,
            });
        } else {
            response = await axios.post('https://api.fish.audio/v1/tts', {
                text,
                reference_id: target.referenceId || DEFAULT_REFERENCE_ID,
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
        }
    } catch (err: any) {
        let msg = err.message;
        if (err.response?.data) {
            try {
                const text = Buffer.isBuffer(err.response.data)
                    ? err.response.data.toString('utf-8')
                    : (typeof err.response.data === 'string' ? err.response.data : JSON.stringify(err.response.data));
                const parsed = JSON.parse(text);
                msg = parsed.message || parsed.detail || text;
            } catch {
                if (typeof err.response.data === 'string') msg = err.response.data;
            }
        }
        throw new HttpError(msg, err.response?.status);
    }

    const body = Buffer.from(response.data);
    if (body.length < 512) {
        throw new HttpError(`Empty audio payload (${body.length} bytes)`, 200);
    }
    fs.writeFileSync(outputPath, body);
}

/**
 * Tries every configured key, in rotation, with exponential backoff.
 */
async function generateWithFailover(
    text: string,
    target: VoiceTarget,
    outputPath: string,
    activeKeys: string[],
    startSlot: number,
    segLabel: string,
    lastErrorRef?: { message: string; status?: number }
): Promise<boolean> {
    // Always leave enough attempts to visit every configured key at least once.
    const maxAttempts = Math.max(MAX_ATTEMPTS_PER_SEGMENT, activeKeys.length);
    const unusableKeys = new Set<number>();
    let keyCursor = startSlot;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        let keyIndex = keyCursor % activeKeys.length;
        let inspected = 0;
        while (unusableKeys.has(keyIndex) && inspected < activeKeys.length) {
            keyCursor++;
            keyIndex = keyCursor % activeKeys.length;
            inspected++;
        }
        if (inspected >= activeKeys.length) break;
        keyCursor = keyIndex + 1;
        const key = activeKeys[keyIndex];
        try {
            await generateSpeech(text, target, outputPath, key);
            return true;
        } catch (err: any) {
            const status = err instanceof HttpError ? err.status : undefined;
            if (lastErrorRef) {
                lastErrorRef.message = err.message;
                lastErrorRef.status = status;
            }
            const lastAttempt = attempt === maxAttempts;
            const action = ttsFailureAction(status);
            console.warn(
                `[TTS] ${segLabel} attempt ${attempt}/${maxAttempts} ` +
                `failed on key #${keyIndex + 1}${status ? ` (HTTP ${status})` : ''}: ${err.message}` +
                (action === 'next-key' ? ' (credential-specific failure; trying the next key)' : '') +
                (action === 'stop' ? ' (request-level failure; stopping retries)' : '')
            );
            if (lastAttempt || action === 'stop') break;
            if (action === 'next-key') {
                unusableKeys.add(keyIndex);
                if (unusableKeys.size === activeKeys.length) break;
            }
            if (action === 'retry') {
                const backoff = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * Math.pow(2, attempt - 1));
                await new Promise((r) => setTimeout(r, backoff + Math.floor(Math.random() * 500)));
            }
        }
    }
    return false;
}

export const ttsWorker = new Worker('tts', async (job: Job) => {
    const { segments, jobId, videoPath, vocalsAudioPath, speakersMetadata, targetLang, voiceMode } = job.data;
    const mode = voiceMode === 'preset' ? 'preset' : 'clone';
    console.log(`[TTS] Generating audio for ${segments.length} segments for job ${jobId} (voiceMode: ${mode})`);

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

    let speakerCloneRefs: Record<string, SpeakerCloneRef> = {};
    if (mode === 'clone') {
        speakerCloneRefs = await prepareSpeakerCloneReferences(uniqueSpeakers, segments, vocalsAudioPath, outputDir);
    }

    // Build VoiceTarget for each speaker
    const speakerTargetMap: Record<string, VoiceTarget> = {};
    const recordedSpeakerVoiceMap: Record<string, string> = {};

    for (const label of uniqueSpeakers) {
        if (mode === 'clone' && speakerCloneRefs[label]) {
            speakerTargetMap[label] = {
                mode: 'clone',
                referenceAudio: speakerCloneRefs[label].audio,
                referenceText: speakerCloneRefs[label].text,
            };
            recordedSpeakerVoiceMap[label] = 'cloned_inline';
            console.log(`[TTS] Speaker ${label} will be CLONED dynamically from original actor vocals.`);
        } else {
            const fallbackVoice = speakerVoiceMap[label] || DEFAULT_REFERENCE_ID;
            speakerTargetMap[label] = {
                mode: 'preset',
                referenceId: fallbackVoice,
            };
            recordedSpeakerVoiceMap[label] = fallbackVoice;
            console.log(`[TTS] Speaker ${label} will use catalog voice ${fallbackVoice}.`);
        }
    }

    // Keep the DB catalogue in sync with the pools this worker actually uses.
    await syncVoicesCatalog(VOICE_POOLS).catch((e: any) =>
        console.warn(`[TTS] Warning: could not sync voices_catalog: ${e.message}`)
    );

    const concurrency = Math.max(1, Math.min(MAX_CONCURRENCY, segments.length || 1));
    console.log(`[TTS] Synthesizing ${segments.length} segments with ${concurrency} slot(s) over ${activeKeys.length} API key(s)...`);

    const finalSegments = new Array(segments.length);
    let nextIndex = 0;
    let silentSegments = 0;
    let failedSpeechSegments = 0;
    const sharedErrorRef = { message: '', status: undefined as number | undefined };

    async function workerTask(workerSlot: number) {
        while (true) {
            const i = nextIndex++;
            if (i >= segments.length) break;

            const seg = segments[i];
            const label = seg.speaker_label || 'SPEAKER_00';
            const target = speakerTargetMap[label] || { mode: 'preset', referenceId: DEFAULT_REFERENCE_ID };
            const referenceIdForSeg = target.mode === 'clone' ? 'cloned_inline' : (target.referenceId || DEFAULT_REFERENCE_ID);
            const rawDurationMs = seg.end_ms - seg.start_ms;
            const targetMs = Math.max(200, rawDurationMs || 500);
            const segLabel = `Seg ${i + 1}/${segments.length} (${target.mode === 'clone' ? 'cloned' : 'preset'})`;

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
                    fish_reference_id: referenceIdForSeg,
                };
                silentSegments++;
                // Translation guarantees text for speakable source segments. If it
                // is missing here, do not let a dialogue gap reach the final video.
                if (String(seg.text || '').replace(/[^\p{L}\p{N}]/gu, '').length >= 3) {
                    failedSpeechSegments++;
                }
                continue;
            }

            const audioPath = path.join(outputDir, `seg_${i}.mp3`);

            const generated = await generateWithFailover(
                translated,
                target,
                audioPath,
                activeKeys,
                workerSlot,
                segLabel,
                sharedErrorRef
            );

            if (generated) {
                finalSegments[i] = {
                    ...seg,
                    translated_text: translated,
                    tts_audio_url: audioPath,
                    speed_used: 1.0,
                    tts_failed: false,
                    fish_reference_id: referenceIdForSeg,
                };
            } else {
                console.error(`[TTS] ${segLabel} failed across all usable credentials. Using temporary silence; the stage will be rejected after synthesis.`);
                const fallbackPath = path.join(outputDir, `seg_${i}.wav`);
                createSilentWav(fallbackPath, targetMs / 1000);
                finalSegments[i] = {
                    ...seg,
                    tts_audio_url: fallbackPath,
                    speed_used: 1.0,
                    tts_failed: true,
                    fish_reference_id: referenceIdForSeg,
                };
                silentSegments++;
                failedSpeechSegments++;
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

    const voicedSegments = finalSegments.length - silentSegments;
    console.log(`[TTS] Synthesis complete for job ${jobId}. ${voicedSegments}/${finalSegments.length} segments voiced, ${silentSegments} silent.`);

    assertSpeechSynthesisComplete(failedSpeechSegments, sharedErrorRef.message, sharedErrorRef.status);

    if (silentSegments > 0) {
        console.warn(`[TTS] ${silentSegments} segment(s) have no usable audio. They will be gaps in the dubbed track.`);
    }

    // Persist the speaker->voice decision so it is auditable and reusable.
    await persistSpeakers(jobId, recordedSpeakerVoiceMap, targetLang || 'Spanish', mode)
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
