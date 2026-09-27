import { Worker, Job } from 'bullmq';
import { GoogleGenAI, Type } from '@google/genai';
import { redisConnection, GEMINI_API_KEY } from '../config';

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// ---------- Types ----------

interface InputSegment {
    text: string;
    start?: number;
    end?: number;
    start_ms?: number;
    end_ms?: number;
    speaker?: string;
    /** Emitted by the Python service. This is the field that actually exists. */
    speaker_label?: string;
    [key: string]: any;
}

interface TranslateJobData {
    jobId: string;
    targetLang?: string;
    segments: InputSegment[];
}

type IndexedSegment = InputSegment & { _globalIndex: number };

interface SegmentPayload {
    index: number;
    text: string;
    duration_sec: number | null;
    speaker?: string;
}

/** True when a segment carries real speech, as opposed to a music tag or a blank. */
function isSpeakable(text: string): boolean {
    return (text || '').replace(/[^\p{L}\p{N}]/gu, '').length >= 3;
}

interface TranslationResultItem {
    index: number;
    translated_text: string;
}

// ---------- Config ----------

// Verified against official Gemini documentation.
const MODEL_FALLBACK_CHAIN = [
    'gemini-3.7-flash',
    'gemini-3.5-flash',
    'gemini-3.8-flash',
    'gemini-3.1-flash-lite',
    'gemini-flash-latest',
];

const BATCH_SIZE = 40; // Segments per call; avoids truncated output on long videos
const MODEL_RETRY_PAUSE_MS = 1500;

// ---------- Helpers ----------

function chunk<T>(arr: T[], size: number): T[][] {
    const out: T[][] = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
}

function getDurationSec(s: InputSegment): number | null {
    const startSec = s.start !== undefined ? s.start : s.start_ms !== undefined ? s.start_ms / 1000 : null;
    const endSec = s.end !== undefined ? s.end : s.end_ms !== undefined ? s.end_ms / 1000 : null;
    if (startSec === null || endSec === null) return null;
    return Number((endSec - startSec).toFixed(2));
}

function buildSystemInstruction(lang: string): string {
    return `
You are an expert audiovisual localization and dubbing specialist.
Translate transcribed speech segments into natural, spoken ${lang}, optimized for
a text-to-speech (TTS) voice and for audio/video dubbing sync.

You will receive:
- "full_context": the complete original transcript, for context ONLY (do not translate it).
- "segments": the subset of segments to translate now: { index, text, duration_sec, speaker? }.
Use "full_context" to resolve ambiguity, pronouns, topic and tone — never translate a
segment in isolation from the story it belongs to.

CORE DUBBING PRINCIPLES (APPLY REGARDLESS OF TOPIC OR LANGUAGE PAIR):

1. TOPIC & TERMINOLOGY CONSISTENCY
   - Identify the domain (tech, cooking, medical, gaming, finance, legal, vlog...) from
     "full_context" and use the terminology native ${lang} speakers of that field actually use.
   - Disambiguate polysemic words or likely speech-recognition errors using the full story.
   - Once you choose a translation for a recurring term or name, reuse it every time it
     reappears. Do not vary it for style.

2. REGISTER & TONE CONSISTENCY
   - Detect each speaker's formality (casual / neutral / formal) and tone from "full_context".
   - If "speaker" is provided, keep that speaker's formality consistent across all their
     segments (e.g. tú/usted, tu/vous, du/Sie) — don't switch mid-video unless the original
     speaker clearly does.

3. DO NOT TRANSLATE (DNT) & CULTURAL INTEGRITY
   - Keep brand names, trademarks, software/engine names, hardware models, product names,
     usernames, file/code identifiers, and person names unaltered and correctly cased.
     Never translate proper nouns literally.
   - Retain iconic greetings or signature catchphrases in their original form when they are
     the speaker's flair or cultural setting, rather than translating them.

4. ORAL NATURALNESS & TTS-SAFE TEXT
   The output is read aloud verbatim by a TTS engine — nothing unspeakable should reach it.
   - Spell out numbers, percentages, ordinals and fractions in full words in ${lang}.
   - Spell out currency as amount + currency name, regardless of symbol.
   - Spell out dates and times in ${lang} conventions.
   - Spell out units; don't silently convert them.
   - Acronyms: keep as-is if natively pronounced as a word (e.g. "NASA"); otherwise render
     so ${lang} TTS reads it naturally, or expand on first use if ambiguous.
   - URLs, emails, handles, hashtags: render as spoken aloud, never as raw symbols.
   - Avoid literal commas between numbers in a sequence; use natural pauses or "y"/"and".
   - Strip anything unspeakable: emojis, markdown, stray symbols.
   - Translate slang/idioms into natural conversational ${lang} equivalents. Never introduce
     religious, bizarre, or out-of-context phrasing.

5. NON-SPEECH & EDGE SEGMENTS
   - Non-verbal segments (laughter, music tag, inaudible): return unchanged or empty per
     convention — never invent speech for them.
   - Leftover disfluencies: translate as a natural filler in ${lang} rather than silently
     dropping them, unless your pipeline already strips fillers upstream.

6. LENGTH GUIDANCE (NOT A HARD CONSTRAINT)
   - "duration_sec" is the time available in the original audio — a *guide*, not a limit.
     It may be null if timing wasn't available; if so, ignore it and translate naturally.
     Prefer the most natural, idiomatic ${lang} phrasing over a literal one that merely
     counts closer. Do not pad or truncate purely to hit a duration target; a downstream
     step corrects timing with speed adjustment.
   - Only if a natural translation would run dramatically longer/shorter (>40% difference),
     prefer the more concise/expanded natural phrasing available in ${lang}.

7. OPTIONAL EXPRESSIVE DELIVERY
   - You may prepend a short bracketed cue (e.g. "[excited]", "[laughing]", "[whispering]")
     only when the original clearly conveys that emotion. Never fabricate emotion.

OUTPUT: return one object per input segment, matching "index" exactly. Never merge, split,
omit, or reorder segments.
`;
}

// ---------- Gemini call with model fallback ----------

async function callGeminiTranslate(systemInstruction: string, prompt: string): Promise<TranslationResultItem[]> {
    let lastErr: any = null;

    for (const model of MODEL_FALLBACK_CHAIN) {
        try {
            const response = await ai.models.generateContent({
                model,
                contents: prompt,
                config: {
                    systemInstruction,
                    responseMimeType: 'application/json',
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            segments: {
                                type: Type.ARRAY,
                                description: 'One translated entry per input segment, matched by index.',
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        index: { type: Type.INTEGER },
                                        translated_text: { type: Type.STRING },
                                    },
                                    required: ['index', 'translated_text'],
                                },
                            },
                        },
                        required: ['segments'],
                    },
                },
            });

            const raw = response.text || '';
            if (!raw) throw new Error(`Empty response from ${model}`);

            const parsed = JSON.parse(raw);
            if (!parsed || !Array.isArray(parsed.segments)) {
                throw new Error(`Malformed response shape from ${model}`);
            }
            return parsed.segments as TranslationResultItem[];
        } catch (err: any) {
            console.warn(`[Translate] Model ${model} failed: ${err.message}. Trying next...`);
            lastErr = err;
            await new Promise((r) => setTimeout(r, MODEL_RETRY_PAUSE_MS));
        }
    }

    throw lastErr || new Error('All Gemini models failed to return a translation.');
}

// ---------- Worker ----------

export const translateWorker = new Worker(
    'translate',
    async (job: Job<TranslateJobData>) => {
        const { segments, targetLang, jobId } = job.data;
        const lang = targetLang || 'Spanish';

        console.log(`[Translate] Translating ${segments.length} segments to ${lang} for job ${jobId}`);

        // Full context without truncation: current flash models support up to
        // 1M context tokens, so a standard transcript fits entirely and does not
        // lose topic/tone in the second half of long videos.
        const fullContext = segments.map((s) => s.text).join(' ');

        const indexedSegments: IndexedSegment[] = segments.map((s, i) => ({ ...s, _globalIndex: i }));
        const batches = chunk(indexedSegments, BATCH_SIZE);
        const systemInstruction = buildSystemInstruction(lang);

        const allResults: TranslationResultItem[] = [];

        // Sequential by design: each batch is an additional Gemini API call beyond
        // what the worker limiter counts (10 jobs/min). If parallelized in the future,
        // adjust the limiter to reflect actual API call volume, not just jobs.
        for (const [batchIdx, batch] of batches.entries()) {
            const segmentsPayload: SegmentPayload[] = batch.map((s) => ({
                index: s._globalIndex,
                text: s.text,
                duration_sec: getDurationSec(s),
                // The Python service emits `speaker_label`; sending `speaker` alone
                // meant the model never received the speaker field, so instruction #2
                // (keep one speaker's register consistent across their segments) was
                // silently unenforced.
                speaker: s.speaker_label ?? s.speaker,
            }));

            const prompt = `
=== FULL CONTEXT (do not translate, reference only) ===
"${fullContext}"

=== SEGMENTS TO TRANSLATE INTO ${lang.toUpperCase()} (batch ${batchIdx + 1}/${batches.length}) ===
${JSON.stringify(segmentsPayload, null, 2)}
`;

            const batchResults = await callGeminiTranslate(systemInstruction, prompt);
            allResults.push(...batchResults);
        }

        // Strict index validation: if anything is missing, the job FAILS so
        // BullMQ can retry, instead of silently continuing with incomplete data.
        //
        // A blank translation for a segment with real speech is also considered a failure:
        // the TTS worker cannot distinguish it from "no translation" and previously
        // fell back to `seg.text`, dubbing the original language with a translated voice.
        // Non-spoken segments (music tags, silent laughs) are allowed to be blank,
        // hence the `isSpeakable` threshold.
        const byIndex = new Map(allResults.map((r) => [r.index, r.translated_text]));
        const missing: number[] = [];
        const blank: number[] = [];

        const translatedSegments = segments.map((seg, index) => {
            const translated = byIndex.get(index);
            if (translated === undefined) {
                missing.push(index);
            } else if (!translated.trim() && isSpeakable(seg.text)) {
                blank.push(index);
            }
            return { ...seg, translated_text: translated ?? '' };
        });

        if (missing.length > 0) {
            throw new Error(
                `[Translate] job ${jobId}: missing translations for indices [${missing.join(', ')}] out of ${segments.length} segments.`
            );
        }

        if (blank.length > 0) {
            const sample = blank.slice(0, 5).map((i) => `#${i} "${String(segments[i].text).slice(0, 50)}"`).join(' | ');
            throw new Error(
                `[Translate] job ${jobId}: ${blank.length} segment(s) with real speech translated to blank text ` +
                `(${sample}${blank.length > 5 ? ' | ...' : ''}).`
            );
        }

        console.log(
            `[Translate] Translation complete for job ${jobId} (${translatedSegments.length} segments, ${batches.length} batch(es))`
        );
        return { ...job.data, segments: translatedSegments };
    },
    {
        connection: redisConnection,
        limiter: {
            max: 10,
            duration: 60000, // Rate limit for Gemini free tier
        },
    }
);