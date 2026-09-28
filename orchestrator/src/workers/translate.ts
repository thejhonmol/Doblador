import { Worker, Job } from 'bullmq';
import { GoogleGenAI, Type } from '@google/genai';
import { redisConnection, GEMINI_API_KEY } from '../config';
import { normalizeContextDiscovery } from '../pipeline-policy';

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// ---------- Types ----------

interface InputSegment {
    text: string;
    start?: number;
    end?: number;
    start_ms?: number;
    end_ms?: number;
    speaker?: string;
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
    emotion?: string;
}

interface TranslationResultItem {
    index: number;
    translated_text: string;
}

interface ContextDiscovery {
    detectedDomain: string;
    overallTone: string;
    glossaryDNT: Array<{ originalTerm: string; handling: string; targetText: string }>;
    transcriptionCorrections: Array<{ suspiciousText: string; probableOriginal: string; contextNote: string }>;
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

const BATCH_SIZE = 35;
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

/** True when a segment carries real speech, as opposed to a music tag or a blank. */
function isSpeakable(text: string): boolean {
    return (text || '').replace(/[^\p{L}\p{N}]/gu, '').length >= 3;
}

// Generic Gemini caller that walks the MODEL_FALLBACK_CHAIN for any schema
async function callGeminiWithFallback<T>(systemInstruction: string, prompt: string, schema: any): Promise<T> {
    let lastErr: any = null;

    for (const model of MODEL_FALLBACK_CHAIN) {
        try {
            const response = await ai.models.generateContent({
                model,
                contents: prompt,
                config: {
                    systemInstruction,
                    responseMimeType: 'application/json',
                    responseSchema: schema,
                },
            });

            const raw = response.text || '';
            if (!raw) throw new Error(`Empty response from ${model}`);

            const parsed = JSON.parse(raw);
            return parsed as T;
        } catch (err: any) {
            console.warn(`[Translate] Model ${model} encountered an issue: ${err.message}. Trying next fallback...`);
            lastErr = err;
            await new Promise((r) => setTimeout(r, MODEL_RETRY_PAUSE_MS));
        }
    }

    throw lastErr || new Error('All configured Gemini models failed to process request.');
}

// ---------- PHASE 1: Universal Context Analysis & Pre-Translation Glossary ----------

async function analyzeUniversalContext(fullTranscript: string, targetLang: string): Promise<ContextDiscovery> {
    const systemInstruction = `
You are a senior multilingual audiovisual dubbing supervisor and linguist.
Analyze the complete provided transcript across ANY genre or topic (e.g. video games, tech, science, culinary, legal, entertainment, casual vlogging, education).

Your objectives:
1. Identify the topic/domain and communication tone (e.g. formal, educational, comedic, colloquial, energetic).
2. Extract all proper nouns, technical concepts, trademarks, brand names, software/hardware identifiers, or cultural expressions that should either:
   - Remain UNTRANSLATED (Do Not Translate / Keep Original).
   - Be mapped to a standardized equivalent in ${targetLang}.
3. Detect likely Speech Recognition (ASR/Whisper) hallucinations or phonetic misinterpretations (e.g., background noise misheard as words, mumbled greetings wrongly transcribed, broken idioms).
`;

    const schema = {
        type: Type.OBJECT,
        properties: {
            detectedDomain: { type: Type.STRING },
            overallTone: { type: Type.STRING },
            glossaryDNT: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        originalTerm: { type: Type.STRING },
                        handling: { type: Type.STRING, description: 'KEEP_ORIGINAL | TRANSLATE_CONSISTENTLY' },
                        targetText: { type: Type.STRING, description: 'Exact term to use in target language' },
                    },
                    required: ['originalTerm', 'handling', 'targetText'],
                },
            },
            transcriptionCorrections: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        suspiciousText: { type: Type.STRING },
                        probableOriginal: { type: Type.STRING },
                        contextNote: { type: Type.STRING },
                    },
                    required: ['suspiciousText', 'probableOriginal'],
                },
            },
        },
        required: ['detectedDomain', 'glossaryDNT'],
    };

    try {
        const result = await callGeminiWithFallback<ContextDiscovery>(
            systemInstruction,
            `Complete video transcript for analysis:\n"""${fullTranscript}"""`,
            schema
        );
        return normalizeContextDiscovery(result);
    } catch (e) {
        console.warn('[Translate] Phase 1 fallback active: proceeding with neutral defaults');
        return {
            detectedDomain: 'general',
            overallTone: 'natural',
            glossaryDNT: [],
            transcriptionCorrections: [],
        };
    }
}

// ---------- PHASE 2: Translation with Glossary Injection & TTS Rules ----------

function buildDubbingSystemInstruction(lang: string, meta: ContextDiscovery): string {
    const glossaryText = meta.glossaryDNT.length > 0
        ? meta.glossaryDNT.map((g) => `- "${g.originalTerm}" -> use "${g.targetText}" (${g.handling})`).join('\n')
        : 'Follow domain conventions standard for this field.';

    const correctionsText = meta.transcriptionCorrections.length > 0
        ? meta.transcriptionCorrections.map((c) => `- If segment contains "${c.suspiciousText}", the speaker meant "${c.probableOriginal}"`).join('\n')
        : 'None detected.';

    return `
You are an expert audiovisual localization and dubbing specialist translating speech segments into natural, spoken ${lang}.
The output will be fed directly to a Text-to-Speech (TTS) engine.

VIDEO CONTEXT:
- Domain / Subject: ${meta.detectedDomain}
- Tone / Style: ${meta.overallTone}

MANDATORY TERMINOLOGY & DO-NOT-TRANSLATE (DNT):
${glossaryText}

KNOWN ASR/TRANSCRIPTION ANOMALIES TO COMPENSATE:
${correctionsText}

UNIVERSAL DUBBING PRINCIPLES:
1. TTS SAFETY & SPEAKABILITY:
   - Spell out all digits, percentages, measurements, symbols, and currencies in full spoken words in ${lang}.
   - Never output raw symbols, URLs, or markdown.
   - Do not invent, hallucinate, or insert religious, absurd, or out-of-context phrases.
2. RHYTHM & METRIC ADAPTATION:
   - Adapt sentence structure so the syllable count matches the original speech duration as closely as possible.
   - Avoid bloated translations; choose concise, natural spoken phrasing.
3. REGISTER & DIALOGUE CONSISTENCY:
   - If a speaker is identified, maintain their level of formality (e.g. casual vs. formal) across all segments.
   - Match the emotional cadence if provided in the segment metadata.

OUTPUT: Return strictly an array of translated segments preserving the exact index.
`;
}

// ---------- PHASE 3: Audit & Correction of Untranslated Words / Anomalies ----------

interface AuditCorrectionResponse {
    corrections: Array<{
        index: number;
        issueFound: string;
        corrected_text: string;
    }>;
}

async function auditTranslatedSegments(
    segmentsToAudit: Array<{ index: number; original: string; translated: string }>,
    lang: string,
    meta: ContextDiscovery
): Promise<Map<number, string>> {
    const systemInstruction = `
You are a Quality Assurance Dubbing Inspector reviewing translated segments destined for a TTS voice in ${lang}.
Examine the pairs of { original, translated }.

Detect and fix any of the following issues:
1. Untranslated words: common vocabulary mistakenly left in the source language (excluding terms intended to stay original according to the glossary).
2. Hallucinations / Oddities: strange, absurd, religious, or completely out-of-context words injected into the translation.
3. Unspeakable elements: numbers or signs not spelled out in full letters.
4. Broken rhythm: translations that are unnecessarily wordy compared to the original meaning.

If a segment has no issues, do not include it. Return only segments requiring correction.
`;

    const schema = {
        type: Type.OBJECT,
        properties: {
            corrections: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        index: { type: Type.INTEGER },
                        issueFound: { type: Type.STRING },
                        corrected_text: { type: Type.STRING },
                    },
                    required: ['index', 'corrected_text'],
                },
            },
        },
        required: ['corrections'],
    };

    const prompt = `
Domain: ${meta.detectedDomain}
Target Language: ${lang}
Segments to verify:
${JSON.stringify(segmentsToAudit, null, 2)}
`;

    try {
        const result = await callGeminiWithFallback<AuditCorrectionResponse>(systemInstruction, prompt, schema);
        const map = new Map<number, string>();
        if (result?.corrections) {
            for (const item of result.corrections) {
                map.set(item.index, item.corrected_text);
            }
        }
        return map;
    } catch (err: any) {
        console.warn(`[Translate] Phase 3 audit warning: ${err.message}. Keeping current translations.`);
        return new Map();
    }
}

// ---------- Worker ----------

export const translateWorker = new Worker(
    'translate',
    async (job: Job<TranslateJobData>) => {
        const { segments, targetLang, jobId } = job.data;
        const lang = targetLang || 'Spanish';

        console.log(`[Translate] Starting job ${jobId}: ${segments.length} segments to ${lang}`);

        // Full global context without truncation
        const fullContext = segments.map((s) => s.text).join(' ');

        // === PHASE 1: Context discovery, glossary & ASR error detection ===
        console.log(`[Translate] [Job ${jobId}] Phase 1: Analyzing global context & terminology...`);
        const contextMeta = await analyzeUniversalContext(fullContext, lang);
        console.log(`[Translate] [Job ${jobId}] Phase 1 complete: domain="${contextMeta.detectedDomain}", tone="${contextMeta.overallTone}", glossary=${contextMeta.glossaryDNT.length} terms, ASR fixes=${contextMeta.transcriptionCorrections.length}`);

        const indexedSegments: IndexedSegment[] = segments.map((s, i) => ({ ...s, _globalIndex: i }));
        const batches = chunk(indexedSegments, BATCH_SIZE);
        const systemInstruction = buildDubbingSystemInstruction(lang, contextMeta);

        const allResults: TranslationResultItem[] = [];

        // === PHASE 2: Batch translation with glossary & guidelines ===
        console.log(`[Translate] [Job ${jobId}] Phase 2: Translating in ${batches.length} batch(es)...`);
        for (const [batchIdx, batch] of batches.entries()) {
            const segmentsPayload: SegmentPayload[] = batch.map((s) => ({
                index: s._globalIndex,
                text: s.text,
                duration_sec: getDurationSec(s),
                speaker: s.speaker_label ?? s.speaker,
                emotion: s.emotion,
            }));

            const prompt = `
=== FULL CONTEXT FOR REFERENCE ===
"${fullContext}"

=== TRANSLATE BATCH (${batchIdx + 1}/${batches.length}) INTO ${lang.toUpperCase()} ===
${JSON.stringify(segmentsPayload, null, 2)}
`;

            const batchSchema = {
                type: Type.OBJECT,
                properties: {
                    segments: {
                        type: Type.ARRAY,
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
            };

            const batchRes = await callGeminiWithFallback<{ segments: TranslationResultItem[] }>(
                systemInstruction,
                prompt,
                batchSchema
            );

            allResults.push(...batchRes.segments);
            await job.updateProgress(Math.round(((batchIdx + 1) / batches.length) * 80));
        }

        const byIndex = new Map(allResults.map((r) => [r.index, r.translated_text]));

        // === PHASE 3: Audit & correction of anomalies / untranslated words ===
        console.log(`[Translate] [Job ${jobId}] Phase 3: Auditing segments for untranslated words or anomalies...`);
        const auditItems = segments.map((s, idx) => ({
            index: idx,
            original: s.text,
            translated: byIndex.get(idx) || '',
        }));

        const auditBatches = chunk(auditItems, 40);
        for (const aBatch of auditBatches) {
            const corrections = await auditTranslatedSegments(aBatch, lang, contextMeta);
            for (const [idx, correctedText] of corrections.entries()) {
                console.log(`[Translate QA] Segment #${idx} revised: "${byIndex.get(idx)}" -> "${correctedText}"`);
                byIndex.set(idx, correctedText);
            }
        }

        await job.updateProgress(100);

        // === Final integrity validation ===
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
            throw new Error(`[Translate] Job ${jobId}: Missing translations for indices [${missing.join(', ')}]`);
        }

        if (blank.length > 0) {
            const sample = blank.slice(0, 5).map((i) => `#${i} "${String(segments[i].text).slice(0, 50)}"`).join(' | ');
            throw new Error(`[Translate] Job ${jobId}: ${blank.length} speakable segment(s) translated to empty text (${sample}).`);
        }

        console.log(`[Translate] Job ${jobId} finished successfully (${translatedSegments.length} segments). Passing to next stage.`);
        return { ...job.data, segments: translatedSegments };
    },
    {
        connection: redisConnection,
        limiter: {
            // Reduced from 10 to 5 jobs/min to accommodate the ~3x API calls per job
            // (Phase 1 context analysis + Phase 2 translation batches + Phase 3 QA audit).
            // Each job now makes 2 + N_batches + ceil(segments/40) Gemini calls.
            max: 5,
            duration: 60000,
        },
    }
);
