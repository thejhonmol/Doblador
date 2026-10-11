// Status 200 is used internally when the provider returns a successful but empty
// audio body; that response is safe to retry.
export const TRANSIENT_TTS_STATUS = new Set([200, 429, 500, 502, 503, 504]);
export const KEY_LOCAL_TTS_STATUS = new Set([401, 402, 403]);

/**
 * A failed credential must not prevent another configured credential from being
 * attempted. Request-level 4xx responses (for example an invalid payload) are
 * terminal because sending the same payload with another key cannot fix them.
 */
export function ttsFailureAction(status?: number): 'retry' | 'next-key' | 'stop' {
    if (status === undefined || TRANSIENT_TTS_STATUS.has(status)) return 'retry';
    if (KEY_LOCAL_TTS_STATUS.has(status)) return 'next-key';
    return 'stop';
}

/** Prevents an apparently successful deliverable containing missing dialogue. */
export function assertSpeechSynthesisComplete(
    failedSpeechSegments: number,
    lastError = '',
    lastStatus?: number
): void {
    if (failedSpeechSegments <= 0) return;
    const isCreditError = lastStatus === 402 || /credit/i.test(lastError);
    const reason = isCreditError
        ? 'Fish Audio API credit balance is exhausted (HTTP 402). API credit is managed independently from platform credit. Please add funds at https://fish.audio/app/developers or update FISH_AUDIO_API_KEY in .env.'
        : (lastError || 'One or more speech segments failed to generate.');
    throw new Error(
        `TTS synthesis incomplete: ${failedSpeechSegments} dialogue segment(s) have no usable audio. ` +
        `The final video was not assembled. ${reason}`
    );
}

export interface NormalizedContextDiscovery {
    detectedDomain: string;
    overallTone: string;
    glossaryDNT: Array<{ originalTerm: string; handling: string; targetText: string }>;
    transcriptionCorrections: Array<{
        suspiciousText: string;
        probableOriginal: string;
        contextNote: string;
    }>;
}

/** Normalizes optional model fields before the translation prompt reads them. */
export function normalizeContextDiscovery(value: unknown): NormalizedContextDiscovery {
    const raw = value && typeof value === 'object' ? value as Record<string, unknown> : {};
    return {
        detectedDomain: typeof raw.detectedDomain === 'string' && raw.detectedDomain.trim()
            ? raw.detectedDomain.trim()
            : 'general',
        overallTone: typeof raw.overallTone === 'string' && raw.overallTone.trim()
            ? raw.overallTone.trim()
            : 'natural',
        glossaryDNT: Array.isArray(raw.glossaryDNT) ? raw.glossaryDNT as NormalizedContextDiscovery['glossaryDNT'] : [],
        transcriptionCorrections: Array.isArray(raw.transcriptionCorrections)
            ? raw.transcriptionCorrections as NormalizedContextDiscovery['transcriptionCorrections']
            : [],
    };
}

// Initial translation guidance only; these are not TTS acceptance limits.
const SYLLABLE_BUDGET_MIN_RATIO = 0.75;
const SYLLABLE_BUDGET_MAX_RATIO = 1.08;

export type DurationAction = 'accept' | 'compress' | 'fail';

/**
 * Short clips keep their natural pause. Every over-long clip is fitted by the
 * stitcher, without rewriting dialogue, regenerating TTS or imposing a speed cap.
 * Only an invalid measurement is a failure.
 */
export function durationAction(ratio: number): DurationAction {
    if (!Number.isFinite(ratio) || ratio <= 0) return 'fail';
    return ratio <= 1 ? 'accept' : 'compress';
}

const SYLLABLES_PER_SECOND: Record<string, number> = {
    spanish: 4.7,
    english: 4.4,
    french: 4.5,
    german: 4.1,
    italian: 4.7,
    portuguese: 4.6,
};

/** Gives Gemini an explicit spoken-length budget rather than a vague "be concise". */
export function syllableBudget(durationSec: number | null, language = ''): {
    target: number | null;
    min: number | null;
    max: number | null;
} {
    if (durationSec === null || !Number.isFinite(durationSec) || durationSec <= 0) {
        return { target: null, min: null, max: null };
    }
    const rate = SYLLABLES_PER_SECOND[language.trim().toLowerCase()] || 4.5;
    const target = Math.max(1, Math.round(durationSec * rate));
    return {
        target,
        min: Math.max(1, Math.floor(target * SYLLABLE_BUDGET_MIN_RATIO)),
        max: Math.max(1, Math.ceil(target * SYLLABLE_BUDGET_MAX_RATIO)),
    };
}

export const PIPELINE_STAGES = ['extract', 'translate', 'tts', 'assemble'] as const;
export type PipelineStage = typeof PIPELINE_STAGES[number];

/** Returns the first stage whose durable DB record is not completed. */
export function nextIncompleteStage(completedStages: Iterable<string>): PipelineStage | null {
    const completed = new Set(completedStages);
    return PIPELINE_STAGES.find((stage) => !completed.has(stage)) || null;
}
