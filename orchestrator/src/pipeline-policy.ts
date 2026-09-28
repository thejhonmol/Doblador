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

// Shared timing policy. Translation, TTS validation and the stitcher must agree on
// these limits or a segment accepted by one stage can be rejected by the next.
export const DURATION_RATIO_MIN = 0.75;
export const DURATION_RATIO_ACCEPT_MAX = 1.08;
export const DURATION_RATIO_REWRITE_MAX = 1.18;
export const DURATION_RATIO_HARD_MAX = 1.25;
export const MAX_SYNC_DRIFT_MS = 900;

export type DurationAction = 'accept' | 'compress' | 'rewrite-shorter' | 'rewrite-longer' | 'fail';

/**
 * Chooses what to do after measuring a generated TTS clip against its slot.
 * One semantic rewrite is allowed. A second over-long result may be compressed up
 * to 1.25x; beyond that the job must stop instead of shipping rushed/truncated audio.
 */
export function durationAction(ratio: number, rewrites = 0): DurationAction {
    if (!Number.isFinite(ratio) || ratio <= 0) return 'fail';
    if (ratio < DURATION_RATIO_MIN) return rewrites === 0 ? 'rewrite-longer' : 'accept';
    if (ratio <= DURATION_RATIO_ACCEPT_MAX) return 'accept';
    if (ratio <= DURATION_RATIO_REWRITE_MAX) return 'compress';
    if (rewrites === 0) return 'rewrite-shorter';
    return ratio <= DURATION_RATIO_HARD_MAX ? 'compress' : 'fail';
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
        min: Math.max(1, Math.floor(target * DURATION_RATIO_MIN)),
        max: Math.max(1, Math.ceil(target * DURATION_RATIO_ACCEPT_MAX)),
    };
}

export const PIPELINE_STAGES = ['extract', 'translate', 'tts', 'assemble'] as const;
export type PipelineStage = typeof PIPELINE_STAGES[number];

/** Returns the first stage whose durable DB record is not completed. */
export function nextIncompleteStage(completedStages: Iterable<string>): PipelineStage | null {
    const completed = new Set(completedStages);
    return PIPELINE_STAGES.find((stage) => !completed.has(stage)) || null;
}
