import { query } from './db';

/**
 * Persists the results of the pipeline to PostgreSQL.
 *
 * Until now only `jobs` and `job_stages` were written: `segments`, `speakers` and
 * `voices_catalog` were declared in init.sql and never touched (0 rows in a database
 * with 31 jobs). That meant the transcription and the translation existed only inside
 * the BullMQ return value, so a restart lost every completed stage and there was no
 * record of which voice each speaker got.
 */

/** Records which Fish Audio voice each detected speaker was dubbed with. */
export async function persistSpeakers(
    jobId: string,
    speakerVoiceMap: Record<string, string>,
    targetLang: string,
    voiceMode: string = 'clone'
): Promise<void> {
    if (!jobId) return;
    const entries = Object.entries(speakerVoiceMap);
    if (entries.length === 0) return;

    for (const [speakerLabel, referenceId] of entries) {
        await query(
            `INSERT INTO speakers (job_id, speaker_label, fish_reference_id, target_lang, voice_mode)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (job_id, speaker_label)
             DO UPDATE SET fish_reference_id = EXCLUDED.fish_reference_id,
                           target_lang = EXCLUDED.target_lang,
                           voice_mode = EXCLUDED.voice_mode`,
            [jobId, speakerLabel, referenceId, targetLang, voiceMode]
        );
    }
}

/**
 * Replaces the segment rows of a job with the current state.
 * A retried TTS stage must not leave stale rows behind, so this is a full replace
 * scoped to the job rather than an upsert.
 */
export async function persistSegments(jobId: string, segments: any[]): Promise<void> {
    if (!jobId || !segments?.length) return;

    await query('DELETE FROM segments WHERE job_id = $1', [jobId]);

    for (const seg of segments) {
        if (!seg) continue;
        const label = seg.speaker_label || 'SPEAKER_00';
        await query(
            `INSERT INTO segments (job_id, speaker_label, start_ms, end_ms, source_text,
                                   translated_text, tts_audio_url, generated_ms, speed_used,
                                   emotion, emotion_confidence, actual_start_ms, actual_end_ms,
                                   start_drift_ms, original_overlap_ms, actual_overlap_ms,
                                   regeneration_count, duration_ratio, sync_status, word_timestamps)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                     $15, $16, $17, $18, $19, $20::jsonb)`,
            [
                jobId,
                label,
                Math.round(seg.start_ms || 0),
                Math.round(seg.end_ms || 0),
                seg.text || '',
                seg.translated_text ?? null,
                seg.tts_audio_url ?? null,
                seg.generated_ms ?? null,
                seg.speed_used ?? 1.0,
                seg.emotion ?? null,
                seg.emotion_confidence ?? null,
                seg.actual_start_ms ?? null,
                seg.actual_end_ms ?? null,
                seg.start_drift_ms ?? null,
                seg.original_overlap_ms ?? null,
                seg.actual_overlap_ms ?? null,
                seg.regeneration_count ?? 0,
                seg.duration_ratio ?? null,
                seg.sync_status ?? null,
                JSON.stringify(Array.isArray(seg.words) ? seg.words : []),
            ]
        );
    }
}

/** Stores the complete JSON report on the durable job row for API download. */
export async function persistSyncReport(jobId: string, report: unknown): Promise<void> {
    if (!jobId || !report) return;
    await query('UPDATE jobs SET sync_report = $2::jsonb WHERE id = $1', [jobId, JSON.stringify(report)]);
}

interface PoolVoiceLike {
    id: string;
    label: string;
}

const POOL_CATEGORY: Record<string, string> = {
    male_young: 'male',
    male_mature: 'male',
    female_young: 'female',
    female_mature: 'female',
    child: 'child',
    default: 'male',
};

/**
 * Every Fish Audio reference voice in the pools is a multilingual model, so this is a
 * property of the catalogue, not of a particular voice. `VARCHAR(10)` was too narrow
 * for the word and the insert failed with "value too long for type character varying(10)".
 */
const VOICE_LANGUAGE = 'multilingual';

/**
 * Mirrors the configured voice pools into `voices_catalog`.
 *
 * The table is a projection of configuration, not an independent source: the worker
 * decides the voice, this only records what is available. Ids missing from the pools
 * are removed so the table can never advertise a voice the system would not use.
 */
export async function syncVoicesCatalog(pools: Record<string, PoolVoiceLike[]>): Promise<number> {
    const rows: { id: string; category: string; slot: number; gender: string; label: string }[] = [];
    const seen = new Map<string, string>();

    for (const [category, voices] of Object.entries(pools)) {
        if (category === 'default') continue;
        voices.forEach((v, i) => {
            const previous = seen.get(v.id);
            if (previous) {
                // A voice id present in two pools cannot be represented: the id is the
                // primary key, so the last insert would silently erase the first
                // category. First pool wins (it is the order the allocator uses) and we
                // say so, because the duplicate also means two characters of different
                // ages can end up sharing one voice.
                console.warn(
                    `[Voices] ${v.id} is in both '${previous}' and '${category}'. ` +
                    `The catalogue keeps '${previous}'. If these are meant to be different ` +
                    'voices, remove the duplicate from the FISH_VOICE_POOL_* variables.'
                );
                return;
            }
            seen.set(v.id, category);
            rows.push({
                id: v.id,
                category,
                slot: i + 1,
                gender: POOL_CATEGORY[category] || 'male',
                label: v.label || v.id,
            });
        });
    }
    if (rows.length === 0) return 0;

    for (const r of rows) {
        await query(
            `INSERT INTO voices_catalog (fish_reference_id, language, gender, label, preview_audio_url, category, slot)
             VALUES ($1, $2, $3, $4, NULL, $5, $6)
             ON CONFLICT (fish_reference_id)
             DO UPDATE SET language = EXCLUDED.language,
                           gender = EXCLUDED.gender,
                           label = EXCLUDED.label,
                           category = EXCLUDED.category,
                           slot = EXCLUDED.slot`,
            [r.id, VOICE_LANGUAGE, r.gender, r.label, r.category, r.slot]
        );
    }

    const ids = rows.map((r) => r.id);
    await query('DELETE FROM voices_catalog WHERE fish_reference_id <> ALL($1::varchar[])', [ids]);
    return ids.length;
}
