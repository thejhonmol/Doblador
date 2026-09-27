import { query } from './db';

/**
 * Applies the schema idempotently on boot.
 *
 * `init.sql` is mounted into `/docker-entrypoint-initdb.d`, which Docker only runs
 * when the volume is EMPTY. On any existing database the schema was frozen at
 * whatever it was the first time, so no column added after that first run ever
 * reached the database. Everything here is IF NOT EXISTS / IF EXISTS, so it is
 * safe to run on every start.
 */
const SCHEMA_STATEMENTS: string[] = [
    `CREATE TABLE IF NOT EXISTS jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL,
        status VARCHAR(32) NOT NULL,
        source_url TEXT NOT NULL,
        target_lang VARCHAR(10) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )`,
    `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS payload JSONB`,
    `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS voice_mode VARCHAR(32) DEFAULT 'clone'`,
    `ALTER TABLE speakers ADD COLUMN IF NOT EXISTS voice_mode VARCHAR(32) DEFAULT 'clone'`,

    `CREATE TABLE IF NOT EXISTS job_stages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
        stage VARCHAR(32) NOT NULL,
        status VARCHAR(32) NOT NULL,
        output_url TEXT,
        error TEXT,
        started_at TIMESTAMP WITH TIME ZONE,
        finished_at TIMESTAMP WITH TIME ZONE
    )`,

    `CREATE TABLE IF NOT EXISTS speakers (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
        speaker_label VARCHAR(32) NOT NULL,
        fish_reference_id VARCHAR(64) NOT NULL,
        target_lang VARCHAR(10) NOT NULL,
        UNIQUE(job_id, speaker_label)
    )`,

    `CREATE TABLE IF NOT EXISTS segments (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
        speaker_label VARCHAR(32) NOT NULL,
        start_ms INT NOT NULL,
        end_ms INT NOT NULL,
        source_text TEXT NOT NULL,
        translated_text TEXT,
        tts_audio_url TEXT,
        generated_ms INT,
        speed_used NUMERIC(4, 2) DEFAULT 1.00,
        emotion VARCHAR(32),
        emotion_confidence NUMERIC(4, 3)
    )`,
    `ALTER TABLE segments ADD COLUMN IF NOT EXISTS emotion VARCHAR(32)`,
    `ALTER TABLE segments ADD COLUMN IF NOT EXISTS emotion_confidence NUMERIC(4, 3)`,

    `CREATE TABLE IF NOT EXISTS voices_catalog (
        fish_reference_id VARCHAR(64) PRIMARY KEY,
        language VARCHAR(32) NOT NULL,
        gender VARCHAR(16) NOT NULL,
        label VARCHAR(64) NOT NULL,
        preview_audio_url TEXT
    )`,
    `ALTER TABLE voices_catalog ALTER COLUMN language TYPE VARCHAR(32)`,
    `ALTER TABLE voices_catalog ALTER COLUMN preview_audio_url DROP NOT NULL`,
    `ALTER TABLE voices_catalog ADD COLUMN IF NOT EXISTS category VARCHAR(32)`,
    `ALTER TABLE voices_catalog ADD COLUMN IF NOT EXISTS slot INT`,

    `CREATE INDEX IF NOT EXISTS idx_segments_job ON segments(job_id)`,
    `CREATE INDEX IF NOT EXISTS idx_job_stages_job ON job_stages(job_id)`,
];

export async function migrate(): Promise<void> {
    for (const stmt of SCHEMA_STATEMENTS) {
        await query(stmt);
    }
    console.log('[DB] Schema verified (idempotent migration applied).');
}

/**
 * Marks jobs that were in flight when the process died as failed.
 *
 * The pipeline is driven by `worker.on('completed')` handlers living in this same
 * process. If the orchestrator is killed mid-run, the `completed` event never
 * fires, the BullMQ job stalls and the row stays 'processing' forever. That
 * already happened in production: 4 jobs were stuck in 'pending'/'processing' and
 * one had failed with "job stalled more than allowable limit".
 *
 * Called before the queues start consuming, so nothing is in flight yet and
 * 'processing' genuinely means "orphaned by a previous run".
 */
export async function reconcileOrphanedJobs(): Promise<number> {
    const { rows } = await query(
        `UPDATE jobs SET status = 'failed'
         WHERE status IN ('pending', 'processing')
         RETURNING id`
    );
    if (rows.length === 0) return 0;

    const ids = rows.map((r: { id: string }) => r.id);
    await query(
        `UPDATE job_stages
         SET status = 'failed',
             error = COALESCE(error, 'Orchestrator restarted while stage was in progress; job cannot resume automatically.'),
             finished_at = NOW()
         WHERE job_id = ANY($1::uuid[]) AND status = 'processing'`,
        [ids]
    );

    console.warn(
        `[DB] Reconciled ${rows.length} orphaned job(s) left in-flight by a previous run: ${ids.join(', ')}`
    );
    return rows.length;
}
