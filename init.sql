-- Overall job state
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    status VARCHAR(32) NOT NULL, -- 'pending', 'processing', 'completed', 'failed'
    source_url TEXT NOT NULL,
    target_lang VARCHAR(10) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Last known payload for each job (output of the latest completed stage).
-- Essential to allow pipeline recovery on orchestrator restarts.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS payload JSONB;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS voice_mode VARCHAR(32) DEFAULT 'clone';
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS sync_report JSONB;

-- Granular stage tracking
CREATE TABLE IF NOT EXISTS job_stages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    stage VARCHAR(32) NOT NULL, -- 'extract', 'translate', 'tts', 'assemble'
    status VARCHAR(32) NOT NULL,
    output_url TEXT,
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE
);

-- Speaker to Fish Audio voice mapping. Written by the TTS worker.
CREATE TABLE IF NOT EXISTS speakers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    speaker_label VARCHAR(32) NOT NULL, -- 'SPEAKER_00', 'SPEAKER_01'
    fish_reference_id VARCHAR(64) NOT NULL,
    target_lang VARCHAR(10) NOT NULL,
    voice_mode VARCHAR(32) DEFAULT 'clone',
    UNIQUE(job_id, speaker_label)
);

-- Segments with timestamps, timing metrics, and playback speed ratio
CREATE TABLE IF NOT EXISTS segments (
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
    emotion_confidence NUMERIC(4, 3),
    actual_start_ms INT,
    actual_end_ms INT,
    start_drift_ms INT,
    original_overlap_ms INT,
    actual_overlap_ms INT,
    regeneration_count INT DEFAULT 0,
    duration_ratio NUMERIC(6, 3),
    sync_status VARCHAR(32),
    word_timestamps JSONB DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_segments_job ON segments(job_id);
CREATE INDEX IF NOT EXISTS idx_job_stages_job ON job_stages(job_id);

-- Voice catalogue seeded from real TTS pools upon startup.
CREATE TABLE IF NOT EXISTS voices_catalog (
    fish_reference_id VARCHAR(64) PRIMARY KEY,
    language VARCHAR(32) NOT NULL,
    gender VARCHAR(16) NOT NULL,
    label VARCHAR(64) NOT NULL,
    preview_audio_url TEXT,
    category VARCHAR(32),
    slot INT
);
