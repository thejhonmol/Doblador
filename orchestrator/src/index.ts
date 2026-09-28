import { Job, Queue, JobsOptions } from 'bullmq';
import { redisConnection } from './config';
import { query } from './db';
import { migrate } from './migrate';
import { nextIncompleteStage, PIPELINE_STAGES, PipelineStage } from './pipeline-policy';

import { extractWorker } from './workers/extract';
import { translateWorker } from './workers/translate';
import { ttsWorker } from './workers/tts';
import { assembleWorker } from './workers/assemble';

console.log('Orchestrator starting...');

/**
 * The DB has to be reachable and migrated before any worker can start consuming,
 * otherwise the first `updateStage` fails and the job looks stalled. Postgres is
 * started by docker-compose in parallel with this process, so wait for it.
 */
async function waitForDatabase(maxWaitMs = 60000): Promise<void> {
    const started = Date.now();
    let lastError = '';
    while (Date.now() - started < maxWaitMs) {
        try {
            await query('SELECT 1');
            return;
        } catch (err: any) {
            lastError = err.message;
            await new Promise((r) => setTimeout(r, 2000));
        }
    }
    throw new Error(`PostgreSQL not reachable after ${maxWaitMs}ms: ${lastError}`);
}

const extractQueue = new Queue('extract', { connection: redisConnection });
const translateQueue = new Queue('translate', { connection: redisConnection });
const ttsQueue = new Queue('tts', { connection: redisConnection });
const assembleQueue = new Queue('assemble', { connection: redisConnection });

// Retry policy per stage. extract and assemble previously called `.add()` with no
// `attempts`, so a single transient failure killed the whole job while translate
// got 5 tries and tts got 3.
const STAGE_ATTEMPTS: Record<PipelineStage, { attempts: number; delay: number }> = {
    extract: { attempts: 3, delay: 10000 },
    translate: { attempts: 5, delay: 3000 },
    tts: { attempts: 3, delay: 2000 },
    assemble: { attempts: 2, delay: 5000 },
};

const queuesByStage: Record<PipelineStage, Queue> = {
    extract: extractQueue,
    translate: translateQueue,
    tts: ttsQueue,
    assemble: assembleQueue,
};

function stageJobOptions(stage: PipelineStage, pipelineJobId: string): JobsOptions {
    const policy = STAGE_ATTEMPTS[stage];
    return {
        jobId: `${pipelineJobId}-${stage}`,
        attempts: policy.attempts,
        backoff: { type: 'exponential', delay: policy.delay },
    };
}

async function enqueueStage(stage: PipelineStage, data: any): Promise<void> {
    const pipelineJobId = data?.jobId;
    if (!pipelineJobId) throw new Error(`Cannot enqueue '${stage}' without a pipeline jobId.`);
    await queuesByStage[stage].add(`${stage}-job`, data, stageJobOptions(stage, pipelineJobId));
}

async function findPipelineJob(stage: PipelineStage, pipelineJobId: string): Promise<Job | undefined> {
    const stableJob = await queuesByStage[stage].getJob(`${pipelineJobId}-${stage}`);
    if (stableJob) return stableJob;

    // Compatibility with jobs created before deterministic stage IDs were added.
    const jobs = await queuesByStage[stage].getJobs(
        ['waiting', 'active', 'delayed', 'paused', 'prioritized'] as any,
        0,
        -1,
        false
    );
    return jobs.find((job) => job.data?.jobId === pipelineJobId);
}

/**
 * Repairs the narrow crash window between completing one worker and enqueuing the
 * next. Existing BullMQ jobs are left alone so its own stalled-job recovery can do
 * its job; only a genuinely missing stage is reconstructed from the durable payload.
 */
async function recoverInterruptedJobs(): Promise<number> {
    const { rows } = await query(
        `SELECT j.id, j.status, j.source_url, j.target_lang, j.voice_mode, j.payload,
                ARRAY(SELECT js.stage FROM job_stages js
                      WHERE js.job_id = j.id AND js.status = 'completed') AS completed_stages,
                EXISTS(SELECT 1 FROM job_stages js
                       WHERE js.job_id = j.id AND js.status = 'failed') AS has_failed_stage
         FROM jobs j
         WHERE j.status IN ('pending', 'processing')`
    );

    let recovered = 0;
    for (const row of rows) {
        if (row.has_failed_stage) {
            await query(`UPDATE jobs SET status = 'failed' WHERE id = $1`, [row.id]);
            continue;
        }

        const stage = nextIncompleteStage(row.completed_stages || []);
        if (!stage) {
            await query(`UPDATE jobs SET status = 'completed' WHERE id = $1`, [row.id]);
            continue;
        }
        const existingJob = await findPipelineJob(stage, row.id);
        if (existingJob) {
            const state = await existingJob.getState();
            if (state === 'completed') {
                const result = existingJob.returnvalue;
                if (!result) {
                    await updateStage(row.id, stage, 'failed', null, `Completed '${stage}' queue job has no return value.`);
                    continue;
                }
                await updateStage(
                    row.id,
                    stage,
                    'completed',
                    stage === 'assemble' ? result.finalVideoPath || null : null
                );
                if (stage !== 'assemble') {
                    await persistPayload(row.id, result);
                    const nextStage = PIPELINE_STAGES[PIPELINE_STAGES.indexOf(stage) + 1];
                    if (nextStage) await enqueueStage(nextStage, { ...result, jobId: row.id });
                }
                recovered++;
                console.warn(`[Recovery] Reconciled completed '${stage}' queue job for ${row.id}.`);
            } else if (state === 'failed') {
                await updateStage(
                    row.id,
                    stage,
                    'failed',
                    null,
                    existingJob.failedReason || `Stage '${stage}' failed before its DB state was recorded.`
                );
            }
            continue;
        }

        let data: any;
        if (stage === 'extract') {
            data = {
                videoPath: row.source_url,
                targetLang: row.target_lang,
                voiceMode: row.voice_mode || 'clone',
                jobId: row.id,
            };
        } else {
            data = typeof row.payload === 'string' ? JSON.parse(row.payload) : row.payload;
            if (data) data = { ...data, jobId: row.id };
        }

        if (!data) {
            const message = `Cannot recover stage '${stage}': the previous stage payload is missing.`;
            await query(`UPDATE jobs SET status = 'failed' WHERE id = $1`, [row.id]);
            await updateStage(row.id, stage, 'failed', null, message);
            console.error(`[Recovery] ${row.id}: ${message}`);
            continue;
        }

        await enqueueStage(stage, data);
        recovered++;
        console.warn(`[Recovery] Re-enqueued missing '${stage}' stage for job ${row.id}.`);
    }
    return recovered;
}

// Helper to update DB stage status
async function updateStage(jobId: string, stage: string, status: string, url: string | null = null, error: string | null = null) {
    if (!jobId) return;
    try {
        // Check if stage exists
        const res = await query('SELECT id FROM job_stages WHERE job_id = $1 AND stage = $2', [jobId, stage]);
        if (res.rowCount === 0) {
            await query(
                'INSERT INTO job_stages (job_id, stage, status, started_at) VALUES ($1, $2, $3, NOW())',
                [jobId, stage, status]
            );
        } else {
            const isFinished = status === 'completed' || status === 'failed';
            await query(
                'UPDATE job_stages SET status = $1, output_url = $2, error = $3, finished_at = (CASE WHEN $6::boolean THEN NOW() ELSE finished_at END) WHERE job_id = $4 AND stage = $5',
                [status, url, error, jobId, stage, isFinished]
            );
        }

        // Also update parent job if complete
        if (stage === 'assemble' && status === 'completed') {
            await query('UPDATE jobs SET status = \'completed\' WHERE id = $1', [jobId]);
        } else if (status === 'failed') {
            await query('UPDATE jobs SET status = \'failed\' WHERE id = $1', [jobId]);
        } else {
            await query('UPDATE jobs SET status = \'processing\' WHERE id = $1 AND status != \'processing\'', [jobId]);
        }
    } catch (e) {
        console.error('DB Update Error:', e);
    }
}

/**
 * Persists the output of a finished stage on the job row.
 *
 * This is what makes the pipeline survive a restart: the payload is the only copy
 * of the transcription/translation/TTS result that lives outside the BullMQ return
 * value, and without it a restart mid-run loses every completed stage.
 */
async function persistPayload(jobId: string, payload: any): Promise<void> {
    if (!jobId || !payload) return;
    try {
        await query('UPDATE jobs SET payload = $2 WHERE id = $1', [jobId, JSON.stringify(payload)]);
    } catch (e: any) {
        console.error('DB Persist Payload Error:', e);
    }
}

// Bind DB events and pipeline flow directly to workers
const workers = [
    { name: 'extract', worker: extractWorker },
    { name: 'translate', worker: translateWorker },
    { name: 'tts', worker: ttsWorker },
    { name: 'assemble', worker: assembleWorker }
];

// A BullMQ worker starts consuming the moment it is constructed, which happens at
// import time — before bootstrap() has verified the database. Pause them all here and
// resume once the schema is migrated; otherwise a job picked up in the first seconds
// cannot record its own stage and looks stalled.
const paused = Promise.all(workers.map(({ worker }) => worker.pause()));

workers.forEach(({ name, worker }) => {
    worker.on('active', (job) => {
        const pJobId = job?.data?.jobId;
        if (pJobId) {
            console.log(`[Stage Active] ${name} started for job ${pJobId}`);
            updateStage(pJobId, name, 'processing');
        }
    });

    worker.on('failed', (job, err) => {
        const pJobId = job?.data?.jobId;
        const attemptsTotal = job?.opts?.attempts || 1;
        const attemptsMade = job?.attemptsMade || 1;

        if (attemptsMade < attemptsTotal) {
            console.log(`[Stage Retry] ${name} attempt ${attemptsMade}/${attemptsTotal} failed for job ${pJobId}. Will retry...`);
            return;
        }

        if (pJobId) {
            console.error(`[Stage Failed] ${name} permanently failed for job ${pJobId}:`, err);
            updateStage(pJobId, name, 'failed', null, err.message);
        }
    });
});

// Flow transitions on worker completion
extractWorker.on('completed', async (job, returnvalue) => {
    const pJobId = job.data?.jobId;
    console.log(`[Flow] Extract completed for ${pJobId}. Moving to Translate.`);
    await updateStage(pJobId, 'extract', 'completed');
    await persistPayload(pJobId, returnvalue);
    await enqueueStage('translate', returnvalue);
});

translateWorker.on('completed', async (job, returnvalue) => {
    const pJobId = job.data?.jobId;
    console.log(`[Flow] Translate completed for ${pJobId}. Moving to TTS.`);
    await updateStage(pJobId, 'translate', 'completed');
    await persistPayload(pJobId, returnvalue);
    await enqueueStage('tts', returnvalue);
});

ttsWorker.on('completed', async (job, returnvalue) => {
    const pJobId = job.data?.jobId;
    console.log(`[Flow] TTS completed for ${pJobId}. Moving to Assemble.`);
    await updateStage(pJobId, 'tts', 'completed');
    await persistPayload(pJobId, returnvalue);
    await enqueueStage('assemble', returnvalue);
});

assembleWorker.on('completed', async (job, returnvalue) => {
    const pJobId = job.data?.jobId;
    console.log(`[Flow] Assemble completed for ${pJobId}. Final Video:`, returnvalue.finalVideoPath);
    await updateStage(pJobId, 'assemble', 'completed', returnvalue.finalVideoPath);
});

// Helper function to trigger a new job (can be attached to an Express API)
export async function startJob(videoPath: string, targetLang: string, providedJobId?: string, voiceMode: string = 'clone') {
    const jobId = providedJobId || `job_${Date.now()}`;
    console.log(`Starting pipeline for ${videoPath} to ${targetLang} (Job: ${jobId}, Voice Mode: ${voiceMode})`);

    await enqueueStage('extract', {
        videoPath,
        targetLang,
        jobId,
        voiceMode
    });

    return jobId;
}

async function bootstrap(): Promise<void> {
    await paused;
    await waitForDatabase();
    await migrate();
    const recovered = await recoverInterruptedJobs();
    if (recovered > 0) {
        console.warn(`[Recovery] Restored ${recovered} interrupted pipeline stage(s).`);
    }

    // Seed voices_catalog at boot, not only when a job reaches the TTS stage: the
    // /api/voices endpoint is read by tooling that has no job to offer.
    try {
        const { syncVoicesCatalog } = await import('./persist');
        const { VOICE_POOLS } = await import('./workers/tts');
        const n = await syncVoicesCatalog(VOICE_POOLS);
        console.log(`[DB] voices_catalog synced (${n} voice(s) available).`);
    } catch (e: any) {
        console.warn(`[DB] Could not sync voices_catalog: ${e.message}`);
    }

    // Only start consuming once the DB is consistent, otherwise a job picked up in
    // the first seconds could not record its own stage.
    for (const { name, worker } of workers) {
        await worker.resume();
        console.log(`[Orchestrator] Worker '${name}' listening.`);
    }

    console.log('Orchestrator started. Workers are listening...');

    const { startServer } = await import('./server');
    startServer(3000);
}

bootstrap().catch((err) => {
    console.error('[Orchestrator] Fatal startup error:', err);
    process.exit(1);
});
