import { Queue, JobsOptions } from 'bullmq';
import { redisConnection } from './config';
import { PipelineStage } from './pipeline-policy';

export const extractQueue = new Queue('extract', { connection: redisConnection });
export const translateQueue = new Queue('translate', { connection: redisConnection });
export const ttsQueue = new Queue('tts', { connection: redisConnection });
export const assembleQueue = new Queue('assemble', { connection: redisConnection });

export const queuesByStage: Record<PipelineStage, Queue> = {
    extract: extractQueue,
    translate: translateQueue,
    tts: ttsQueue,
    assemble: assembleQueue,
};

const STAGE_ATTEMPTS: Record<PipelineStage, { attempts: number; delay: number }> = {
    extract: { attempts: 3, delay: 10000 },
    translate: { attempts: 5, delay: 3000 },
    tts: { attempts: 3, delay: 2000 },
    assemble: { attempts: 2, delay: 5000 },
};

function stageJobOptions(stage: PipelineStage, pipelineJobId: string): JobsOptions {
    const policy = STAGE_ATTEMPTS[stage];
    return {
        jobId: `${pipelineJobId}-${stage}`,
        attempts: policy.attempts,
        backoff: { type: 'exponential', delay: policy.delay },
    };
}

export async function enqueueStage(stage: PipelineStage, data: any): Promise<void> {
    const pipelineJobId = data?.jobId;
    if (!pipelineJobId) throw new Error(`Cannot enqueue '${stage}' without a pipeline jobId.`);
    await queuesByStage[stage].add(`${stage}-job`, data, stageJobOptions(stage, pipelineJobId));
}

export async function startJob(videoPath: string, targetLang: string, jobId: string, voiceMode = 'clone'): Promise<string> {
    if (!jobId) throw new Error('A durable job UUID is required before queueing the pipeline.');
    await enqueueStage('extract', { videoPath, targetLang, jobId, voiceMode });
    return jobId;
}
