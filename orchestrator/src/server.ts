import express, { NextFunction, Response } from 'express';
import multer from 'multer';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { query } from './db';
import { startJob } from './pipeline';
import { createRateLimiter, parsePagination } from './api-policy';
import {
    AuthenticatedRequest,
    authenticateUser,
    clearSessionCookie,
    createSession,
    createUser,
    requireAuth,
    revokeSession,
    setSessionCookie,
} from './auth';

export const app = express();

const ALLOWED_ORIGINS = (process.env.CORS_ORIGINS || 'http://localhost:5173,http://127.0.0.1:5173')
    .split(',').map((s) => s.trim()).filter(Boolean);

app.disable('x-powered-by');
app.use(cors({
    credentials: true,
    origin: (origin, callback) => {
        if (!origin || ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
        return callback(new Error('Origin not allowed'));
    },
}));
app.use(express.json({ limit: '1mb' }));

// Browser state-changing calls must come from the configured frontend. Requests made
// by non-browser clients without Origin remain usable with Bearer authentication.
app.use((req, res, next) => {
    if (!['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method)) return next();
    const origin = req.get('origin');
    if (!origin || ALLOWED_ORIGINS.includes(origin)) return next();
    return res.status(403).json({ error: 'Untrusted request origin.' });
});

const UPLOADS_DIR = path.resolve(process.env.UPLOADS_DIR || path.join(__dirname, '..', 'uploads'));
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const defaultUploadMb = process.env.NODE_ENV === 'production' ? '250' : '2048';
const configuredUploadMb = Math.max(1, parseInt(process.env.MAX_UPLOAD_MB || defaultUploadMb, 10));
const MAX_UPLOAD_MB = process.env.NODE_ENV === 'production'
    ? Math.min(configuredUploadMb, 250)
    : configuredUploadMb;
const REGISTRATION_ENABLED = process.env.AUTH_REGISTRATION_ENABLED
    ? process.env.AUTH_REGISTRATION_ENABLED === 'true'
    : process.env.NODE_ENV !== 'production';
const BETA_INVITE_CODE = process.env.ADMIN_BETA_INVITE_CODE || '';
const MAX_ACTIVE_JOBS_PER_USER = Math.max(1, parseInt(
    process.env.MAX_ACTIVE_JOBS_PER_USER || (process.env.NODE_ENV === 'production' ? '2' : '5'),
    10
));
const AUTH_RATE_WINDOW_MINUTES = Math.max(1, parseInt(process.env.AUTH_RATE_WINDOW_MINUTES || '15', 10));
const LOGIN_RATE_LIMIT = Math.max(1, parseInt(process.env.AUTH_LOGIN_RATE_LIMIT || '10', 10));
const REGISTER_RATE_LIMIT = Math.max(1, parseInt(process.env.AUTH_REGISTER_RATE_LIMIT || '5', 10));
const loginRateLimiter = createRateLimiter(LOGIN_RATE_LIMIT, AUTH_RATE_WINDOW_MINUTES * 60 * 1000);
const registerRateLimiter = createRateLimiter(REGISTER_RATE_LIMIT, AUTH_RATE_WINDOW_MINUTES * 60 * 1000);
const ALLOWED_VIDEO_EXT = new Set(['.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.mpg', '.mpeg']);
const ALLOWED_VIDEO_MIME = new Set([
    'video/mp4', 'video/x-matroska', 'video/quicktime', 'video/x-msvideo',
    'video/webm', 'video/mpeg', 'application/octet-stream',
]);

const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, UPLOADS_DIR),
    filename: (_req, file, cb) => {
        cb(null, `${Date.now()}-${crypto.randomBytes(10).toString('hex')}${path.extname(file.originalname).toLowerCase()}`);
    },
});

const upload = multer({
    storage,
    limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024, files: 1 },
    fileFilter: (_req, file, cb) => {
        const ext = path.extname(file.originalname || '').toLowerCase();
        if (!ALLOWED_VIDEO_EXT.has(ext) || !ALLOWED_VIDEO_MIME.has(file.mimetype)) {
            return cb(new Error('Unsupported video file type.'));
        }
        cb(null, true);
    },
});

function hasKnownVideoSignature(filePath: string): boolean {
    const fd = fs.openSync(filePath, 'r');
    try {
        const buf = Buffer.alloc(16);
        const n = fs.readSync(fd, buf, 0, buf.length, 0);
        if (n < 4) return false;
        const hex4 = buf.subarray(0, 4).toString('hex');
        if (hex4 === '1a45dfa3') return true; // Matroska/WebM
        if (buf.subarray(4, 8).toString('ascii') === 'ftyp') return true; // MP4/MOV/M4V
        if (buf.subarray(0, 4).toString('ascii') === 'RIFF' && buf.subarray(8, 12).toString('ascii') === 'AVI ') return true;
        if (hex4 === '000001ba' || hex4 === '000001b3') return true; // MPEG program/video
        return false;
    } finally {
        fs.closeSync(fd);
    }
}

function cleanName(value: unknown): string {
    const name = String(value || '').trim().replace(/\s+/g, ' ');
    if (!name || name.length > 120) throw Object.assign(new Error('Project name must contain 1-120 characters.'), { statusCode: 400 });
    return name;
}

function validTargetLang(value: unknown): string {
    const lang = String(value || '').trim();
    if (!lang || lang.length > 32 || !/^[\p{L} -]+$/u.test(lang)) {
        throw Object.assign(new Error('targetLang must be a language code/name up to 32 characters.'), { statusCode: 400 });
    }
    return lang;
}

function routeParam(value: string | string[]): string {
    return Array.isArray(value) ? value[0] : value;
}

function mapProject(row: any) {
    return {
        id: row.id,
        name: row.name,
        createdAt: new Date(row.created_at).toISOString(),
        updatedAt: new Date(row.updated_at).toISOString(),
    };
}

function mapJob(row: any) {
    return {
        id: row.id,
        projectId: row.project_id,
        status: row.status,
        targetLang: row.target_lang,
        voiceMode: row.voice_mode,
        originalFilename: row.original_filename,
        createdAt: new Date(row.created_at).toISOString(),
        updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : null,
        completedAt: row.completed_at ? new Date(row.completed_at).toISOString() : null,
    };
}

async function ownedProject(projectId: string, userId: string): Promise<any | null> {
    const result = await query('SELECT * FROM projects WHERE id = $1 AND user_id = $2', [projectId, userId]);
    return result.rowCount === 1 ? result.rows[0] : null;
}

async function ownedJob(jobId: string, userId: string): Promise<any | null> {
    const result = await query(
        `SELECT id, project_id, status, target_lang, voice_mode, original_filename,
                created_at, updated_at, completed_at, sync_report
         FROM jobs WHERE id = $1 AND user_id = $2`,
        [jobId, userId]
    );
    return result.rowCount === 1 ? result.rows[0] : null;
}

async function issueSession(res: Response, user: { id: string; email: string; createdAt: string }) {
    const session = await createSession(user.id);
    setSessionCookie(res, session.token);
    return { user, session: { expiresAt: session.expiresAt } };
}

app.get('/api/health', (_req, res) => {
    res.json({ service: 'doblador-orchestrator', signature: 'doblador-orchestrator', status: 'ok' });
});

app.post('/api/v1/auth/register', registerRateLimiter, async (req, res, next) => {
    try {
        if (!REGISTRATION_ENABLED) {
            return res.status(403).json({ error: 'Registration is closed for this private beta.' });
        }
        if (BETA_INVITE_CODE && req.body?.inviteCode !== BETA_INVITE_CODE) {
            return res.status(403).json({ error: 'A valid beta invite code is required.' });
        }
        const user = await createUser(req.body?.email, req.body?.password);
        return res.status(201).json(await issueSession(res, user));
    } catch (err) { next(err); }
});

app.post('/api/v1/auth/login', loginRateLimiter, async (req, res, next) => {
    try {
        const user = await authenticateUser(req.body?.email, req.body?.password);
        if (!user) return res.status(401).json({ error: 'Invalid email or password.' });
        return res.json(await issueSession(res, user));
    } catch (err) { next(err); }
});

app.post('/api/v1/auth/logout', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        await revokeSession(req.sessionToken || null);
        clearSessionCookie(res);
        res.status(204).end();
    } catch (err) { next(err); }
});

app.get('/api/v1/auth/me', requireAuth, (req: AuthenticatedRequest, res) => {
    res.json({ user: req.authUser });
});

app.get('/api/v1/projects', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const result = await query('SELECT * FROM projects WHERE user_id = $1 ORDER BY created_at DESC', [req.authUser!.id]);
        res.json({ projects: result.rows.map(mapProject) });
    } catch (err) { next(err); }
});

app.post('/api/v1/projects', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const result = await query(
            'INSERT INTO projects (user_id, name) VALUES ($1, $2) RETURNING *',
            [req.authUser!.id, cleanName(req.body?.name)]
        );
        res.status(201).json({ project: mapProject(result.rows[0]) });
    } catch (err) { next(err); }
});

app.get('/api/v1/projects/:projectId', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const project = await ownedProject(routeParam(req.params.projectId), req.authUser!.id);
        if (!project) return res.status(404).json({ error: 'Project not found.' });
        return res.json({ project: mapProject(project) });
    } catch (err) { next(err); }
});

app.get('/api/v1/projects/:projectId/jobs', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const project = await ownedProject(routeParam(req.params.projectId), req.authUser!.id);
        if (!project) return res.status(404).json({ error: 'Project not found.' });
        const result = await query(
            `SELECT id, project_id, status, target_lang, voice_mode, original_filename,
                    created_at, updated_at, completed_at
             FROM jobs WHERE project_id = $1 AND user_id = $2 ORDER BY created_at DESC`,
            [project.id, req.authUser!.id]
        );
        return res.json({ jobs: result.rows.map(mapJob) });
    } catch (err) { next(err); }
});

async function createJobFromUpload(req: AuthenticatedRequest, res: Response, projectId: string): Promise<void> {
    const project = await ownedProject(projectId, req.authUser!.id);
    if (!project) {
        if (req.file?.path) fs.rmSync(req.file.path, { force: true });
        res.status(404).json({ error: 'Project not found.' });
        return;
    }
    if (!req.file) {
        res.status(400).json({ error: 'Video file is required.' });
        return;
    }
    if (!hasKnownVideoSignature(req.file.path)) {
        fs.rmSync(req.file.path, { force: true });
        res.status(415).json({ error: 'Uploaded file does not contain a recognized video container.' });
        return;
    }

    const videoPath = path.resolve(req.file.path);
    let jobId: string | null = null;
    try {
        const targetLang = validTargetLang(req.body?.targetLang);
        const voiceMode = req.body?.voiceMode === 'preset' ? 'preset' : 'clone';
        const active = await query(
            `SELECT count(*)::int AS count FROM jobs
             WHERE user_id = $1 AND status IN ('pending', 'processing')`,
            [req.authUser!.id]
        );
        if ((active.rows[0]?.count || 0) >= MAX_ACTIVE_JOBS_PER_USER) {
            throw Object.assign(
                new Error(`Active job limit reached (${MAX_ACTIVE_JOBS_PER_USER}). Wait for a current job to finish.`),
                { statusCode: 429 }
            );
        }
        const inserted = await query(
            `INSERT INTO jobs (user_id, project_id, status, source_url, target_lang, voice_mode, original_filename, updated_at)
             VALUES ($1, $2, 'pending', $3, $4, $5, $6, NOW()) RETURNING *`,
            [req.authUser!.id, project.id, videoPath, targetLang, voiceMode, path.basename(req.file.originalname)]
        );
        jobId = inserted.rows[0].id;
        await startJob(videoPath, targetLang, jobId!, voiceMode);
        res.status(202).json({ job: mapJob(inserted.rows[0]) });
    } catch (err) {
        fs.rmSync(videoPath, { force: true });
        if (jobId) await query('DELETE FROM jobs WHERE id = $1 AND user_id = $2', [jobId, req.authUser!.id]).catch(() => undefined);
        throw err;
    }
}

app.post('/api/v1/projects/:projectId/jobs', requireAuth, upload.single('video'), async (req: AuthenticatedRequest, res, next) => {
    try { await createJobFromUpload(req, res, routeParam(req.params.projectId)); } catch (err) { next(err); }
});

app.get('/api/v1/jobs/:jobId', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const job = await ownedJob(routeParam(req.params.jobId), req.authUser!.id);
        if (!job) return res.status(404).json({ error: 'Job not found.' });

        const [stagesResult, speakersResult, segmentsResult, segmentCountResult, emotionsResult] = await Promise.all([
            query(`SELECT stage, status, error, started_at, finished_at FROM job_stages WHERE job_id = $1 ORDER BY started_at ASC NULLS LAST`, [job.id]),
            query(`SELECT speaker_label, fish_reference_id, target_lang, voice_mode FROM speakers WHERE job_id = $1 ORDER BY speaker_label`, [job.id]),
            query(`SELECT speaker_label, start_ms, end_ms, source_text, translated_text, emotion, emotion_confidence,
                          generated_ms, speed_used, actual_start_ms, actual_end_ms, start_drift_ms,
                          original_overlap_ms, actual_overlap_ms, regeneration_count, duration_ratio, sync_status
                   FROM segments WHERE job_id = $1 ORDER BY start_ms ASC LIMIT 100`, [job.id]),
            query('SELECT count(*)::int AS total FROM segments WHERE job_id = $1', [job.id]),
            query(`SELECT emotion, count(*)::int AS count FROM segments
                   WHERE job_id = $1 AND emotion IS NOT NULL GROUP BY emotion ORDER BY count DESC`, [job.id]),
        ]);
        const emotionsSummary: Record<string, number> = {};
        for (const row of emotionsResult.rows) emotionsSummary[row.emotion] = row.count;
        return res.json({
            job: mapJob(job),
            stages: stagesResult.rows,
            speakers: speakersResult.rows,
            segmentCount: segmentCountResult.rows[0]?.total || 0,
            emotionsSummary,
            segments: segmentsResult.rows,
            syncReport: job.sync_report || null,
        });
    } catch (err) { next(err); }
});

app.get('/api/v1/jobs/:jobId/segments', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const job = await ownedJob(routeParam(req.params.jobId), req.authUser!.id);
        if (!job) return res.status(404).json({ error: 'Job not found.' });
        const { limit, offset } = parsePagination(req.query as Record<string, unknown>);
        const [segmentsResult, countResult] = await Promise.all([
            query(
                `SELECT speaker_label, start_ms, end_ms, source_text, translated_text, emotion, emotion_confidence,
                        generated_ms, speed_used, actual_start_ms, actual_end_ms, start_drift_ms,
                        original_overlap_ms, actual_overlap_ms, regeneration_count, duration_ratio, sync_status
                 FROM segments WHERE job_id = $1 ORDER BY start_ms ASC, id ASC LIMIT $2 OFFSET $3`,
                [job.id, limit, offset]
            ),
            query('SELECT count(*)::int AS total FROM segments WHERE job_id = $1', [job.id]),
        ]);
        return res.json({
            segments: segmentsResult.rows,
            total: countResult.rows[0]?.total || 0,
            limit,
            offset,
        });
    } catch (err) { next(err); }
});

app.get('/api/v1/jobs/:jobId/sync-report', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const job = await ownedJob(routeParam(req.params.jobId), req.authUser!.id);
        if (!job) return res.status(404).json({ error: 'Job not found.' });
        if (!job.sync_report) return res.status(404).json({ error: 'Synchronization report is not ready.' });
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${job.id}_sync_report.json"`);
        return res.send(JSON.stringify(job.sync_report, null, 2));
    } catch (err) { next(err); }
});

app.get('/api/v1/jobs/:jobId/result', requireAuth, async (req: AuthenticatedRequest, res, next) => {
    try {
        const job = await ownedJob(routeParam(req.params.jobId), req.authUser!.id);
        if (!job) return res.status(404).json({ error: 'Job not found.' });
        const result = await query(
            `SELECT output_url FROM job_stages WHERE job_id = $1 AND stage = 'assemble'
             AND status = 'completed' AND output_url IS NOT NULL ORDER BY finished_at DESC LIMIT 1`,
            [job.id]
        );
        if (result.rowCount !== 1) return res.status(404).json({ error: 'Finished video is not ready.' });
        const outputPath = path.resolve(result.rows[0].output_url);
        const root = UPLOADS_DIR + path.sep;
        if (!outputPath.startsWith(root) || !outputPath.endsWith('_dubbed.mp4')) {
            return res.status(400).json({ error: 'Invalid stored result path.' });
        }
        if (!fs.existsSync(outputPath)) return res.status(410).json({ error: 'Result file no longer exists.' });
        return res.download(outputPath, `${job.id}_dubbed.mp4`);
    } catch (err) { next(err); }
});

app.get('/api/v1/voices', requireAuth, async (_req, res, next) => {
    try {
        const result = await query(
            'SELECT fish_reference_id, language, gender, label, category, slot FROM voices_catalog ORDER BY category, slot'
        );
        res.json({ voices: result.rows });
    } catch (err) { next(err); }
});

app.use((err: any, _req: express.Request, res: express.Response, _next: NextFunction) => {
    if (err instanceof multer.MulterError) {
        return res.status(err.code === 'LIMIT_FILE_SIZE' ? 413 : 400).json({
            error: err.code === 'LIMIT_FILE_SIZE' ? `File exceeds the ${MAX_UPLOAD_MB} MB upload limit.` : err.message,
        });
    }
    const status = Number(err?.statusCode) || (err?.message === 'Origin not allowed' ? 403 : 500);
    if (status >= 500) console.error('[API] Request failed:', err);
    return res.status(status).json({ error: status >= 500 ? 'Internal server error.' : err.message });
});

export function startServer(port = 3000) {
    const bindHost = process.env.API_BIND_HOST || '127.0.0.1';
    return app.listen(port, bindHost, () => {
        console.log(`[API] Listening on http://${bindHost}:${port}; upload limit ${MAX_UPLOAD_MB} MB.`);
    });
}
