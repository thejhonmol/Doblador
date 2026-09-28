import express from 'express';
import multer from 'multer';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { query } from './db';
import { startJob } from './index';

const app = express();

/**
 * CORS is restricted to the local frontend instead of `cors()` (which reflects any
 * origin). The API has no authentication, so an open policy let any page in the
 * user's browser drive uploads and read job state.
 */
const ALLOWED_ORIGINS = (process.env.CORS_ORIGINS || 'http://localhost:5173,http://127.0.0.1:5173')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

app.use(cors({
    origin: (origin, callback) => {
        // Same-origin/curl requests carry no Origin header.
        if (!origin || ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
        console.warn(`[API] Blocked CORS origin: ${origin}`);
        return callback(new Error('Origin not allowed'));
    }
}));
app.use(express.json());

const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

const MAX_UPLOAD_MB = parseInt(process.env.MAX_UPLOAD_MB || '2048', 10);
const ALLOWED_VIDEO_EXT = new Set(['.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.mpg', '.mpeg']);

// Configure multer storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOADS_DIR);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname).toLowerCase());
    }
});

const upload = multer({
    storage,
    limits: {
        fileSize: MAX_UPLOAD_MB * 1024 * 1024,
        files: 1,
    },
    // Without this, any file the operator happened to have could be POSTed as a
    // "video" and handed straight to ffmpeg.
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname || '').toLowerCase();
        if (!ALLOWED_VIDEO_EXT.has(ext)) {
            return cb(new Error(`Unsupported file type '${ext || 'none'}'. Allowed: ${[...ALLOWED_VIDEO_EXT].join(', ')}`));
        }
        cb(null, true);
    }
});

// POST /upload
app.post('/api/upload', upload.single('video'), async (req, res) => {
    try {
        const targetLang = req.body.targetLang;
        const voiceMode = req.body.voiceMode === 'preset' ? 'preset' : 'clone';
        const file = req.file;

        if (!file || !targetLang) {
            return res.status(400).json({ error: 'Video file and targetLang are required.' });
        }

        const videoPath = path.resolve(file.path);

        // 1. Create a dummy user UUID for MVP (In reality, extracted from Auth token)
        const dummyUserId = '00000000-0000-0000-0000-000000000000';

        // 2. Insert into DB to get UUID and track state
        const insertResult = await query(
            `INSERT INTO jobs (user_id, status, source_url, target_lang, voice_mode)
             VALUES ($1, $2, $3, $4, $5) RETURNING id`,
            [dummyUserId, 'pending', videoPath, targetLang, voiceMode]
        );

        const jobId = insertResult.rows[0].id;

        // 3. Queue the initial job step
        await startJob(videoPath, targetLang, jobId, voiceMode);

        res.status(202).json({
            message: 'Job created successfully',
            jobId: jobId
        });

    } catch (error: any) {
        console.error('Error during upload:', error);
        // Remove the orphaned upload so a rejected file does not sit on disk.
        if (req.file?.path && fs.existsSync(req.file.path)) {
            try { fs.unlinkSync(req.file.path); } catch { /* ignore */ }
        }
        res.status(500).json({ error: error.message });
    }
});

// GET /jobs/:id
app.get('/api/jobs/:id', async (req, res) => {
    try {
        const jobId = req.params.id;

        const jobResult = await query(`SELECT * FROM jobs WHERE id = $1`, [jobId]);
        if (jobResult.rowCount === 0) {
            return res.status(404).json({ error: 'Job not found' });
        }

        const stagesResult = await query(`SELECT * FROM job_stages WHERE job_id = $1 ORDER BY started_at ASC`, [jobId]);

        const stages = stagesResult.rows.map(s => {
            if (s.output_url) {
                return {
                    ...s,
                    download_url: `/api/jobs/${jobId}/download`
                };
            }
            return s;
        });

        // Speaker -> voice assignment and the persisted segments, so the client can
        // show which voice was used and inspect the translation. Both tables were
        // previously written by nobody.
        let speakers = [];
        let segmentCount = 0;
        let emotionsSummary: Record<string, number> = {};
        let segmentsList = [];
        try {
            const speakersRes = await query(
                'SELECT speaker_label, fish_reference_id, target_lang, voice_mode FROM speakers WHERE job_id = $1 ORDER BY speaker_label',
                [jobId]
            );
            speakers = speakersRes.rows;
            const segRes = await query('SELECT count(*)::int AS n FROM segments WHERE job_id = $1', [jobId]);
            segmentCount = segRes.rows[0].n;

            const emotionsRes = await query(
                `SELECT emotion, count(*)::int AS count
                 FROM segments
                 WHERE job_id = $1 AND emotion IS NOT NULL
                 GROUP BY emotion
                 ORDER BY count DESC`,
                [jobId]
            );
            for (const r of emotionsRes.rows) {
                emotionsSummary[r.emotion] = r.count;
            }

            const detailedSegs = await query(
                `SELECT speaker_label, start_ms, end_ms, source_text, translated_text,
                        emotion, emotion_confidence, generated_ms, speed_used,
                        actual_start_ms, actual_end_ms, start_drift_ms,
                        original_overlap_ms, actual_overlap_ms, regeneration_count,
                        duration_ratio, sync_status
                 FROM segments
                 WHERE job_id = $1
                 ORDER BY start_ms ASC LIMIT 100`,
                [jobId]
            );
            segmentsList = detailedSegs.rows;
        } catch (dbErr: any) {
            console.warn(`[API] Could not read speakers/segments for job ${jobId}: ${dbErr.message}`);
        }

        const { sync_report: syncReport, ...jobRow } = jobResult.rows[0];
        res.json({
            job: jobRow,
            stages,
            speakers,
            segmentCount,
            emotionsSummary,
            segments: segmentsList,
            syncReport: syncReport || null,
        });
    } catch (error: any) {
        console.error('Error fetching job:', error);
        res.status(500).json({ error: error.message });
    }
});

/** Download the auditable per-segment synchronization metrics as JSON. */
app.get('/api/jobs/:id/sync-report', async (req, res) => {
    try {
        const { rows, rowCount } = await query('SELECT sync_report FROM jobs WHERE id = $1', [req.params.id]);
        if (rowCount === 0) return res.status(404).json({ error: 'Job not found' });
        if (!rows[0].sync_report) return res.status(404).json({ error: 'Synchronization report is not ready' });
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${req.params.id}_sync_report.json"`);
        return res.send(JSON.stringify(rows[0].sync_report, null, 2));
    } catch (error: any) {
        console.error('Error downloading synchronization report:', error);
        return res.status(500).json({ error: error.message });
    }
});

/**
 * GET /jobs/:id/download
 *
 * Serves the dubbed MP4 for a job. This replaces `app.use('/uploads', express.static(...))`,
 * which published the source video, the extracted WAV, the Demucs stems and every
 * individual TTS clip to anyone who could reach the port. Now the only thing reachable
 * is the finished file, and only for a job id the caller already knows.
 */
app.get('/api/jobs/:id/download', async (req, res) => {
    try {
        const jobId = req.params.id;
        const stageResult = await query(
            `SELECT output_url FROM job_stages
             WHERE job_id = $1 AND stage = 'assemble' AND status = 'completed' AND output_url IS NOT NULL
             ORDER BY finished_at DESC LIMIT 1`,
            [jobId]
        );
        if (stageResult.rowCount === 0) {
            return res.status(404).json({ error: 'No finished video for this job' });
        }

        const outputPath = path.resolve(stageResult.rows[0].output_url);
        // Defence in depth: the path comes from the DB, but only ever serve a file
        // that lives in the uploads directory and looks like the final artefact.
        const uploadsRoot = path.resolve(UPLOADS_DIR) + path.sep;
        if (!outputPath.startsWith(uploadsRoot) || !outputPath.endsWith('_dubbed.mp4')) {
            console.error(`[API] Refusing to serve suspicious output path: ${outputPath}`);
            return res.status(400).json({ error: 'Invalid output path' });
        }
        if (!fs.existsSync(outputPath)) {
            return res.status(410).json({ error: 'The output file no longer exists on disk' });
        }

        res.download(outputPath, path.basename(outputPath));
    } catch (error: any) {
        console.error('Error downloading job output:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/health
 *
 * Also the signature used by scripts/prepare_ports.ps1 to tell "our own previous
 * instance" apart from an unrelated process squatting on the port. Guessing by image
 * name was not good enough: any other Node server on port 3000 would have been
 * silently killed.
 */
app.get('/api/health', (_req, res) => {
    res.json({
        service: 'doblador-orchestrator',
        signature: 'doblador-orchestrator',
        status: 'ok',
    });
});

/**
 * GET /api/voices
 * Exposes the voice catalogue the TTS worker actually uses, so a future speaker-mapping
 * UI has a single source of truth instead of hardcoding IDs in the frontend.
 */
app.get('/api/voices', async (_req, res) => {
    try {
        const { rows } = await query(
            'SELECT fish_reference_id, language, gender, label, category, slot FROM voices_catalog ORDER BY category, slot'
        );
        res.json({ voices: rows });
    } catch (error: any) {
        console.error('Error fetching voices:', error);
        res.status(500).json({ error: error.message });
    }
});

// Multer errors (size limit, bad extension) must not become opaque 500s.
app.use((err: any, _req: express.Request, res: express.Response, next: express.NextFunction) => {
    if (err instanceof multer.MulterError) {
        const tooBig = err.code === 'LIMIT_FILE_SIZE';
        return res.status(400).json({
            error: tooBig
                ? `File exceeds the ${MAX_UPLOAD_MB} MB limit (MAX_UPLOAD_MB).`
                : `Upload rejected: ${err.message}`
        });
    }
    if (err) {
        console.error('[API] Unhandled error:', err.message);
        return res.status(400).json({ error: err.message });
    }
    return next();
});

export function startServer(port = 3000) {
    // Loopback by default: this API has no authentication, so exposing it on all
    // interfaces would let anyone on the network upload files and read job state.
    const bindHost = process.env.API_BIND_HOST || '127.0.0.1';
    app.listen(port, bindHost, () => {
        console.log(
            `[API] Server is running on http://${bindHost}:${port} ` +
            `(max upload ${MAX_UPLOAD_MB} MB, CORS: ${ALLOWED_ORIGINS.join(' ')})`
        );
    });
}
