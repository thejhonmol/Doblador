# Doblador orchestrator

Node.js API and BullMQ pipeline coordinator for ZhowPlus dubbing.

## Processes

Build once, then run the HTTP API and pipeline workers as separate processes:

```bash
npm run build
npm run start:api
npm run start:workers
```

The API defaults to `127.0.0.1:3000`. Both processes use PostgreSQL; API job creation also uses Redis to enqueue the first stage. Workers recover interrupted durable jobs at startup.

## Authentication and private beta

Authentication uses opaque random sessions stored in PostgreSQL. Passwords are stored with salted scrypt hashes. Browser sessions use an `HttpOnly`, `SameSite=Lax` cookie named `zhowplus_session`; non-browser clients may send the same session value as `Authorization: Bearer <token>`.

Production requires `AUTH_SECRET` with at least 32 characters. Registration is closed by default in production. To enable invited registration, set `AUTH_REGISTRATION_ENABLED=true` and a strong `ADMIN_BETA_INVITE_CODE`. Until transactional email verification is wired, keep the beta private. `MAX_ACTIVE_JOBS_PER_USER` limits concurrent expensive jobs per account (production default: 2). Login and registration also have an in-memory per-IP fixed-window limiter (`AUTH_LOGIN_RATE_LIMIT`, `AUTH_REGISTER_RATE_LIMIT`, `AUTH_RATE_WINDOW_MINUTES`). It protects a single API instance; multi-instance production should enforce the same policy in Redis or at the reverse proxy/edge.

State-changing browser requests are restricted to `CORS_ORIGINS`. Production uploads are hard-capped at 250 MB even if `MAX_UPLOAD_MB` is configured higher, and accepted files must match both an allowed extension/MIME and a recognized media-container signature.

## API v1

Public:

- `GET /api/health`
- `POST /api/v1/auth/register` — `{ email, password, inviteCode? }`; returns `{ user, session: { expiresAt } }` and sets the session cookie.
- `POST /api/v1/auth/login` — `{ email, password }`; same response/session behavior.

Authenticated:

- `POST /api/v1/auth/logout`
- `GET /api/v1/auth/me`
- `GET /api/v1/projects`
- `POST /api/v1/projects` — `{ name }`
- `GET /api/v1/projects/:projectId`
- `GET /api/v1/projects/:projectId/jobs`
- `POST /api/v1/projects/:projectId/jobs` — multipart `video`, `targetLang`, optional `voiceMode=clone|preset`
- `GET /api/v1/jobs/:jobId`
- `GET /api/v1/jobs/:jobId/segments?limit=100&offset=0` — owner-only pagination, `limit` 1–500, returns `{ segments, total, limit, offset }`
- `GET /api/v1/jobs/:jobId/sync-report`
- `GET /api/v1/jobs/:jobId/result`
- `GET /api/v1/voices`

Every project/job query is scoped to the authenticated user. API responses do not expose the server-side source/output filesystem path.

## Translation provider

Gemini remains the fallback translation provider. To use an OpenAI-compatible endpoint such as a Modal-hosted model, configure all three:

```env
LLM_BASE_URL=https://example/v1
LLM_API_KEY=...
LLM_MODEL=deepseek-flash-v4.1
```

The worker calls `/chat/completions` (unless `LLM_BASE_URL` already ends with that path), requests JSON, retries once without `response_format` on HTTP 400 for compatible servers, then falls back to Gemini. The exact Modal endpoint still needs a real integration request before its protocol compatibility can be claimed.

## Pipeline

Stages are `extract -> translate -> tts -> assemble`. Stage payloads, segments, speaker/voice assignments and synchronization reports are persisted so interrupted jobs can recover after restart.

TTS duration is measured and recorded, but duration ratio no longer triggers semantic rewrites, regeneration, or rejection. Timing adaptation belongs to the downstream stitching step. Real synthesis/provider failures still fail the job.

## Storage limitation

Uploads and rendered videos currently live on the orchestrator host filesystem. This is suitable for a single-host private beta, but it is not shared storage for horizontally scaled API/workers. S3/object storage and signed-download URLs are still required before multi-host production deployment.
