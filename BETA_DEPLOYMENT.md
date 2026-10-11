# ZhowPlus Doblador — beta deployment runbook

This runbook describes the repository state prepared for a private beta. It keeps
provider credentials outside the image, gives every long-running local service a
health check, and preserves the current pipeline contract instead of introducing a
second orchestration layer.

## Target routing

The public deployment is expected to use one HTTPS origin:

| Path | Destination |
| --- | --- |
| `/` | ZhowPlus static landing page |
| `/app/` | React dubbing frontend |
| `/api/v1/` | Express orchestrator; the edge preserves the full `/api/v1/...` URI |

`scripts/infra/nginx.local.conf` implements that contract for local verification.
Both proxy locations preserve their path prefixes: `/api/v1/...` reaches the Express
routes unchanged, and `/app/...` reaches Vite unchanged so its configured `/app/`
base keeps asset and HMR URLs valid. The production reverse proxy can use the same
routing rules after TLS and the real hostnames are configured.

The current frontend/orchestrator owners must keep the browser API base compatible
with `/api/v1/` before production. The local edge is deliberately a routing layer; it
does not rewrite application JavaScript.

## 1. Local infrastructure smoke test

Prerequisites: Docker Desktop with the Linux engine running.

From the repository root:

```powershell
Copy-Item scripts/infra/.env.compose.example scripts/infra/.env.compose.local
# Change both local passwords in .env.compose.local.
docker compose --env-file scripts/infra/.env.compose.local config --quiet
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/infra/verify_local.ps1 -EnvFile scripts/infra/.env.compose.local -WithEdge
```

Expected health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz
docker compose --env-file scripts/infra/.env.compose.local ps
```

The base Compose starts PostgreSQL and Redis. The `edge` profile additionally starts
nginx. It expects the ZhowPlus landing repository at `../zhowplus` by default and the
native React and Express development processes on ports 5173 and 3000.

## 2. Python timing regression test

The timing module is intentionally independent from model loading. It verifies the
PR #4 behavior without downloading Whisper, MOSS, Demucs or provider models:

```powershell
cd python-services
python -m pip install numpy==2.4.6 scipy==1.17.1
python -m unittest -v test_audio_timing
```

The test requires `ffmpeg` on `PATH`. It covers long takes above 1.25x, multi-stage
`atempo`, extreme fitting, duplicate timestamps, source speaker overlap and real
processing failures. Long generated speech is fitted once; duration alone is not a
reason to regenerate TTS.

## 3. Native Python service

The current Windows development path remains supported:

```powershell
cd python-services
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Check readiness after models finish loading:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
```

The service serializes GPU-heavy inference to avoid simultaneous VRAM spikes. This is
a deliberate single-GPU beta policy, so throughput is bounded by one heavy Python
request at a time.

## 4. Optional NVIDIA container

`python-services/Dockerfile` is based on CUDA 12.4 and matches the pinned PyTorch CUDA
wheel family. The profile needs Docker's NVIDIA GPU support and a compatible host
driver:

```powershell
nvidia-smi
docker compose --env-file scripts/infra/.env.compose.local --profile gpu config --quiet
docker compose --env-file scripts/infra/.env.compose.local --profile gpu build python-ai
docker compose --env-file scripts/infra/.env.compose.local --profile gpu up -d python-ai
docker compose --env-file scripts/infra/.env.compose.local --profile gpu ps
```

The Python API accepts absolute filesystem paths for several DSP operations. For that
reason the orchestrator and Python service must see the same media files. The GPU
profile mounts `DOBLADOR_WORKDIR` at `/work`; a fully containerized orchestrator must
use that same mount and container-visible paths before this profile can replace the
native Python process end to end.

## 5. Provider configuration

Keep real credentials only in the deployment secret store or local ignored `.env`.
Do not bake them into images or nginx configuration.

Fish Audio remains the TTS provider and its configured concurrency must respect the
limits of the actual account(s). The beta should use accounts legitimately assigned
to the project; the deployment plan does not rely on creating multiple trial accounts
to multiply free quotas.

The audEERING age/gender checkpoint remains in the pipeline for now. Its licensing is
not being treated as cleared for commercial production in this runbook. Verify the
model/card terms for the intended commercial use or replace that classifier before a
commercial launch. The current product only needs gender classification, so a future
replacement does not need an age head unless voice allocation still uses age buckets.

## S3 and Modal boundary

No S3 or Modal adapter is added at this stage because the current pipeline passes
absolute local paths between Express and the Python DSP service. A thin adapter would
be misleading: a remote worker cannot use those paths and would need object transfer,
job-scoped storage, retries, integrity checks and cleanup semantics.

A safe future remote-worker contract is:

1. Express stores the source/job artifacts under a job-scoped S3 prefix and issues
   short-lived access only for that job.
2. A worker receives object keys plus operation metadata rather than host paths.
3. The worker downloads to ephemeral storage, runs the existing operation, uploads
   outputs atomically and returns object keys plus checksums/metadata.
4. Express persists those object keys and drives the next stage.
5. Lifecycle rules delete temporary job objects after the retention window.

Modal is viable once that object-based contract exists. Until then, keeping the Python
worker near the shared filesystem is the lower-risk beta architecture.

## Known beta limits

- Heavy Python model startup can take minutes on a cold machine because model weights
  are loaded before the FastAPI process becomes ready.
- The optional GPU container has not been runtime-smoke-tested on this workstation
  when Docker Desktop's Linux engine is stopped. Compose syntax can still be validated
  without the daemon.
- MOSS/Sortformer use `transcribe-cpp` and GPU/Vulkan behavior may differ between the
  native Windows environment and Linux containers; validate transcription quality on
  the exact production GPU before switching the worker to the container profile.
- Strong automatic speech acceleration completes the timing fit but may reduce
  intelligibility. The synchronization report records the effective speed for review.
- External API availability, quotas, pricing and provider-specific limits remain
  external dependencies and should be verified against the accounts used for beta.
