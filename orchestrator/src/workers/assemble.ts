import { Worker, Job } from 'bullmq';
import { redisConnection } from '../config';
import { stitchSegments } from './stitch';
import { probeDurationSec } from './extract';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import fs from 'fs';

// Mix gains. Speech is already LUFS-matched to the original vocals by the Python
// service, so these only shape the final balance against the background stem.
const BG_GAIN = 0.45;
const SPEECH_GAIN = 1.15;
const AAC_BITRATE = '192k';

export const assembleWorker = new Worker('assemble', async (job: Job) => {
    const { segments, videoPath, jobId, backgroundAudioPath, fullTtsAudioPath } = job.data;
    console.log(`[Assemble] Muxing final video for job ${jobId}`);

    const finalVideoPath = path.join(path.dirname(videoPath), `${jobId}_dubbed.mp4`);

    // ── Output length is owned by the source video, never by the audio ──
    // The previous build let the audio define it through `-shortest`: whenever Demucs
    // was unavailable the only audio input was the TTS track, whose length came from
    // `last_segment_end + 5s`, so any tail without dialogue (outro, credits, silence)
    // was cut off the delivered file.
    let videoDurationSec: number | null =
        typeof job.data.videoDurationSec === 'number' && job.data.videoDurationSec > 0
            ? job.data.videoDurationSec
            : null;
    if (videoDurationSec === null) {
        videoDurationSec = await probeDurationSec(videoPath);
        if (videoDurationSec === null) {
            console.warn(
                '[Assemble] Video duration unknown; falling back to -shortest (output may be trimmed to the audio length).'
            );
        }
    }
    const durStr = videoDurationSec !== null ? videoDurationSec.toFixed(3) : null;
    console.log(`[Assemble] Target output duration: ${durStr !== null ? `${durStr}s` : 'unknown'}`);

    /**
     * Pads a stream with silence and hard-bounds it to the video length, so the
     * mixed audio is exactly as long as the picture: never shorter (would let
     * `-shortest` clip the video) and never longer (would append trailing black).
     */
    const fitToVideo = (chain: string): string =>
        durStr !== null ? `apad,atrim=duration=${durStr},asetpts=N/SR/TB,${chain}` : chain;

    const LIMITER = 'alimiter=limit=0.95:level=false';

    const complexFilter: string[] = [];
    const inputs = [videoPath]; // index 0: [0:v] (copied), [0:a] (never mapped)

    const hasBackground = !!(backgroundAudioPath && fs.existsSync(backgroundAudioPath));

    // Ensure we have a continuous pre-stitched audio track to prevent Windows command-line limit issues
    const defaultStitchedPath = path.join(path.dirname(videoPath), `${jobId}_tts_full.wav`);
    let stitchedTrackPath = (fullTtsAudioPath && fs.existsSync(fullTtsAudioPath)) ? fullTtsAudioPath : null;
    if (!stitchedTrackPath && fs.existsSync(defaultStitchedPath)) {
        stitchedTrackPath = defaultStitchedPath;
    }
    if (!stitchedTrackPath && segments && segments.length > 0) {
        stitchedTrackPath = await stitchSegments(segments, videoPath, jobId, videoDurationSec, '[Assemble]');
    }

    if (stitchedTrackPath) {
        console.log(`[Assemble] Using continuous speech track: ${stitchedTrackPath}`);
        const speechIndex = inputs.push(stitchedTrackPath) - 1; // [n:a]

        if (hasBackground) {
            const bgIndex = inputs.push(backgroundAudioPath) - 1; // [m:a]
            console.log(`[Assemble] Using clean background track from Demucs: ${backgroundAudioPath}`);
            complexFilter.push(
                `[${bgIndex}:a]${fitToVideo(`volume=${BG_GAIN}`)}[bg_music]`,
                `[${speechIndex}:a]${fitToVideo(`volume=${SPEECH_GAIN}`)}[fg_speech]`,
                // normalize=0 is mandatory: amix defaults to normalize=true, which
                // halves the sum and costs ~6 dB on both stems. The fallback branch
                // below set it explicitly; this branch did not.
                `[bg_music][fg_speech]amix=inputs=2:duration=longest:normalize=0:dropout_transition=0,${LIMITER}[mixed]`
            );
        } else {
            complexFilter.push(
                `[${speechIndex}:a]${fitToVideo(`volume=${SPEECH_GAIN}`)},${LIMITER}[mixed]`
            );
        }
    } else {
        // Fallback: individual segment delay
        let bgIndex = -1;
        if (hasBackground) {
            bgIndex = inputs.push(backgroundAudioPath) - 1;
            console.log(`[Assemble] Using clean background track from Demucs: ${backgroundAudioPath}`);
        }

        const ttsStartIndex = inputs.length;
        const placed: any[] = segments.filter((s: any) => s?.tts_audio_url && fs.existsSync(s.tts_audio_url));
        placed.forEach((seg: any, i: number) => {
            inputs.push(seg.tts_audio_url);
            const delay = Math.max(0, seg.start_ms || 0);
            complexFilter.push(`[${ttsStartIndex + i}:a]adelay=${delay}|${delay}[aud${i}]`);
        });

        if (placed.length > 0) {
            const mixInputs = placed.map((_: any, i: number) => `[aud${i}]`).join('');
            complexFilter.push(`${mixInputs}amix=inputs=${placed.length}:normalize=0:dropout_transition=0[tts_voices]`);
        } else {
            complexFilter.push(`anullsrc=r=44100:cl=stereo[tts_voices]`);
        }

        if (hasBackground) {
            complexFilter.push(
                `[${bgIndex}:a]${fitToVideo(`volume=${BG_GAIN}`)}[bg_music]`,
                `[tts_voices]${fitToVideo(`volume=${SPEECH_GAIN}`)}[fg_speech]`,
                `[bg_music][fg_speech]amix=inputs=2:duration=longest:normalize=0:dropout_transition=0,${LIMITER}[mixed]`
            );
        } else {
            complexFilter.push(`[tts_voices]${fitToVideo(`volume=${SPEECH_GAIN}`)},${LIMITER}[mixed]`);
        }
    }

    return new Promise((resolve, reject) => {
        const cmd = ffmpeg();

        inputs.forEach(inp => cmd.input(inp));

        const outputOptions = [
            '-map 0:v:0',
            '-map [mixed]',
            '-c:v copy',
            '-c:a aac',
            '-b:a', AAC_BITRATE
        ];
        // With a known video length the audio is already padded/trimmed to match,
        // so `-t` is the safe cap. Without it, `-shortest` is the only option left
        // and the output is exposed to the audio-length problem described above.
        if (durStr !== null) {
            outputOptions.push('-t', durStr);
        } else {
            outputOptions.push('-shortest');
        }

        cmd.complexFilter(complexFilter)
            .outputOptions(outputOptions)
            .save(finalVideoPath)
            .on('end', () => {
                console.log(`[Assemble] Completed. Final video at ${finalVideoPath}`);
                resolve({ finalVideoPath, jobId, durationSec: videoDurationSec });
            })
            .on('error', (err) => {
                console.error(`[Assemble] Error:`, err);
                reject(err);
            });
    });

}, { connection: redisConnection });
