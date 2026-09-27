import { PYTHON_SERVICES_URL } from '../config';
import fs from 'fs';
import path from 'path';
import axios from 'axios';
import FormData from 'form-data';

/**
 * Tail margin added to the video duration so the stitch buffer is never shorter
 * than the source. Segments are placed at their original timestamps, so the video
 * length alone is enough; the margin only absorbs a final segment that overflows.
 */
const TAIL_MARGIN_SEC = 5;

/**
 * Length of the continuous speech buffer. The source video duration is
 * authoritative; the last segment end is only a floor for the degenerate case
 * where a segment sits beyond the video length.
 */
export function resolveTrackDurationSec(
    videoDurationSec: number | null | undefined,
    segments: any[]
): number {
    const lastEndSec = segments.length
        ? (segments[segments.length - 1]?.end_ms || 0) / 1000
        : 0;
    const floorSec = lastEndSec + TAIL_MARGIN_SEC;
    const realSec = typeof videoDurationSec === 'number' && videoDurationSec > 0
        ? videoDurationSec
        : 0;
    return Math.max(10, realSec || floorSec, floorSec);
}

/**
 * Asks the Python service to stitch the individual TTS clips into one continuous
 * 44.1 kHz track. Returns the path on success, or null if the service is
 * unreachable / the write failed (callers then fall back to per-segment mixing).
 */
export async function stitchSegments(
    segments: any[],
    videoPath: string,
    jobId: string,
    videoDurationSec: number | null | undefined,
    logPrefix: string
): Promise<string | null> {
    if (!segments || segments.length === 0) return null;

    const outputPath = path.join(path.dirname(videoPath), `${jobId}_tts_full.wav`);
    const totalDurationSec = resolveTrackDurationSec(videoDurationSec, segments);

    try {
        console.log(
            `${logPrefix} Stitching ${segments.length} segments into a ${totalDurationSec.toFixed(2)}s continuous track...`
        );
        const form = new FormData();
        form.append('segments_json', JSON.stringify(segments));
        form.append('total_duration_sec', totalDurationSec.toString());
        form.append('output_path', outputPath);

        const res = await axios.post(`${PYTHON_SERVICES_URL}/stitch-tts`, form, {
            headers: { ...form.getHeaders() },
            timeout: 30 * 60 * 1000
        });
        const data = res.data || {};

        if (!fs.existsSync(outputPath)) {
            console.warn(`${logPrefix} Stitch reported success but ${outputPath} is missing.`);
            return null;
        }

        console.log(
            `${logPrefix} Stitched track ready at ${outputPath} ` +
            `(${data.segments_processed} segs | natural ${data.natural_count} | ` +
            `compressed ${data.compressed_count} | resynced ${data.resynced_count} | ` +
            `max start drift ${data.max_start_drift_ms}ms | overrun ${data.overrun_ms}ms)`
        );
        if (typeof data.overrun_ms === 'number' && data.overrun_ms > 0) {
            console.warn(
                `${logPrefix} Warning: stitched audio runs ${data.overrun_ms}ms past the buffer. ` +
                'Dialogue is longer than the video; check the transcribed timestamps.'
            );
        }
        return outputPath;
    } catch (err: any) {
        console.warn(`${logPrefix} Warning: could not stitch continuous audio track: ${err.message}`);
        return null;
    }
}
