import { PYTHON_SERVICES_URL } from '../config';
import fs from 'fs';
import path from 'path';
import axios from 'axios';
import FormData from 'form-data';

/**
 * Length of the continuous speech buffer. The source video duration is
 * authoritative. Adding a hidden tail concealed dialogue that would later be cut
 * by the video muxer, so the strict path uses the exact picture duration.
 */
export function resolveTrackDurationSec(
    videoDurationSec: number | null | undefined,
    segments: any[]
): number {
    const lastEndSec = segments.reduce(
        (max, segment) => Math.max(max, Number(segment?.end_ms || 0) / 1000),
        0
    );
    const realSec = typeof videoDurationSec === 'number' && videoDurationSec > 0
        ? videoDurationSec
        : 0;
    return Math.max(0.1, realSec || lastEndSec + 0.5);
}

export interface StitchResult {
    outputPath: string;
    reportPath: string;
    report: any;
}

/**
 * Asks the Python service to stitch the individual TTS clips into one continuous
 * 44.1 kHz track. Synchronization failures are terminal: silently falling back to
 * per-segment adelay would bypass every drift and overlap quality gate.
 */
export async function stitchSegments(
    segments: any[],
    videoPath: string,
    jobId: string,
    videoDurationSec: number | null | undefined,
    logPrefix: string
): Promise<StitchResult> {
    if (!Array.isArray(segments)) throw new Error(`${logPrefix} Cannot stitch a non-array segment payload.`);

    const outputPath = path.join(path.dirname(videoPath), `${jobId}_tts_full.wav`);
    const reportPath = path.join(path.dirname(videoPath), `${jobId}_sync_report.json`);
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
            throw new Error(`${logPrefix} Stitch reported success but ${outputPath} is missing.`);
        }

        if (!data.report || !Array.isArray(data.report.segments)) {
            throw new Error(`${logPrefix} Stitch response did not include the synchronization report.`);
        }
        const report = {
            ...data.report,
            job_id: jobId,
            generated_at: new Date().toISOString(),
        };
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');

        console.log(
            `${logPrefix} Stitched track ready at ${outputPath} ` +
            `(${data.segments_processed} segs | natural ${data.natural_count} | ` +
            `compressed ${data.compressed_count} | resynced ${data.resynced_count} | ` +
            `max start drift ${data.max_start_drift_ms}ms | overrun ${data.overrun_ms}ms)`
        );
        return { outputPath, reportPath, report };
    } catch (err: any) {
        const detail = err?.response?.data?.detail || err?.message || String(err);
        throw new Error(`${logPrefix} Strict synchronization failed: ${detail}`);
    }
}
