import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveTrackDurationSec } from '../src/workers/stitch';

test('known video duration is the exact strict stitch boundary', () => {
    const segments = [{ start_ms: 0, end_ms: 12_000 }];
    assert.equal(resolveTrackDurationSec(10, segments), 10);
});

test('unknown video duration falls back to the last dialogue with a small pad', () => {
    const segments = [{ start_ms: 0, end_ms: 2_000 }];
    assert.equal(resolveTrackDurationSec(null, segments), 2.5);
});
