import test from 'node:test';
import assert from 'node:assert/strict';
import {
    assertSpeechSynthesisComplete,
    durationAction,
    nextIncompleteStage,
    normalizeContextDiscovery,
    syllableBudget,
    ttsFailureAction,
} from '../src/pipeline-policy';

test('credential failures move to the next Fish Audio key', () => {
    assert.equal(ttsFailureAction(401), 'next-key');
    assert.equal(ttsFailureAction(402), 'next-key');
    assert.equal(ttsFailureAction(403), 'next-key');
});

test('transient failures retry and payload failures stop', () => {
    assert.equal(ttsFailureAction(429), 'retry');
    assert.equal(ttsFailureAction(200), 'retry');
    assert.equal(ttsFailureAction(503), 'retry');
    assert.equal(ttsFailureAction(400), 'stop');
});

test('optional Gemini context fields receive safe defaults', () => {
    assert.deepEqual(normalizeContextDiscovery({ detectedDomain: 'technology', glossaryDNT: [] }), {
        detectedDomain: 'technology',
        overallTone: 'natural',
        glossaryDNT: [],
        transcriptionCorrections: [],
    });
});

test('one missing dialogue segment prevents final assembly', () => {
    assert.doesNotThrow(() => assertSpeechSynthesisComplete(0));
    assert.throws(
        () => assertSpeechSynthesisComplete(1, 'mock provider error', 503),
        /1 dialogue segment\(s\).*final video was not assembled/
    );
});

test('recovery resumes at the first incomplete pipeline stage', () => {
    assert.equal(nextIncompleteStage([]), 'extract');
    assert.equal(nextIncompleteStage(['extract']), 'translate');
    assert.equal(nextIncompleteStage(['extract', 'translate', 'tts']), 'assemble');
    assert.equal(nextIncompleteStage(['extract', 'translate', 'tts', 'assemble']), null);
});

test('duration policy rewrites outliers once and enforces the hard ceiling', () => {
    assert.equal(durationAction(0.70), 'rewrite-longer');
    assert.equal(durationAction(0.70, 1), 'accept');
    assert.equal(durationAction(1.05), 'accept');
    assert.equal(durationAction(1.12), 'compress');
    assert.equal(durationAction(1.22), 'rewrite-shorter');
    assert.equal(durationAction(1.22, 1), 'compress');
    assert.equal(durationAction(1.26, 1), 'fail');
});

test('syllable budget is explicit and safe when timestamps are unavailable', () => {
    assert.deepEqual(syllableBudget(null, 'Spanish'), { target: null, min: null, max: null });
    assert.deepEqual(syllableBudget(2, 'Spanish'), { target: 9, min: 6, max: 10 });
});
