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

test('duration policy keeps short takes and compresses long takes without a ceiling', () => {
    assert.equal(durationAction(0.70), 'accept');
    assert.equal(durationAction(1), 'accept');
    for (const ratio of [1.005, 1.12, 1.22, 1.26, 2, 6, 12]) {
        assert.equal(durationAction(ratio), 'compress');
    }
});

test('invalid duration measurements still fail', () => {
    for (const ratio of [0, -1, NaN, Infinity]) {
        assert.equal(durationAction(ratio), 'fail');
    }
});

test('syllable budget is explicit and safe when timestamps are unavailable', () => {
    assert.deepEqual(syllableBudget(null, 'Spanish'), { target: null, min: null, max: null });
    assert.deepEqual(syllableBudget(2, 'Spanish'), { target: 9, min: 6, max: 10 });
});
