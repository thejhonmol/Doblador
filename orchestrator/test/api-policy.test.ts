import test from 'node:test';
import assert from 'node:assert/strict';
import { createRateLimiter, parsePagination } from '../src/api-policy';

test('segment pagination has bounded defaults and rejects invalid values', () => {
    assert.deepEqual(parsePagination({}), { limit: 100, offset: 0 });
    assert.deepEqual(parsePagination({ limit: '25', offset: '50' }), { limit: 25, offset: 50 });
    assert.throws(() => parsePagination({ limit: '0' }), /limit must be/);
    assert.throws(() => parsePagination({ limit: '501' }), /limit must be/);
    assert.throws(() => parsePagination({ offset: '-1' }), /offset must be/);
    assert.throws(() => parsePagination({ offset: '1.5' }), /offset must be/);
});

test('auth rate limiter returns 429 after the configured request budget', () => {
    const limiter = createRateLimiter(2, 60_000);
    const headers: Record<string, string> = {};
    let statusCode = 200;
    let body: unknown;
    let nextCalls = 0;
    const req = { ip: '203.0.113.10', path: '/api/v1/auth/login', socket: {} } as any;
    const res = {
        setHeader(name: string, value: string) { headers[name] = value; },
        status(code: number) { statusCode = code; return this; },
        json(value: unknown) { body = value; return this; },
    } as any;
    const next = () => { nextCalls += 1; };

    limiter(req, res, next);
    limiter(req, res, next);
    limiter(req, res, next);

    assert.equal(nextCalls, 2);
    assert.equal(statusCode, 429);
    assert.deepEqual(body, { error: 'Too many authentication attempts. Try again later.' });
    assert.equal(headers['Retry-After'], '60');
});
