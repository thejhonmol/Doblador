import test from 'node:test';
import assert from 'node:assert/strict';
import { extractSessionToken, hashPassword, validatePassword, verifyPassword } from '../src/auth';

test('password hashes are salted and verify without storing plaintext', () => {
    const password = 'correct horse battery staple';
    const first = hashPassword(password);
    const second = hashPassword(password);

    assert.notEqual(first, second);
    assert.equal(first.includes(password), false);
    assert.equal(verifyPassword(password, first), true);
    assert.equal(verifyPassword('wrong password', first), false);
});

test('password validation rejects weak secrets', () => {
    assert.throws(() => validatePassword('short'), /at least/);
    assert.equal(validatePassword('a sufficiently long passphrase'), 'a sufficiently long passphrase');
});

test('bearer auth takes precedence over the session cookie', () => {
    const request = {
        headers: { cookie: 'other=x; zhowplus_session=cookie-token' },
        get(name: string) {
            return name.toLowerCase() === 'authorization' ? 'Bearer bearer-token' : undefined;
        },
    } as any;

    assert.equal(extractSessionToken(request), 'bearer-token');
    request.get = () => undefined;
    assert.equal(extractSessionToken(request), 'cookie-token');
});
