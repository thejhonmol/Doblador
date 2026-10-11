import crypto from 'crypto';
import { NextFunction, Request, Response } from 'express';
import { query } from './db';

const SESSION_COOKIE = 'zhowplus_session';
const SESSION_HOURS = Math.max(1, parseInt(process.env.AUTH_SESSION_HOURS || '168', 10));
const PASSWORD_MIN_LENGTH = Math.max(8, parseInt(process.env.AUTH_PASSWORD_MIN_LENGTH || '12', 10));
const AUTH_SECRET = process.env.AUTH_SECRET || '';

export function assertAuthConfiguration(): void {
    if (process.env.NODE_ENV === 'production' && AUTH_SECRET.length < 32) {
        throw new Error('AUTH_SECRET must contain at least 32 characters in production.');
    }
}

export interface AuthUser {
    id: string;
    email: string;
    createdAt: string;
}

export interface AuthenticatedRequest extends Request {
    authUser?: AuthUser;
    sessionToken?: string;
}

function normalizeEmail(email: unknown): string {
    return String(email || '').trim().toLowerCase();
}

function validateEmail(email: string): void {
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || email.length > 254) {
        throw new Error('A valid email address is required.');
    }
}

export function validatePassword(password: unknown): string {
    const value = String(password || '');
    if (value.length < PASSWORD_MIN_LENGTH) {
        throw new Error(`Password must contain at least ${PASSWORD_MIN_LENGTH} characters.`);
    }
    if (value.length > 256) throw new Error('Password is too long.');
    return value;
}

export function hashPassword(password: string): string {
    const salt = crypto.randomBytes(16);
    const derived = crypto.scryptSync(password, salt, 64, { N: 16384, r: 8, p: 1 });
    return `scrypt$16384$8$1$${salt.toString('base64')}$${derived.toString('base64')}`;
}

export function verifyPassword(password: string, encoded: string): boolean {
    try {
        const [algorithm, n, r, p, saltB64, hashB64] = encoded.split('$');
        if (algorithm !== 'scrypt') return false;
        const expected = Buffer.from(hashB64, 'base64');
        const actual = crypto.scryptSync(password, Buffer.from(saltB64, 'base64'), expected.length, {
            N: parseInt(n, 10), r: parseInt(r, 10), p: parseInt(p, 10),
        });
        return expected.length === actual.length && crypto.timingSafeEqual(expected, actual);
    } catch {
        return false;
    }
}

function tokenHash(token: string): string {
    if (AUTH_SECRET) return crypto.createHmac('sha256', AUTH_SECRET).update(token).digest('hex');
    return crypto.createHash('sha256').update(token).digest('hex');
}

function parseCookies(header: string | undefined): Record<string, string> {
    if (!header) return {};
    return header.split(';').reduce<Record<string, string>>((out, part) => {
        const idx = part.indexOf('=');
        if (idx <= 0) return out;
        const key = part.slice(0, idx).trim();
        const value = part.slice(idx + 1).trim();
        try { out[key] = decodeURIComponent(value); } catch { out[key] = value; }
        return out;
    }, {});
}

export function extractSessionToken(req: Request): string | null {
    const auth = req.get('authorization');
    if (auth?.startsWith('Bearer ')) {
        const token = auth.slice(7).trim();
        if (token) return token;
    }
    return parseCookies(req.headers.cookie)[SESSION_COOKIE] || null;
}

function userFromRow(row: any): AuthUser {
    return {
        id: row.id,
        email: row.email,
        createdAt: new Date(row.created_at).toISOString(),
    };
}

export async function createUser(emailInput: unknown, passwordInput: unknown): Promise<AuthUser> {
    const email = normalizeEmail(emailInput);
    validateEmail(email);
    const password = validatePassword(passwordInput);
    try {
        const result = await query(
            `INSERT INTO users (email, password_hash) VALUES ($1, $2)
             RETURNING id, email, created_at`,
            [email, hashPassword(password)]
        );
        return userFromRow(result.rows[0]);
    } catch (err: any) {
        if (err?.code === '23505') throw Object.assign(new Error('An account with that email already exists.'), { statusCode: 409 });
        throw err;
    }
}

export async function authenticateUser(emailInput: unknown, passwordInput: unknown): Promise<AuthUser | null> {
    const email = normalizeEmail(emailInput);
    const password = String(passwordInput || '');
    if (!email || !password) return null;
    const result = await query(
        'SELECT id, email, password_hash, created_at FROM users WHERE email = $1 AND disabled_at IS NULL',
        [email]
    );
    if (result.rowCount !== 1 || !verifyPassword(password, result.rows[0].password_hash)) return null;
    return userFromRow(result.rows[0]);
}

export async function createSession(userId: string): Promise<{ token: string; expiresAt: string }> {
    const token = crypto.randomBytes(32).toString('base64url');
    const expiresAt = new Date(Date.now() + SESSION_HOURS * 60 * 60 * 1000);
    await query(
        `INSERT INTO auth_sessions (user_id, token_hash, expires_at) VALUES ($1, $2, $3)`,
        [userId, tokenHash(token), expiresAt]
    );
    return { token, expiresAt: expiresAt.toISOString() };
}

export async function revokeSession(token: string | null): Promise<void> {
    if (!token) return;
    await query('DELETE FROM auth_sessions WHERE token_hash = $1', [tokenHash(token)]);
}

export function setSessionCookie(res: Response, token: string): void {
    res.cookie(SESSION_COOKIE, token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: SESSION_HOURS * 60 * 60 * 1000,
        path: '/',
    });
}

export function clearSessionCookie(res: Response): void {
    res.clearCookie(SESSION_COOKIE, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        path: '/',
    });
}

export async function requireAuth(req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> {
    try {
        const token = extractSessionToken(req);
        if (!token) {
            res.status(401).json({ error: 'Authentication required.' });
            return;
        }
        const result = await query(
            `SELECT u.id, u.email, u.created_at
             FROM auth_sessions s
             JOIN users u ON u.id = s.user_id
             WHERE s.token_hash = $1 AND s.expires_at > NOW() AND u.disabled_at IS NULL`,
            [tokenHash(token)]
        );
        if (result.rowCount !== 1) {
            clearSessionCookie(res);
            res.status(401).json({ error: 'Session expired or invalid.' });
            return;
        }
        req.authUser = userFromRow(result.rows[0]);
        req.sessionToken = token;
        next();
    } catch (err) {
        next(err);
    }
}
