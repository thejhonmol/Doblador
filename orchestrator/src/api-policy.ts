import { NextFunction, Request, Response } from 'express';

export interface Pagination {
    limit: number;
    offset: number;
}

export function parsePagination(
    query: Record<string, unknown>,
    defaultLimit = 100,
    maxLimit = 500
): Pagination {
    const rawLimit = Array.isArray(query.limit) ? query.limit[0] : query.limit;
    const rawOffset = Array.isArray(query.offset) ? query.offset[0] : query.offset;
    const parsedLimit = rawLimit === undefined ? defaultLimit : Number(rawLimit);
    const parsedOffset = rawOffset === undefined ? 0 : Number(rawOffset);

    if (!Number.isInteger(parsedLimit) || parsedLimit < 1 || parsedLimit > maxLimit) {
        throw Object.assign(new Error(`limit must be an integer between 1 and ${maxLimit}.`), { statusCode: 400 });
    }
    if (!Number.isInteger(parsedOffset) || parsedOffset < 0) {
        throw Object.assign(new Error('offset must be a non-negative integer.'), { statusCode: 400 });
    }
    return { limit: parsedLimit, offset: parsedOffset };
}

interface RateBucket {
    count: number;
    resetAt: number;
}

/**
 * Small fixed-window limiter for auth endpoints. It is deliberately process-local:
 * enough to stop simple brute force on a single API instance without another runtime
 * dependency. Multi-instance production should move this counter to Redis/the edge.
 */
export function createRateLimiter(maxRequests: number, windowMs: number) {
    const buckets = new Map<string, RateBucket>();
    let lastSweep = 0;

    return (req: Request, res: Response, next: NextFunction): void => {
        const now = Date.now();
        if (now - lastSweep > windowMs) {
            for (const [key, bucket] of buckets.entries()) {
                if (bucket.resetAt <= now) buckets.delete(key);
            }
            lastSweep = now;
        }

        const address = req.ip || req.socket?.remoteAddress || 'unknown';
        const key = `${req.path}:${address}`;
        let bucket = buckets.get(key);
        if (!bucket || bucket.resetAt <= now) {
            bucket = { count: 0, resetAt: now + windowMs };
            buckets.set(key, bucket);
        }

        bucket.count += 1;
        res.setHeader('X-RateLimit-Limit', String(maxRequests));
        res.setHeader('X-RateLimit-Remaining', String(Math.max(0, maxRequests - bucket.count)));
        res.setHeader('X-RateLimit-Reset', String(Math.ceil(bucket.resetAt / 1000)));

        if (bucket.count > maxRequests) {
            const retryAfter = Math.max(1, Math.ceil((bucket.resetAt - now) / 1000));
            res.setHeader('Retry-After', String(retryAfter));
            res.status(429).json({ error: 'Too many authentication attempts. Try again later.' });
            return;
        }
        next();
    };
}
