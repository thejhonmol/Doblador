import { ConnectionOptions } from 'bullmq';
import dotenv from 'dotenv';
dotenv.config();

export const redisConnection: ConnectionOptions = {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
};

export const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';

const rawFishKeys = [
    process.env.FISH_AUDIO_API_KEY,
    process.env.FISH_AUDIO_API_KEY_2,
    process.env.FISH_AUDIO_API_KEY_3,
];

export const FISH_AUDIO_API_KEYS: string[] = rawFishKeys
    .flatMap(k => (k ? k.split(',') : []))
    .map(k => k.trim())
    .filter(k => k.length > 10 && !k.includes('tu_api_key') && !k.includes('your_'));

export const FISH_AUDIO_API_KEY = FISH_AUDIO_API_KEYS[0] || '';
export const PYTHON_SERVICES_URL = process.env.PYTHON_SERVICES_URL || 'http://localhost:8000';

