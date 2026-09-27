import { Pool } from 'pg';
import dotenv from 'dotenv';
dotenv.config();

export const pool = new Pool({
    user: process.env.DB_USER || 'doblador',
    host: process.env.DB_HOST || 'localhost',
    database: process.env.DB_NAME || 'doblador_db',
    password: process.env.DB_PASSWORD || 'doblador_password',
    port: parseInt(process.env.DB_PORT || '5432', 10),
});

export const query = (text: string, params?: any[]) => pool.query(text, params);
