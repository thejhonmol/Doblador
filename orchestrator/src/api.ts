import { assertAuthConfiguration } from './auth';
import { query } from './db';
import { migrate } from './migrate';
import { startServer } from './server';

async function waitForDatabase(maxWaitMs = 60000): Promise<void> {
    const started = Date.now();
    let lastError = '';
    while (Date.now() - started < maxWaitMs) {
        try {
            await query('SELECT 1');
            return;
        } catch (err: any) {
            lastError = err.message;
            await new Promise((resolve) => setTimeout(resolve, 2000));
        }
    }
    throw new Error(`PostgreSQL not reachable after ${maxWaitMs}ms: ${lastError}`);
}

async function bootstrap(): Promise<void> {
    assertAuthConfiguration();
    await waitForDatabase();
    await migrate();
    const port = parseInt(process.env.API_PORT || '3000', 10);
    startServer(port);
}

bootstrap().catch((err) => {
    console.error('[API] Fatal startup error:', err);
    process.exit(1);
});
