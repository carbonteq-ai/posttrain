import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: { baseURL: 'http://127.0.0.1:4173', viewport: { width: 1440, height: 1024 } },
  webServer: [
    { command: 'POSTTRAIN_OBSERVATORY_SOURCE=fixture uv run --package posttrain-observatory posttrain-observatory serve --port 7862', cwd: '../../..', port: 7862, reuseExistingServer: false },
    { command: 'OBSERVATORY_API_URL=http://127.0.0.1:7862 npm run dev -- --host 127.0.0.1 --port 4173 --strictPort', port: 4173, reuseExistingServer: false }
  ]
});
