import { expect, test } from '@playwright/test';

test('opens a curated run and trace evaluation', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Optimization evidence' })).toBeVisible();
  await expect(page.getByText('Final loss')).toBeVisible();
  await page.getByRole('button', { name: 'Traces & evaluation' }).click();
  await expect(page.getByRole('heading', { name: 'Traces & evaluation' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Trace population' })).toBeVisible();
  await expect(page.getByText('12 of 12 traces')).toBeVisible();
});
