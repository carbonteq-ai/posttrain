import { expect, test } from '@playwright/test';

test('opens a curated run and trace evaluation', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Select run Eval Violet River' }).click();
  await expect(page.getByRole('heading', { name: 'Evaluation evidence' })).toBeVisible();
  await page.getByRole('button', { name: 'Traces & evaluation' }).click();
  await expect(page.getByRole('heading', { name: 'Traces & evaluation' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Trace population' })).toBeVisible();
  await expect(page.getByText('12 of 12 traces')).toBeVisible();
});

test('explains a serving benchmark without falling back to generic evidence', async ({ page }) => {
  await page.goto('/');
  await page.getByText('Serve Cedar Point', { exact: true }).click();

  await expect(page.getByRole('heading', { name: 'Serving benchmark' })).toBeVisible();
  await expect(page.getByText('Constraint-relative result')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Product constraints' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Serving configuration' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Representative requests' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Concurrency points' })).toBeVisible();
  await expect(page.getByText('Generic evidence workspace')).toHaveCount(0);
});

test('keeps the serving decision readable at a narrow viewport', async ({ page }) => {
  await page.goto('/');
  await page.getByText('Serve Cedar Point', { exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });

  await expect(page.getByRole('heading', { name: 'Serving benchmark' })).toBeVisible();
  await expect(page.getByText('Aggregate output TPS', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Product constraints' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Concurrency points' })).toBeVisible();
});
