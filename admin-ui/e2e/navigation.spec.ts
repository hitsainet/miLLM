import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('should load the dashboard page', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/miLLM/i);
    await expect(page.locator('text=Dashboard')).toBeVisible();
  });

  test('should navigate to Models page', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Models');
    await expect(page).toHaveURL(/.*models/);
    await expect(page.locator('h1')).toContainText(/Models/i);
  });

  test('should navigate to SAEs page', async ({ page }) => {
    await page.goto('/');
    await page.click('text=SAEs');
    await expect(page).toHaveURL(/.*saes/);
    await expect(page.locator('h1')).toContainText(/SAE/i);
  });

  test('should navigate to Steering page', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Steering');
    await expect(page).toHaveURL(/.*steering/);
    await expect(page.locator('h1')).toContainText(/Steering/i);
  });

  test('should navigate to Profiles page', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Profiles');
    await expect(page).toHaveURL(/.*profiles/);
    await expect(page.locator('h1')).toContainText(/Profile/i);
  });

  // ⚠ EXACT labels, not substrings. "Feature Monitor" and "Probe Monitors" both contain
  // "Monitor", so a substring click is ambiguous and Playwright resolves it by position — a test
  // that passes today and clicks the other page the moment the sidebar is reordered.
  //
  // The previous version of this test clicked `text=Monitoring`, which matched NEITHER: the
  // sidebar label was "Probe" at `/monitoring`.
  test('should navigate to the Feature Monitor page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: 'Feature Monitor', exact: true }).click();
    await expect(page).toHaveURL(/.*monitoring/);
  });

  test('should navigate to the Probe Monitors page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: 'Probe Monitors', exact: true }).click();
    await expect(page).toHaveURL(/.*probe-monitors/);
    await expect(page.locator('h1')).toContainText(/Probe Monitors/i);
  });

  test('the two monitoring pages are distinguishable by name', async ({ page }) => {
    // D8's whole reason: two pages both called "Probe" is a coin flip for an operator.
    await page.goto('/');
    await expect(page.getByRole('link', { name: 'Feature Monitor', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Probe Monitors', exact: true })).toBeVisible();
  });

  test('should navigate to Settings page', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Settings');
    await expect(page).toHaveURL(/.*settings/);
    await expect(page.locator('h1')).toContainText(/Settings/i);
  });

  test('should display status bar with system info', async ({ page }) => {
    await page.goto('/');
    // Status bar should show connection status
    await expect(page.locator('[data-testid="status-bar"]').or(page.locator('.status-bar')).or(page.locator('header'))).toBeVisible();
  });
});
