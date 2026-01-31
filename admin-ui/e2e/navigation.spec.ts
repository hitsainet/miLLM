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

  test('should navigate to Monitoring page', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Monitoring');
    await expect(page).toHaveURL(/.*monitoring/);
    await expect(page.locator('h1')).toContainText(/Monitor/i);
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
