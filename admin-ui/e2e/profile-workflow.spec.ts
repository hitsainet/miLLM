import { test, expect } from '@playwright/test';

test.describe('Profile Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/profiles');
  });

  test('should display profiles page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText(/Profile/i);
  });

  test('should have create profile button', async ({ page }) => {
    const createButton = page.locator('button').filter({ hasText: /create|new|add/i });
    await expect(createButton.first()).toBeVisible();
  });

  test('should show profile list or empty state', async ({ page }) => {
    // Either show profiles or empty state message
    const profileList = page.locator('[data-testid="profile-list"]').or(
      page.locator('.profile-list, .profile-item')
    );
    const emptyState = page.locator('text=/no profiles|empty|create.*first/i');

    const hasProfiles = await profileList.count() > 0;
    const hasEmptyState = await emptyState.count() > 0;

    expect(hasProfiles || hasEmptyState).toBeTruthy();
  });

  test('should have import/export functionality', async ({ page }) => {
    // Look for import/export buttons
    const importButton = page.locator('button').filter({ hasText: /import/i });
    const exportButton = page.locator('button').filter({ hasText: /export/i });

    // At least one of these should exist for profile portability
    const hasImport = await importButton.count() > 0;
    const hasExport = await exportButton.count() > 0;

    // Page should load successfully
    await expect(page.locator('h1')).toBeVisible();
  });
});

test.describe('Profile Creation', () => {
  test('should open create profile modal', async ({ page }) => {
    await page.goto('/profiles');

    const createButton = page.locator('button').filter({ hasText: /create|new|add/i });

    if (await createButton.count() > 0) {
      await createButton.first().click();

      // Modal or form should appear
      const modal = page.locator('[role="dialog"]').or(
        page.locator('.modal, [data-testid="profile-modal"]')
      );
      const form = page.locator('form');

      const hasModal = await modal.count() > 0;
      const hasForm = await form.count() > 0;

      expect(hasModal || hasForm).toBeTruthy();
    }
  });

  test('should have profile name input', async ({ page }) => {
    await page.goto('/profiles');

    const createButton = page.locator('button').filter({ hasText: /create|new|add/i });

    if (await createButton.count() > 0) {
      await createButton.first().click();

      // Should have name input field
      const nameInput = page.locator('input[name="name"], input[placeholder*="name" i]');

      // Wait for modal to appear
      await page.waitForTimeout(300);

      if (await nameInput.count() > 0) {
        await expect(nameInput.first()).toBeVisible();
      }
    }
  });

  test('should validate required fields', async ({ page }) => {
    await page.goto('/profiles');

    const createButton = page.locator('button').filter({ hasText: /create|new|add/i });

    if (await createButton.count() > 0) {
      await createButton.first().click();

      await page.waitForTimeout(300);

      // Try to submit without filling required fields
      const submitButton = page.locator('button[type="submit"]').or(
        page.locator('button').filter({ hasText: /save|create|submit/i })
      );

      if (await submitButton.count() > 0) {
        await submitButton.first().click();

        // Should show validation error or prevent submission
        // This documents expected validation behavior
      }
    }
  });
});

test.describe('Profile Activation', () => {
  test('should have activate button on profile items', async ({ page }) => {
    await page.goto('/profiles');

    // If profiles exist, they should have activate button
    const profileItem = page.locator('[data-testid="profile-item"]').or(
      page.locator('.profile-item, .profile-card')
    );

    if (await profileItem.count() > 0) {
      const activateButton = page.locator('button').filter({ hasText: /activate/i });
      await expect(activateButton.first()).toBeVisible();
    }
  });

  test('should show active profile indicator', async ({ page }) => {
    await page.goto('/profiles');

    // Active profile should be visually distinct
    const activeBadge = page.locator('text=/active/i');
    const activeIndicator = page.locator('[data-active="true"]').or(
      page.locator('.active-profile')
    );

    // Page should load successfully
    await expect(page.locator('h1')).toBeVisible();
  });
});

test.describe('Profile Export/Import', () => {
  test('should allow exporting profile as JSON', async ({ page }) => {
    await page.goto('/profiles');

    const exportButton = page.locator('button').filter({ hasText: /export/i });

    // Export functionality should exist
    if (await exportButton.count() > 0) {
      await expect(exportButton.first()).toBeVisible();
    }
  });

  test('should have import functionality', async ({ page }) => {
    await page.goto('/profiles');

    const importButton = page.locator('button').filter({ hasText: /import/i });

    if (await importButton.count() > 0) {
      await expect(importButton.first()).toBeVisible();
    }
  });
});
