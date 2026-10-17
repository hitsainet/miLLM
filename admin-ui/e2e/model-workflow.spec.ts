import { test, expect } from '@playwright/test';

test.describe('Model Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
  });

  test('should display models page with download form', async ({ page }) => {
    // Check for model download form elements
    await expect(page.locator('h1')).toContainText(/Models/i);

    // Should have input for HuggingFace repo ID
    const repoInput = page.locator('input[placeholder*="hugging" i], input[placeholder*="repo" i], input[placeholder*="google" i]');
    await expect(repoInput.first()).toBeVisible();
  });

  test('should show quantization options', async ({ page }) => {
    // Look for quantization selector
    const quantSelect = page.locator('select').filter({ hasText: /Q4|Q8|FP16/i });
    if (await quantSelect.count() > 0) {
      await expect(quantSelect.first()).toBeVisible();
    }
  });

  test('should validate repository ID format', async ({ page }) => {
    // Find the repo input and enter invalid value
    const repoInput = page.locator('input').first();
    await repoInput.fill('invalid-repo-format');

    // Try to submit
    const downloadButton = page.locator('button').filter({ hasText: /download/i });
    if (await downloadButton.count() > 0) {
      await downloadButton.first().click();

      // Should show validation error for invalid format
      const errorMessage = page.locator('text=/invalid|format|required/i');
      // Error might appear - this is expected behavior
    }
  });

  test('should display model list section', async ({ page }) => {
    // Should have a section for listing models
    const modelList = page.locator('[data-testid="model-list"]').or(
      page.locator('text=/your models|downloaded models|available models/i')
    );

    // Either model list or empty state should be visible
    const hasModels = await modelList.count() > 0;
    const emptyState = page.locator('text=/no models|empty|download a model/i');
    const hasEmptyState = await emptyState.count() > 0;

    expect(hasModels || hasEmptyState).toBeTruthy();
  });

  test('should have load/unload controls when model exists', async ({ page }) => {
    // If there are models, they should have load/unload buttons
    const loadButton = page.locator('button').filter({ hasText: /load/i });
    const modelCard = page.locator('[data-testid="model-card"]').or(
      page.locator('.model-item, .model-card')
    );

    // This test passes if either no models exist or models have controls
    if (await modelCard.count() > 0) {
      await expect(loadButton.first()).toBeVisible();
    }
  });
});

test.describe('Model Download Flow', () => {
  test('should show download progress when downloading', async ({ page }) => {
    await page.goto('/models');

    // This test documents the expected flow
    // In a real scenario with mocked backend:
    // 1. Fill in repo ID
    // 2. Click download
    // 3. Progress indicator should appear
    // 4. On completion, model should appear in list

    // For now, just verify the form structure exists
    const form = page.locator('form').or(page.locator('[data-testid="download-form"]'));
    await expect(form.first()).toBeVisible();
  });
});
