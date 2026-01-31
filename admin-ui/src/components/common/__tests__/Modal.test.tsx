import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Modal } from '../Modal';

// Mock the uiStore
vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    modal: { id: null },
    closeModal: vi.fn(),
  })),
}));

describe('Modal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders when isOpen is true', () => {
    render(
      <Modal id="test-modal" isOpen={true}>
        <p>Modal content</p>
      </Modal>
    );
    expect(screen.getByText('Modal content')).toBeInTheDocument();
  });

  it('does not render when isOpen is false', () => {
    render(
      <Modal id="test-modal" isOpen={false}>
        <p>Modal content</p>
      </Modal>
    );
    expect(screen.queryByText('Modal content')).not.toBeInTheDocument();
  });

  it('renders with title', () => {
    render(
      <Modal id="test-modal" isOpen={true} title="Test Title">
        <p>Content</p>
      </Modal>
    );
    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('renders footer', () => {
    render(
      <Modal id="test-modal" isOpen={true} footer={<button>Save</button>}>
        <p>Content</p>
      </Modal>
    );
    expect(screen.getByText('Save')).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Modal id="test-modal" isOpen={true} title="Test" onClose={onClose}>
        <p>Content</p>
      </Modal>
    );

    const closeButton = screen.getByRole('button');
    await user.click(closeButton);

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when overlay is clicked and closeOnOverlay is true', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Modal id="test-modal" isOpen={true} closeOnOverlay={true} onClose={onClose}>
        <p>Content</p>
      </Modal>
    );

    // Click on the backdrop (first element with inset-0)
    const backdrop = document.querySelector('.bg-black\\/60');
    if (backdrop) {
      await user.click(backdrop);
    }

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does not close when overlay is clicked and closeOnOverlay is false', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Modal id="test-modal" isOpen={true} closeOnOverlay={false} onClose={onClose}>
        <p>Content</p>
      </Modal>
    );

    const backdrop = document.querySelector('.bg-black\\/60');
    if (backdrop) {
      await user.click(backdrop);
    }

    expect(onClose).not.toHaveBeenCalled();
  });

  it('calls onClose when Escape key is pressed', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Modal id="test-modal" isOpen={true} closeOnEscape={true} onClose={onClose}>
        <p>Content</p>
      </Modal>
    );

    await user.keyboard('{Escape}');

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does not close when Escape is pressed and closeOnEscape is false', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Modal id="test-modal" isOpen={true} closeOnEscape={false} onClose={onClose}>
        <p>Content</p>
      </Modal>
    );

    await user.keyboard('{Escape}');

    expect(onClose).not.toHaveBeenCalled();
  });

  it('applies size styles', () => {
    render(
      <Modal id="test-modal" isOpen={true} size="lg">
        <p>Content</p>
      </Modal>
    );

    const modal = document.querySelector('.max-w-lg');
    expect(modal).toBeInTheDocument();
  });

  it('does not close when clicking inside modal content', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Modal id="test-modal" isOpen={true} onClose={onClose}>
        <p>Click me</p>
      </Modal>
    );

    await user.click(screen.getByText('Click me'));

    expect(onClose).not.toHaveBeenCalled();
  });
});
