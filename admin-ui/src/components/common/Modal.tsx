import { useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';

export interface ModalProps {
  id: string;
  title?: string;
  children: ReactNode;
  footer?: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlay?: boolean;
  closeOnEscape?: boolean;
  isOpen?: boolean;
  onClose?: () => void;
}

const sizeStyles = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
};

export function Modal({
  id,
  title,
  children,
  footer,
  size = 'md',
  closeOnOverlay = true,
  closeOnEscape = true,
  isOpen: controlledIsOpen,
  onClose,
}: ModalProps) {
  const { modal, closeModal } = useUIStore();
  // Support both controlled and store-based modal state
  const isOpen = controlledIsOpen !== undefined ? controlledIsOpen : modal.id === id;
  const handleClose = onClose || closeModal;

  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && closeOnEscape && isOpen) {
        handleClose();
      }
    },
    [closeOnEscape, isOpen, handleClose]
  );

  const handleOverlayClick = useCallback(() => {
    if (closeOnOverlay) {
      handleClose();
    }
  }, [closeOnOverlay, handleClose]);

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleEscape]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={handleOverlayClick}
      />

      {/* Modal */}
      <div
        className={`
          relative w-full mx-4
          bg-slate-900 border border-slate-700/50 rounded-xl
          shadow-2xl animate-slide-in
          ${sizeStyles[size]}
        `}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700/50">
            <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
            <button
              onClick={handleClose}
              className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Body */}
        <div className="px-6 py-4 max-h-[60vh] overflow-y-auto">{children}</div>

        {/* Footer */}
        {footer && (
          <div className="px-6 py-4 border-t border-slate-700/50 flex justify-end gap-3">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

export default Modal;
