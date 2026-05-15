import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, children, footer }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 transition-opacity duration-300 ease-out data-[state=closed]:opacity-0">
      <div className="bg-slate-800 rounded-lg shadow-xl w-full max-w-md mx-auto p-6 border border-white/10 transform transition-all duration-300 ease-out data-[state=closed]:scale-95 data-[state=closed]:opacity-0">
        <div className="flex justify-between items-center border-b border-white/10 pb-3 mb-4">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="text-slate-300 mb-4">
          {children}
        </div>
        {footer && (
          <div className="border-t border-white/10 pt-3 flex justify-end space-x-2">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
};

export default Modal;
