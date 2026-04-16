import React from "react";

interface ImageModalProps {
  src: string;
  alt: string;
  onClose: () => void;
}

export default function ImageModal({ src, alt, onClose }: ImageModalProps) {
  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 cursor-zoom-out"
      onClick={onClose}
    >
      <button 
        className="absolute top-4 right-4 text-white text-2xl hover:text-gray-300 transition-colors"
        onClick={onClose}
      >
        ✕
      </button>
      <img 
        src={src} 
        alt={alt} 
        className="max-w-full max-h-full rounded-lg object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
