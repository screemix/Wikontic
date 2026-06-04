import React from 'react';

type HighlightedFactProps = {
  children: React.ReactNode;
  tone?: 'blue' | 'green' | 'amber' | 'violet';
  progress?: number;
  compact?: boolean;
};

export const HighlightedFact: React.FC<HighlightedFactProps> = ({
  children,
  tone = 'blue',
  progress = 1,
  compact = false,
}) => (
  <div
    className={`highlightedFact fact-${tone} ${compact ? 'highlightedFactCompact' : ''}`}
    style={{
      opacity: progress,
      transform: `translateY(${(1 - progress) * 22}px) scale(${0.96 + progress * 0.04})`,
    }}
  >
    {children}
  </div>
);

