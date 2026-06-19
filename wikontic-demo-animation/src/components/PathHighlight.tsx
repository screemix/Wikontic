import React from 'react';

type PathHighlightProps = {
  path: string;
  progress?: number;
  compact?: boolean;
  fontSize?: number;
};

export const PathHighlight: React.FC<PathHighlightProps> = ({
  path,
  progress = 1,
  compact = false,
  fontSize,
}) => {
  const parts = path.split(' -> ');
  const visibleCount = Math.max(1, Math.ceil(parts.length * Math.max(0, Math.min(1, progress))));
  return (
    <div
      className={`pathHighlight ${compact ? 'pathHighlightCompact' : ''}`}
      style={fontSize ? {fontSize} : undefined}
    >
      {parts.map((part, index) => {
        const visible = index < visibleCount;
        const isRelation = index % 2 === 1;
        return (
          <React.Fragment key={`${part}-${index}`}>
            <span className={isRelation ? 'pathRelation' : 'pathNode'} style={{opacity: visible ? 1 : 0.18}}>
              {part}
            </span>
            {index < parts.length - 1 ? (
              <span className="pathArrow" style={{opacity: index + 1 < visibleCount ? 1 : 0.18}}>
                →
              </span>
            ) : null}
          </React.Fragment>
        );
      })}
    </div>
  );
};
