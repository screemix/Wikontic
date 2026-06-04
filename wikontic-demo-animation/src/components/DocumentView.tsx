import React from 'react';
import {FileText} from 'lucide-react';
import type {DocumentLine} from '../data/animation1';

type DocumentViewProps = {
  lines: DocumentLine[];
  activeFactIds?: string[];
  dimNonFacts?: boolean;
  progress?: number;
  closeup?: number;
};

const toneByFact: Record<string, string> = {
  'requirements-date': 'hlAmber',
  buildings: 'hlBlue',
  parking: 'hlGreen',
  'monitoring-networks': 'hlViolet',
  'risk-control': 'hlBlue',
};

export const DocumentView: React.FC<DocumentViewProps> = ({
  lines,
  activeFactIds = [],
  dimNonFacts = false,
  progress = 1,
  closeup = 0,
}) => {
  const active = new Set(activeFactIds);
  return (
    <div
      className="documentView"
      style={{
        opacity: progress,
        transform: `scale(${1 + closeup * 0.08}) translateY(${closeup * 10}px)`,
      }}
    >
      <div className="documentToolbar">
        <FileText size={22} />
        <span>Документ</span>
      </div>
      <div className="documentHeading">Проектные требования к объекту</div>
      <div className="documentText">
        {lines.map((line, lineIndex) => (
          <p key={`line-${lineIndex}`}>
            {line.map((part, partIndex) => {
              if (!part.factId) {
                return (
                  <span
                    key={`${lineIndex}-${partIndex}`}
                    className={dimNonFacts ? 'docMutedText' : undefined}
                  >
                    {part.text}
                  </span>
                );
              }
              const isActive = active.has(part.factId);
              return (
                <span
                  key={`${lineIndex}-${partIndex}`}
                  className={`docHighlight ${toneByFact[part.factId] ?? 'hlBlue'}`}
                  style={{
                    opacity: active.size === 0 || isActive || dimNonFacts ? 1 : 0.42,
                    boxShadow: isActive ? '0 0 0 3px rgba(47, 109, 246, 0.18)' : undefined,
                  }}
                >
                  {part.text}
                </span>
              );
            })}
          </p>
        ))}
      </div>
    </div>
  );
};

