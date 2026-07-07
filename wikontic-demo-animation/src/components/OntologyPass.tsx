import React from 'react';
import {BadgeCheck, ScanSearch} from 'lucide-react';
import {colors} from '../theme';

type OntologyCheck = {
  source: string;
  result: string;
  tone: 'blue' | 'green' | 'amber' | 'violet';
};

type OntologyPassProps = {
  checks: readonly OntologyCheck[];
  progress: number;
  title?: string;
};

const toneColor: Record<OntologyCheck['tone'], string> = {
  blue: colors.blue,
  green: colors.green,
  amber: colors.amber,
  violet: colors.violet,
};

export const OntologyPass: React.FC<OntologyPassProps> = ({checks, progress, title = 'Онтологическая проверка'}) => {
  const scannerX = Math.max(0, Math.min(1, progress)) * 100;
  return (
    <div className="ontologyPass">
      <div className="ontologyHeader">
        <ScanSearch size={26} />
        <span>{title}</span>
      </div>
      <div className="ontologyGrid">
        {checks.map((check, index) => {
          const visible = progress > index * 0.17;
          return (
            <div
              key={`${check.source}-${check.result}`}
              className="ontologyRow"
              style={{
                opacity: visible ? 1 : 0.24,
                transform: `translateX(${visible ? 0 : -16}px)`,
                borderColor: visible ? toneColor[check.tone] : colors.faint,
              }}
            >
              <span>{check.source}</span>
              <strong>{check.result}</strong>
              {visible ? <BadgeCheck size={21} color={toneColor[check.tone]} /> : null}
            </div>
          );
        })}
      </div>
      <div className="ontologyScanner" style={{left: `${scannerX}%`}} />
    </div>
  );
};
