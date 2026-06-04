import React from 'react';
import {GitMerge} from 'lucide-react';
import {colors} from '../theme';

type DedupMergeProps = {
  progress: number;
};

const lerp = (from: number, to: number, progress: number) => from + (to - from) * progress;

export const DedupMerge: React.FC<DedupMergeProps> = ({progress}) => {
  const p = Math.max(0, Math.min(1, progress));
  const aliases = [
    {label: 'ЖК', x: 190, y: 170},
    {label: 'жилой комплекс', x: 380, y: 96},
    {label: 'объект', x: 560, y: 170},
  ];
  const target = {x: 380, y: 280};
  return (
    <div className="dedupPanel">
      <div className="dedupHeader">
        <GitMerge size={25} />
        <span>Очистка и дедупликация</span>
      </div>
      <svg viewBox="0 0 760 430" className="dedupSvg" role="img">
        <defs>
          <filter id="dedupGlow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="7" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {aliases.map((alias) => {
          const x = lerp(alias.x, target.x, p);
          const y = lerp(alias.y, target.y, p);
          return (
            <g key={alias.label} opacity={1 - p * 0.55}>
              <line x1={x} y1={y} x2={target.x} y2={target.y} stroke={colors.faint} strokeWidth="3" />
              <rect x={x - 92} y={y - 28} width="184" height="56" rx="8" fill="#fff" stroke={colors.amber} strokeWidth="3" />
              <text x={x} y={y + 7} textAnchor="middle" className="mergeAlias">
                {alias.label}
              </text>
            </g>
          );
        })}
        <g filter={p > 0.6 ? 'url(#dedupGlow)' : undefined}>
          <rect
            x={target.x - 142}
            y={target.y - 42}
            width="284"
            height="84"
            rx="10"
            fill={p > 0.6 ? colors.greenSoft : '#fff'}
            stroke={colors.green}
            strokeWidth="4"
          />
          <text x={target.x} y={target.y - 2} textAnchor="middle" className="mergeCanonical">
            жилой комплекс
          </text>
          <text x={target.x} y={target.y + 26} textAnchor="middle" className="mergeTags">
            aliases: ЖК · объект
          </text>
        </g>
        <g opacity={Math.max(0, p - 0.35) / 0.65}>
          <rect x="258" y="358" width="244" height="44" rx="8" fill={colors.blueSoft} stroke={colors.blue} />
          <text x="380" y="386" textAnchor="middle" className="mergeFooter">
            каноническая сущность
          </text>
        </g>
      </svg>
    </div>
  );
};
