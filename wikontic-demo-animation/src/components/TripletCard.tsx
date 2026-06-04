import React from 'react';
import {Database, Link2} from 'lucide-react';

type TripletCardProps = {
  subject: string;
  relation: string;
  object: string;
  qualifier?: string;
  progress?: number;
  compact?: boolean;
};

export const TripletCard: React.FC<TripletCardProps> = ({
  subject,
  relation,
  object,
  qualifier,
  progress = 1,
  compact = false,
}) => {
  return (
    <div
      className={`tripletCard ${compact ? 'tripletCompact' : ''}`}
      style={{
        opacity: progress,
        transform: `translateY(${(1 - progress) * 36}px) scale(${0.96 + progress * 0.04})`,
      }}
    >
      <div className="tripletRow">
        <span>{subject}</span>
        <span className="tripletRelation">
          <Link2 size={compact ? 16 : 18} />
          {relation}
        </span>
        <span>{object}</span>
      </div>
      {qualifier ? (
        <div className="qualifier">
          <Database size={14} />
          {qualifier}
        </div>
      ) : null}
    </div>
  );
};
