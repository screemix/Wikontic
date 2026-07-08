import React from 'react';

export const MetricBadge: React.FC<{value: string; label: string; tone?: 'blue' | 'green' | 'amber'}> = ({
  value,
  label,
  tone = 'blue',
}) => (
  <div className={`metricBadge metric-${tone}`}>
    <strong>{value}</strong>
    <span>{label}</span>
  </div>
);
