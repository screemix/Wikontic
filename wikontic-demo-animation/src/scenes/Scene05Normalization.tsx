import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {GitMerge} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {dirtyEdges, dirtyNodes} from '../data/graphBefore';
import {aliasMerges, refinedEdges, refinedNodes} from '../data/graphAfter';

export const Scene05Normalization: React.FC = () => {
  const frame = useCurrentFrame();
  const merge = interpolate(frame, [45, 230], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const cleanReveal = interpolate(frame, [220, 320], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <SceneLayout
      eyebrow="05 / Entity normalization"
      title="Дубли схлопываются в канонические сущности"
      subtitle="Alias-aware deduplication сохраняет поверхностные формы, но строит компактный граф."
      frameLabel="aliases -> canonical"
    >
      <div style={{display: 'grid', gridTemplateColumns: '1fr 450px', gap: 34, height: '100%', alignItems: 'center'}}>
        <div style={{position: 'relative', height: 640}}>
          <div style={{position: 'absolute', inset: 0, opacity: 1 - cleanReveal}}>
            <GraphView nodes={dirtyNodes} edges={dirtyEdges} reveal={1} mergeProgress={merge} muted={merge > 0.4} />
          </div>
          <div style={{position: 'absolute', inset: 0, opacity: cleanReveal}}>
            <GraphView nodes={refinedNodes} edges={refinedEdges} reveal={1} showTypes />
          </div>
        </div>
        <Panel style={{padding: 30}}>
          <div className="pipelinePill" style={{color: '#1748b5', background: '#dbe8ff'}}>
            <GitMerge size={24} />
            Нормализация
          </div>
          <div style={{display: 'grid', gap: 18, marginTop: 26}}>
            {aliasMerges.map((mergeItem, index) => (
              <div key={mergeItem.to} style={{padding: 18, borderRadius: 8, background: '#f4f7fb', fontSize: 23, lineHeight: 1.35, fontWeight: 720}}>
                <span style={{color: '#647084'}}>{mergeItem.from.join(' + ')}</span>
                <span style={{color: '#2f6df6'}}> {'->'} </span>
                <span>{mergeItem.to}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </SceneLayout>
  );
};
