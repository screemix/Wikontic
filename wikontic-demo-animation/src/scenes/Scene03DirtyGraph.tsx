import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {AlertTriangle} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {dirtyEdges, dirtyNodes} from '../data/graphBefore';

export const Scene03DirtyGraph: React.FC = () => {
  const frame = useCurrentFrame();
  const reveal = interpolate(frame, [20, 160], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <SceneLayout
      eyebrow="03 / Raw graph"
      title="Сырой граф шумный"
      subtitle="Дубли, разные названия и неясные связи мешают использовать граф как надежную базу фактов."
      frameLabel="duplicates + ambiguity"
    >
      <div style={{display: 'grid', gridTemplateColumns: '1fr 420px', gap: 34, height: '100%', alignItems: 'center'}}>
        <GraphView nodes={dirtyNodes} edges={dirtyEdges} reveal={reveal} />
        <Panel style={{padding: 30}}>
          <div className="pipelinePill" style={{color: '#8a5c00', background: '#fff2cc'}}>
            <AlertTriangle size={24} />
            Что нужно исправить
          </div>
          <div style={{display: 'grid', gap: 16, marginTop: 26, fontSize: 26, lineHeight: 1.35, color: '#172033', fontWeight: 720}}>
            <span>“ЖК”, “объект”, “жилой комплекс”</span>
            <span>“содержит”, “имеет в составе”</span>
            <span>“сети” против “инженерные сети”</span>
          </div>
        </Panel>
      </div>
    </SceneLayout>
  );
};
