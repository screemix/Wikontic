import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {GraphView} from '../components/GraphView';
import {OntologyScanner} from '../components/OntologyScanner';
import {SceneLayout} from '../components/SceneLayout';
import {dirtyEdges, dirtyNodes} from '../data/graphBefore';

export const Scene04OntologyRefinement: React.FC = () => {
  const frame = useCurrentFrame();
  const scanner = interpolate(frame, [35, 300], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const types = frame > 160;

  return (
    <SceneLayout
      eyebrow="04 / Ontology-aware refinement"
      title="Онтология проверяет типы и допустимые отношения"
      subtitle="Wikontic выбирает канонические отношения и отбрасывает связи, которые не проходят domain/range-проверку."
      frameLabel="type + relation checks"
    >
      <div style={{display: 'grid', gridTemplateColumns: '1fr 560px', gap: 34, height: '100%', alignItems: 'center'}}>
        <GraphView
          nodes={dirtyNodes.map((node) => ({
            ...node,
            type:
              node.kind === 'project'
                ? 'строительный объект'
                : node.kind === 'system'
                  ? 'инженерная система'
                  : node.kind === 'asset'
                    ? 'элемент объекта'
                    : node.kind === 'requirement'
                      ? 'требование'
                      : 'дата',
          }))}
          edges={dirtyEdges}
          reveal={1}
          showTypes={types}
          scannerProgress={scanner}
        />
        <OntologyScanner progress={scanner} />
      </div>
    </SceneLayout>
  );
};
