import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {BrainCircuit, FileStack} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {SceneLayout} from '../components/SceneLayout';
import {qaPathEdgeIds, qaPathNodeIds, refinedEdges, refinedNodes} from '../data/graphAfter';
import {syntheticCards} from '../data/syntheticCards';

export const Scene07SyntheticData: React.FC = () => {
  const frame = useCurrentFrame();
  const select = interpolate(frame, [20, 90], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <SceneLayout
      eyebrow="07 / Synthetic data"
      title="Проверяемый граф становится контролируемой синтетикой"
      subtitle="Выбираем подграф, генерируем QA/reasoning-карточки и обучаем малую доменную модель."
      frameLabel="graph -> training cards"
    >
      <div style={{display: 'grid', gridTemplateColumns: '620px 1fr 320px', gap: 30, height: '100%', alignItems: 'center'}}>
        <GraphView
          nodes={refinedNodes}
          edges={refinedEdges}
          reveal={1}
          showTypes
          highlightedNodeIds={select > 0.3 ? qaPathNodeIds : []}
          highlightedEdgeIds={select > 0.3 ? qaPathEdgeIds : []}
          muted={select > 0.3}
          width={760}
          height={620}
        />
        <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16}}>
          {syntheticCards.map((card, index) => {
            const progress = interpolate(frame, [80 + index * 28, 130 + index * 28], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            });
            return (
              <div
                className="syntheticCard"
                key={card.id}
                style={{
                  opacity: progress,
                  transform: `translateX(${(1 - progress) * -34}px)`,
                }}
              >
                <strong>{card.label}</strong>
                <span>{card.text}</span>
              </div>
            );
          })}
        </div>
        <div className="smallModel">
          <div>
            <BrainCircuit size={76} />
            <div>малая<br />доменная<br />модель</div>
          </div>
        </div>
        <div style={{position: 'absolute', left: 690, top: 620}} className="pipelinePill">
          <FileStack size={22} />
          проверяемые примеры
        </div>
      </div>
    </SceneLayout>
  );
};
