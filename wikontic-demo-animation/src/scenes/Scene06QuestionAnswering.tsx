import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {Search, Waypoints} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {answer, question} from '../data/sampleDocument';
import {qaPathEdgeIds, qaPathNodeIds, refinedEdges, refinedNodes} from '../data/graphAfter';

export const Scene06QuestionAnswering: React.FC = () => {
  const frame = useCurrentFrame();
  const pathProgress = interpolate(frame, [120, 270], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const answerReveal = interpolate(frame, [280, 390], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const highlightedEdges = pathProgress > 0.35 ? qaPathEdgeIds.slice(0, pathProgress > 0.72 ? 2 : 1) : [];

  return (
    <SceneLayout
      eyebrow="06 / Multi-hop QA"
      title="Для сложного вопроса ищем путь в графе"
      subtitle="Не утверждаем, что это замена RAG для всего: сила Wikontic в вопросах, где нужно связать несколько фактов."
      frameLabel="relations, not fragments"
    >
      <div style={{display: 'grid', gridTemplateColumns: '520px 1fr', gap: 34, height: '100%', alignItems: 'center'}}>
        <div style={{display: 'grid', gap: 24}}>
          <Panel style={{padding: 28}}>
            <div className="pipelinePill">
              <Search size={23} />
              Ordinary RAG
            </div>
            <div className="chunk">“...жилой комплекс включает систему...”</div>
            <div className="chunk">“...мониторинг относится к сетям...”</div>
            <div className="chunk">LLM должен собрать цепочку из фрагментов</div>
          </Panel>
          <Panel style={{padding: 28, borderColor: '#cbd9ff'}}>
            <div className="pipelinePill" style={{color: '#1748b5', background: '#dbe8ff'}}>
              <Waypoints size={23} />
              Wikontic
            </div>
            <p style={{fontSize: 24, lineHeight: 1.34, fontWeight: 760, margin: '20px 0 0'}}>{question}</p>
          </Panel>
        </div>
        <div style={{display: 'grid', gap: 22}}>
          <GraphView
            nodes={refinedNodes}
            edges={refinedEdges}
            reveal={1}
            showTypes
            highlightedNodeIds={pathProgress > 0.3 ? qaPathNodeIds : []}
            highlightedEdgeIds={highlightedEdges}
            muted={pathProgress > 0.3}
          />
          <div className="answerBox" style={{opacity: answerReveal, transform: `translateY(${(1 - answerReveal) * 24}px)`}}>
            {answer}
          </div>
        </div>
      </div>
    </SceneLayout>
  );
};
