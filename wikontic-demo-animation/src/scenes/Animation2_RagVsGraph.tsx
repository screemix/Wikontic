import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {MessageSquareText, Search, Split, Waypoints} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {PathHighlight} from '../components/PathHighlight';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {
  answerPathEdgeIds,
  answerPathNodeIds,
  internalPathText,
  ragChunks,
  ragQuestion,
  ragVsGraphEdges,
  ragVsGraphNodes,
  ragVsWikonticAnswer,
} from '../data/animation2';

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const progress = (frame: number, from: number, to: number) =>
  interpolate(frame, [from, to], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

export const RAG_VS_WIKONTIC_FRAMES = 660;

export const Animation2_RagVsGraph: React.FC = () => {
  const frame = useCurrentFrame();
  const questionIn = progress(frame, 0, 70);
  const panelsIn = progress(frame, 95, 170);
  const chunksIn = progress(frame, 145, 300);
  const graphIn = progress(frame, 155, 250);
  const pathIn = progress(frame, 250, 475);
  const answerIn = progress(frame, 485, 620);
  const visiblePathCount = Math.ceil(answerPathEdgeIds.length * pathIn);

  return (
    <SceneLayout
      eyebrow="RAG comparison"
      title="Сложный вопрос на несколько фактов"
      subtitle="RAG ищет фрагменты. Wikontic показывает явный путь между проверенными фактами."
      frameLabel="Multi-hop"
    >
      <div className="ragQuestionWrap" style={{opacity: questionIn, transform: `translateY(${(1 - questionIn) * 22}px)`}}>
        <MessageSquareText size={30} />
        <span>{ragQuestion}</span>
      </div>

      <div className="comparisonGrid" style={{opacity: panelsIn}}>
        <Panel className="comparisonPanel">
          <div className="comparisonHeader ragHeader">
            <Search size={24} />
            <span>RAG</span>
          </div>
          <div className="chunkStack">
            {ragChunks.map((chunk, index) => {
              const itemIn = clamp(chunksIn * 1.35 - index * 0.16);
              return (
                <div
                  key={chunk.title}
                  className="ragChunk"
                  style={{
                    opacity: itemIn,
                    transform: `translateY(${(1 - itemIn) * 22}px) rotate(${(index - 1.5) * 0.4}deg)`,
                  }}
                >
                  <strong>{chunk.title}</strong>
                  <p>{chunk.text}</p>
                </div>
              );
            })}
          </div>
          <div className="ragNote" style={{opacity: progress(frame, 300, 400)}}>
            <Split size={21} />
            <span>Связь между фактами нужно собрать заново</span>
          </div>
        </Panel>

        <Panel className="comparisonPanel wikonticPanel">
          <div className="comparisonHeader wikonticHeader">
            <Waypoints size={24} />
            <span>Wikontic</span>
          </div>
          <GraphView
            nodes={ragVsGraphNodes}
            edges={ragVsGraphEdges}
            reveal={graphIn}
            showTypes
            highlightedNodeIds={answerPathNodeIds.slice(0, visiblePathCount + 1)}
            highlightedEdgeIds={answerPathEdgeIds.slice(0, visiblePathCount)}
            width={900}
            height={470}
            muted={pathIn > 0.05}
          />
          <div className="internalPath" style={{opacity: progress(frame, 340, 500)}}>
            <div className="internalPathLabel">Внутреннее объяснение</div>
            <PathHighlight path={internalPathText} progress={pathIn} compact />
          </div>
        </Panel>
      </div>

      <div
        className="answerOverlay"
        style={{
          opacity: answerIn,
          transform: `translateY(${(1 - answerIn) * 34}px)`,
        }}
      >
        <strong>Ответ</strong>
        <p>{ragVsWikonticAnswer}</p>
        <div>RAG ищет фрагменты. Wikontic ищет путь между фактами.</div>
      </div>
    </SceneLayout>
  );
};

