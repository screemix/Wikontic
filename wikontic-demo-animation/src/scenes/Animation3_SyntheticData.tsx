import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {BoxSelect, DatabaseZap, FileStack, Send} from 'lucide-react';
import {DatasetCard, SmallModelIcon} from '../components/DatasetCard';
import {GraphView} from '../components/GraphView';
import {PathHighlight} from '../components/PathHighlight';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {
  datasetLabels,
  naturalQACards,
  syntheticAnswerNodeId,
  syntheticGraphEdges,
  syntheticGraphNodes,
  syntheticPaths,
} from '../data/animation3';

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const progress = (frame: number, from: number, to: number) =>
  interpolate(frame, [from, to], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

export const SYNTHETIC_DATA_FRAMES = 660;

export const Animation3_SyntheticData: React.FC = () => {
  const frame = useCurrentFrame();
  const graphIn = progress(frame, 0, 95);
  const answerIn = progress(frame, 90, 175);
  const pathsIn = progress(frame, 175, 335);
  const cardsIn = progress(frame, 320, 485);
  const datasetIn = progress(frame, 470, 585);
  const modelIn = progress(frame, 555, 640);
  const activePathIndex = Math.min(syntheticPaths.length - 1, Math.floor(pathsIn * syntheticPaths.length));
  const activePath = syntheticPaths[activePathIndex];

  return (
    <SceneLayout
      eyebrow="Synthetic data"
      title={
        frame < 170
          ? 'Построен граф знаний'
          : frame < 320
            ? 'Выбираем пути разной сложности'
            : frame < 500
              ? 'Генерируем естественные вопросы по путям'
              : 'Граф → пути → естественные вопросы → синтетический датасет'
      }
      subtitle="Проверенный граф становится контролируемым источником QA и reasoning-примеров"
      frameLabel="Graph → Data"
    >
      <div className="syntheticStage">
        <Panel className="syntheticGraphPanel">
          <div className="panelTitleRow">
            <DatabaseZap size={24} />
            <span>{answerIn < 0.5 ? 'Построен граф знаний' : '1. Выбираем ответ — вершину графа'}</span>
          </div>
          <GraphView
            nodes={syntheticGraphNodes}
            edges={syntheticGraphEdges}
            reveal={graphIn}
            showTypes
            highlightedNodeIds={
              pathsIn > 0.05 ? activePath.nodeIds : answerIn > 0.2 ? [syntheticAnswerNodeId] : []
            }
            highlightedEdgeIds={pathsIn > 0.05 ? activePath.edgeIds : []}
            width={940}
            height={560}
            muted={pathsIn > 0.05}
          />
          <div className="selectedAnswer" style={{opacity: answerIn}}>
            <BoxSelect size={20} />
            <span>ответ: инженерные сети</span>
          </div>
        </Panel>

        <div className="syntheticRightRail">
          <Panel
            className="pathSampler"
            style={{
              opacity: pathsIn,
              transform: `translateX(${(1 - pathsIn) * 28}px)`,
            }}
          >
            <div className="panelTitleRow">
              <BoxSelect size={22} />
              <span>2. Пути разной сложности</span>
            </div>
            {syntheticPaths.map((path, index) => {
              const itemIn = clamp(pathsIn * 1.4 - index * 0.22);
              return (
                <div key={path.id} className="sampledPath" style={{opacity: itemIn}}>
                  <strong>{path.difficulty}</strong>
                  <PathHighlight path={path.path} progress={itemIn} compact />
                </div>
              );
            })}
          </Panel>

          <div className="qaCardRail" style={{opacity: cardsIn * (1 - datasetIn)}}>
            {naturalQACards.map((card, index) => (
              <DatasetCard
                key={card.question}
                {...card}
                progress={clamp(cardsIn * 1.35 - index * 0.18)}
              />
            ))}
          </div>

          <div
            className="datasetAssembly"
            style={{
              opacity: datasetIn,
              transform: `translateY(${(1 - datasetIn) * 22}px)`,
            }}
          >
            <Panel className="datasetStackPanel">
              <div className="panelTitleRow">
                <FileStack size={23} />
                <span>Проверяемая синтетика</span>
              </div>
              <div className="datasetLabelGrid">
                {datasetLabels.map((label, index) => (
                  <span key={label} style={{opacity: clamp(datasetIn * 1.5 - index * 0.18)}}>
                    {label}
                  </span>
                ))}
              </div>
            </Panel>
            <Send className="datasetArrow" size={34} />
            <SmallModelIcon progress={modelIn} />
          </div>
        </div>
      </div>
    </SceneLayout>
  );
};
