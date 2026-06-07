import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {CheckCircle2} from 'lucide-react';
import {DedupMerge} from '../components/DedupMerge';
import {DocumentView} from '../components/DocumentView';
import {GraphView} from '../components/GraphView';
import {OntologyPass} from '../components/OntologyPass';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {TripletCard} from '../components/TripletCard';
import {
  compactGraphEdges,
  compactGraphNodes,
  highlightedFacts,
  methodDocument,
  methodTriplets,
  ontologyChecks,
} from '../data/animation1';

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const progress = (frame: number, from: number, to: number) =>
  interpolate(frame, [from, to], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

// Same spread-out grid layout used in Animation 2 — nodes never overlap at any size.
// Columns shifted to give edge labels clearance: complex→right, center-column→right,
// networks→right; risks moved up so "связаны с" becomes diagonal (longer path, label fits).
const finalGraphLayout: Record<string, {x: number; y: number}> = {
  requirements: {x: 0.08, y: 0.14},
  year:         {x: 0.3,  y: 0.14},
  project:      {x: 0.05, y: 0.5},
  complex:      {x: 0.285, y: 0.5},
  buildings:    {x: 0.51, y: 0.14},
  monitoring:   {x: 0.51, y: 0.5},
  parking:      {x: 0.51, y: 0.86},
  networks:     {x: 0.74, y: 0.5},
  risks:        {x: 0.91, y: 0.25},
};
const finalGraphNodes = compactGraphNodes.map((node) => ({
  ...node,
  ...(finalGraphLayout[node.id] ?? {}),
}));

export const TEXT_TO_GRAPH_FRAMES = 870;

export const Animation1_TextToGraph: React.FC = () => {
  const frame = useCurrentFrame();
  const activeFactCount = Math.min(highlightedFacts.length, Math.floor(frame / 22) + 1);
  const activeFactIds = highlightedFacts.slice(0, activeFactCount).map((fact) => fact.id);
  const factsRemain = progress(frame, 110, 150);
  const tripletsIn = progress(frame, 120, 180);
  const ontologyIn = progress(frame, 270, 455);
  const dedupIn = progress(frame, 455, 600);
  // Panel background fades in fast (opaque quickly, less "transparent" time).
  // Node reveal is separate and slower — each of 9 nodes gets ~18 frames to itself.
  const graphFadeIn = progress(frame, 600, 635);
  const graphReveal = progress(frame, 605, 770);
  const title =
    frame < 120
      ? 'Документы содержат факты'
      : frame < 270
        ? '1. Из текста извлекаются триплеты-кандидаты'
        : frame < 455
          ? '2. Верификация и согласование графа с онтологией'
          : frame < 625
            ? '3. Очистка и дедупликация графа'
            : 'Итог: информация в компактном и проверяемом графе знаний';

  // During step 2 (ontologyIn): document fades out and triplets slide left
  const docFadeForOntology = progress(frame, 265, 300);
  const ontologyFadeIn = progress(frame, 260, 305);
  const tripletsSlide = progress(frame, 270, 295);
  const tripletsLeft = interpolate(tripletsSlide, [0, 1], [840, 40]);
  const dedupFadeIn = progress(frame, 450, 470);
  const dedupSlide = progress(frame, 450, 485);
  const tripletsDedupScale = interpolate(dedupSlide, [0, 1], [1, 0.80]);
  const tripletsScale = interpolate(tripletsSlide, [0, 1], [1, 0.88]) * tripletsDedupScale;
  const tripletsOpacityDedup = interpolate(dedupSlide, [0, 1], [1, 0.40]);

  const ontologyDedupX = interpolate(dedupSlide, [0, 1], [0, -560]);
  const ontologyDedupScale = interpolate(dedupSlide, [0, 1], [1, 0.65]);
  const ontologyOpacityDedup = interpolate(dedupSlide, [0, 1], [1, 0.60]);

  return (
    <SceneLayout
      eyebrow="Метод"
      title={title}
    >
      <div className="methodStage">
        <div
          className="methodDocumentSlot"
          style={{
            opacity: graphFadeIn > 0 ? 0 : 1 - docFadeForOntology,
            transform: `translateX(${graphFadeIn * -70 - docFadeForOntology * 80}px) scale(${1 - graphFadeIn * 0.12})`,
          }}
        >
          <DocumentView
            lines={methodDocument}
            activeFactIds={activeFactIds}
            dimNonFacts={factsRemain > 0.45}
            closeup={1 - progress(frame, 40, 130)}
          />
        </div>

        <div
          className="methodTriplets"
          style={{
            left: `${tripletsLeft}px`,
            opacity: tripletsIn * tripletsOpacityDedup * (1 - graphFadeIn),
            transform: `scale(${tripletsScale})`,
            transformOrigin: 'top left',
          }}
        >
          {methodTriplets.map((triplet, index) => (
            <TripletCard
              key={`${triplet.subject}-${triplet.object}`}
              {...triplet}
              compact
              progress={clamp(tripletsIn * 1.35 - index * 0.13)}
            />
          ))}
        </div>

        <div
          className="methodOntology"
          style={{
            opacity: ontologyFadeIn * ontologyOpacityDedup * (1 - graphFadeIn),
            transform: `translateX(${(1 - ontologyIn) * 40 + ontologyDedupX}px) scale(${ontologyDedupScale})`,
            transformOrigin: 'top left',
          }}
        >
          <OntologyPass checks={ontologyChecks} progress={ontologyIn} />
        </div>

        <div
          className="methodDedup"
          style={{
            opacity: dedupFadeIn * (1 - graphFadeIn),
            transform: `translateY(${(1 - dedupFadeIn) * 24}px)`,
          }}
        >
          <DedupMerge progress={dedupIn} />
        </div>

        <div
          className="methodFinalGraph"
          style={{
            opacity: graphFadeIn,
            transform: `translateX(${(1 - graphFadeIn) * 36}px)`,
          }}
        >
          <Panel className="finalGraphPanel finalComparisonPanel">
            <div className="finalComparisonGrid">
              <div className="finalDocumentPanel">
                <div className="panelTitleRow compactTitleRow">
                  <CheckCircle2 size={22} />
                  <span>Исходный текст</span>
                </div>
                <div className="finalDocumentCompact">
                  <DocumentView
                    lines={methodDocument}
                    activeFactIds={highlightedFacts.map((fact) => fact.id)}
                    closeup={0}
                  />
                </div>
              </div>
              <div className="finalGraphColumn">
                <div className="panelTitleRow compactTitleRow">
                  <CheckCircle2 size={22} />
                  <span>Граф знаний</span>
                </div>
                <GraphView
                  nodes={finalGraphNodes}
                  edges={compactGraphEdges}
                  reveal={graphReveal}
                  showTypes={graphReveal > 0.55}
                  softReveal={true}
                  typeOutside={true}
                  nodeRadius={42}
                  nodeAspect={1.32}
                  labelFontSize={16}
                  typeFontSize={11}
                  edgeFontSize={13}
                  width={680}
                  height={400}
                />
              </div>
            </div>
            <div className="metricTransform">
              <div className="metricTransformBlock metricBefore">
                <span>Было</span>
                <strong>420</strong>
                <em>токенов</em>
              </div>
              <div className="metricTransformArrow">→</div>
              <div className="metricTransformBlock metricAfter">
                <span>Стало</span>
                <div className="structuredMetrics">
                  <strong>9</strong>
                  <em>сущностей и</em>
                  <strong>12</strong>
                  <em>триплетов</em>
                </div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </SceneLayout>
  );
};
