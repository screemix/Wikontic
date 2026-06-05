import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {CheckCircle2} from 'lucide-react';
import {DedupMerge} from '../components/DedupMerge';
import {DocumentView} from '../components/DocumentView';
import {GraphView} from '../components/GraphView';
import {MetricBadge} from '../components/MetricBadge';
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

export const TEXT_TO_GRAPH_FRAMES = 800;

export const Animation1_TextToGraph: React.FC = () => {
  const frame = useCurrentFrame();
  const activeFactCount = Math.min(highlightedFacts.length, Math.floor(frame / 22) + 1);
  const activeFactIds = highlightedFacts.slice(0, activeFactCount).map((fact) => fact.id);
  const factsRemain = progress(frame, 110, 195);
  const tripletsIn = progress(frame, 120, 260);
  const ontologyIn = progress(frame, 270, 455);
  const dedupIn = progress(frame, 455, 625);
  const graphIn = progress(frame, 615, 750);
  const title =
    frame < 120
      ? 'Документ содержит факты'
      : frame < 270
        ? '1. Кандидатные триплеты'
        : frame < 455
          ? '2. Проверка на онтологию / верификация'
          : frame < 625
            ? '3. Очистка и дедупликация'
            : 'Вся информация представлена компактно в графе';

  // During step 2 (ontologyIn): document fades out and triplets slide left
  const docFadeForOntology = progress(frame, 265, 310);
  const ontologyFadeIn = progress(frame, 270, 305);
  const tripletsSlide = progress(frame, 270, 320);
  const tripletsLeft = interpolate(tripletsSlide, [0, 1], [840, 40]);
  const dedupFadeIn = progress(frame, 455, 490);
  const dedupSlide = progress(frame, 455, 500);
  const tripletsDedupScale = interpolate(dedupSlide, [0, 1], [1, 0.80]);
  const tripletsScale = interpolate(tripletsSlide, [0, 1], [1, 0.88]) * tripletsDedupScale;
  const tripletsOpacityDedup = interpolate(dedupSlide, [0, 1], [1, 0.60]);

  const ontologyDedupX = interpolate(dedupSlide, [0, 1], [0, -560]);
  const ontologyDedupScale = interpolate(dedupSlide, [0, 1], [1, 0.65]);
  const ontologyOpacityDedup = interpolate(dedupSlide, [0, 1], [1, 0.60]);

  return (
    <SceneLayout
      eyebrow="Метод"
      title={title}
      subtitle="Wikontic превращает неструктурированный текст в компактный проверяемый граф знаний"
      frameLabel="Text → KG"
    >
      <div className="methodStage">
        <div
          className="methodDocumentSlot"
          style={{
            opacity: graphIn > 0.45 ? 0.22 : 1 - docFadeForOntology,
            transform: `translateX(${graphIn * -70 - docFadeForOntology * 80}px) scale(${1 - graphIn * 0.12})`,
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
            opacity: tripletsIn * tripletsOpacityDedup * (1 - graphIn),
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
            opacity: ontologyFadeIn * ontologyOpacityDedup * (1 - graphIn),
            transform: `translateX(${(1 - ontologyIn) * 40 + ontologyDedupX}px) scale(${ontologyDedupScale})`,
            transformOrigin: 'top left',
          }}
        >
          <OntologyPass checks={ontologyChecks} progress={ontologyIn} />
        </div>

        <div
          className="methodDedup"
          style={{
            opacity: dedupFadeIn * (1 - graphIn),
            transform: `translateY(${(1 - dedupFadeIn) * 24}px)`,
          }}
        >
          <DedupMerge progress={dedupIn} />
        </div>

        <div
          className="methodFinalGraph"
          style={{
            opacity: graphIn,
            transform: `translateX(${(1 - graphIn) * 36}px)`,
          }}
        >
          <Panel className="finalGraphPanel">
            <div className="panelTitleRow">
              <CheckCircle2 size={24} />
              <span>Компактный проверяемый граф</span>
            </div>
            <GraphView
              nodes={compactGraphNodes}
              edges={compactGraphEdges}
              reveal={graphIn}
              showTypes={graphIn > 0.55}
              width={1040}
              height={560}
            />
            <div className="metricRow">
              <MetricBadge value="420" label="токенов в тексте" tone="amber" />
              <MetricBadge value="9 / 12" label="сущностей / триплетов" tone="green" />
            </div>
          </Panel>
        </div>
      </div>
    </SceneLayout>
  );
};
