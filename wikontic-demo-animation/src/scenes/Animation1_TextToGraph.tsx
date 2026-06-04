import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {CheckCircle2} from 'lucide-react';
import {DedupMerge} from '../components/DedupMerge';
import {DocumentView} from '../components/DocumentView';
import {GraphView} from '../components/GraphView';
import {HighlightedFact} from '../components/HighlightedFact';
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

export const TEXT_TO_GRAPH_FRAMES = 810;

export const Animation1_TextToGraph: React.FC = () => {
  const frame = useCurrentFrame();
  const activeFactCount = Math.min(highlightedFacts.length, Math.floor(frame / 22) + 1);
  const activeFactIds = highlightedFacts.slice(0, activeFactCount).map((fact) => fact.id);
  const factsRemain = progress(frame, 110, 195);
  const tripletsIn = progress(frame, 210, 350);
  const ontologyIn = progress(frame, 360, 505);
  const dedupIn = progress(frame, 505, 635);
  const graphIn = progress(frame, 625, 760);
  const title =
    frame < 120
      ? 'Документ содержит факты'
      : frame < 210
        ? 'Из текста выделяются факты'
        : frame < 360
          ? '1. Кандидатные триплеты'
          : frame < 505
            ? '2. Проверка на онтологию / верификация'
            : frame < 635
              ? '3. Очистка и дедупликация'
              : 'Вся информация представлена компактно в графе';

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
            opacity: graphIn > 0.45 ? 0.22 : 1,
            transform: `translateX(${graphIn * -70}px) scale(${1 - graphIn * 0.12})`,
          }}
        >
          <DocumentView
            lines={methodDocument}
            activeFactIds={activeFactIds}
            dimNonFacts={factsRemain > 0.45}
            closeup={1 - progress(frame, 40, 130)}
          />
        </div>

        <div className="floatingFacts" style={{opacity: factsRemain * (1 - graphIn)}}>
          {highlightedFacts.map((fact, index) => (
            <HighlightedFact
              key={fact.id}
              tone={fact.tone}
              progress={clamp(factsRemain * 1.25 - index * 0.11)}
            >
              {fact.text}
            </HighlightedFact>
          ))}
        </div>

        <div className="methodTriplets" style={{opacity: tripletsIn * (1 - ontologyIn) * (1 - graphIn)}}>
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
            opacity: ontologyIn * (1 - dedupIn) * (1 - graphIn),
            transform: `translateX(${(1 - ontologyIn) * 40}px)`,
          }}
        >
          <OntologyPass checks={ontologyChecks} progress={ontologyIn} />
        </div>

        <div
          className="methodDedup"
          style={{
            opacity: dedupIn * (1 - graphIn),
            transform: `translateY(${(1 - dedupIn) * 24}px)`,
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
