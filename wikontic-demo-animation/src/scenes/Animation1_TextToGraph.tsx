import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {CheckCircle2} from 'lucide-react';
import {DedupMerge} from '../components/DedupMerge';
import {DocumentView} from '../components/DocumentView';
import {GraphView} from '../components/GraphView';
import {OntologyPass} from '../components/OntologyPass';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {TripletCard} from '../components/TripletCard';
import {getAnimation1Content} from '../data/animation1';
import type {Locale} from '../i18n/types';

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
export const TEXT_TO_GRAPH_FRAMES = 870;

type Animation1Props = {
  locale?: Locale;
};

export const Animation1_TextToGraph: React.FC<Animation1Props> = ({locale = 'ru'}) => {
  const content = getAnimation1Content(locale);
  const finalGraphNodes = content.compactGraphNodes.map((node) => ({
    ...node,
    ...(finalGraphLayout[node.id] ?? {}),
  }));
  const frame = useCurrentFrame();
  const activeFactCount = Math.min(content.highlightedFacts.length, Math.floor(frame / 22) + 1);
  const activeFactIds = content.highlightedFacts.slice(0, activeFactCount).map((fact) => fact.id);
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
      ? content.titles.document
      : frame < 270
        ? content.titles.triplets
        : frame < 455
          ? content.titles.ontology
          : frame < 625
            ? content.titles.dedup
            : content.titles.final;

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
      eyebrow={content.labels.eyebrow}
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
            lines={content.methodDocument}
            activeFactIds={activeFactIds}
            dimNonFacts={factsRemain > 0.45}
            closeup={1 - progress(frame, 40, 130)}
            toolbarLabel={content.labels.documentToolbar}
            heading={content.labels.documentHeading}
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
          {content.methodTriplets.map((triplet, index) => (
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
          <OntologyPass checks={content.ontologyChecks} progress={ontologyIn} title={content.labels.ontologyTitle} />
        </div>

        <div
          className="methodDedup"
          style={{
            opacity: dedupFadeIn * (1 - graphFadeIn),
            transform: `translateY(${(1 - dedupFadeIn) * 24}px)`,
          }}
        >
          <DedupMerge
            progress={dedupIn}
            title={content.labels.dedupTitle}
            aliases={content.dedupGroups[0].aliases.map((label, index) => [
              {label, x: 190, y: 155},
              {label, x: 380, y: 82},
              {label, x: 560, y: 155},
            ][index])}
            canonical={content.dedupGroups[0].canonical}
            aliasTags={[content.dedupGroups[0].aliases[0], content.dedupGroups[0].aliases[2]]}
            aliasesPrefix={content.labels.dedupAliasesPrefix}
            footer={content.labels.dedupFooter}
          />
        </div>

        <div
          className="methodFinalGraph"
          style={{
            opacity: graphFadeIn,
            transform: `translateX(${(1 - graphFadeIn) * 36}px)`,
          }}
        >
          <div className="methodFinalLayout">
            <Panel className="finalGraphPanel finalComparisonPanel">
              <div className="finalComparisonGrid">
                <div className="finalDocumentPanel">
                  <div className="panelTitleRow compactTitleRow">
                    <CheckCircle2 size={22} />
                    <span>{content.labels.sourceText}</span>
                  </div>
                  <div className="finalDocumentCompact">
                    <DocumentView
                      lines={content.methodDocument}
                      activeFactIds={content.highlightedFacts.map((fact) => fact.id)}
                      closeup={0}
                      toolbarLabel={content.labels.documentToolbar}
                      heading={content.labels.documentHeading}
                    />
                  </div>
                </div>
                <div className="finalGraphColumn">
                  <div className="panelTitleRow compactTitleRow">
                    <CheckCircle2 size={22} />
                    <span>{content.labels.graph}</span>
                  </div>
                  <GraphView
                    nodes={finalGraphNodes}
                    edges={content.compactGraphEdges}
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
                  <span>{content.labels.before}</span>
                  <strong>420</strong>
                  <em>{content.labels.tokens}</em>
                </div>
                <div className="metricTransformArrow">→</div>
                <div className="metricTransformBlock metricAfter">
                  <span>{content.labels.after}</span>
                  <div className="structuredMetrics">
                    <strong>9</strong>
                    <em>{content.labels.entitiesAnd}</em>
                    <strong>12</strong>
                    <em>{content.labels.triplets}</em>
                  </div>
                </div>
              </div>
            </Panel>
            <div className="finalGraphBenefits">
              {content.graphBenefits.map((benefit, index) => {
                const itemIn = clamp(graphReveal * 1.45 - index * 0.16);
                return (
                  <div
                    key={benefit}
                    className="finalGraphBenefit"
                    style={{
                      opacity: itemIn,
                      transform: `translateX(${(1 - itemIn) * 18}px)`,
                    }}
                  >
                    <CheckCircle2 size={24} />
                    <span>{benefit}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </SceneLayout>
  );
};
