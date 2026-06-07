import React from 'react';
import {AbsoluteFill, Sequence, interpolate, useCurrentFrame} from 'remotion';
import {Brain, DatabaseZap, Layers3, Network, Send, Waypoints} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {PathHighlight} from '../components/PathHighlight';
import {colors, font} from '../theme';
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

const FADE = 11;
const PHASES = {
  graph: 210,
  paths: 240,
  data: 270,
} as const;

const GRAPH_FROM = 0;
const PATHS_FROM = GRAPH_FROM + PHASES.graph;
const DATA_FROM = PATHS_FROM + PHASES.paths;

export const SYNTHETIC_DATA_FRAMES = DATA_FROM + PHASES.data;

const STAGE_W = 1540;
const GRAPH_W = 1420;
const GRAPH_H = 600;

const syntheticLayout: Record<string, {x: number; y: number}> = {
  requirements: {x: 0.1, y: 0.16},
  year: {x: 0.28, y: 0.16},
  project: {x: 0.08, y: 0.52},
  complex: {x: 0.3, y: 0.52},
  buildings: {x: 0.53, y: 0.16},
  monitoring: {x: 0.51, y: 0.52},
  parking: {x: 0.53, y: 0.82},
  networks: {x: 0.8, y: 0.52},
  risks: {x: 0.9, y: 0.22},
};

const graphNodes = syntheticGraphNodes.map((node) => ({
  ...node,
  ...(syntheticLayout[node.id] ?? {}),
}));

const pathColors: Record<string, string> = {
  'one-hop': colors.green,
  'two-hop': colors.blue,
  'three-hop': colors.amber,
};

const pathOffsets: Record<string, number> = {
  'one-hop': -13,
  'two-hop': 0,
  'three-hop': 13,
};

const answerColor = colors.violet;
const answerFill = colors.violetSoft;

const nodeById = new Map(graphNodes.map((node) => [node.id, node]));
const edgeById = new Map(syntheticGraphEdges.map((edge) => [edge.id, edge]));

const answerNode = nodeById.get(syntheticAnswerNodeId)!;

const labelLines = (label: string) => {
  const words = label.split(' ');
  if (label.length < 14 || words.length === 1) {
    return [label];
  }
  const midpoint = Math.ceil(words.length / 2);
  return [words.slice(0, midpoint).join(' '), words.slice(midpoint).join(' ')];
};

const Phase: React.FC<{duration: number; children: React.ReactNode}> = ({duration, children}) => {
  const frame = useCurrentFrame();
  const opacity = Math.min(
    interpolate(frame, [0, FADE], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    interpolate(frame, [duration - FADE, duration], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
  );
  return (
    <AbsoluteFill
      style={{
        opacity,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '62px 80px',
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

const Header: React.FC<{
  eyebrow: string;
  title: string;
  icon: React.ReactNode;
  subtitle?: string;
  progress: number;
}> = ({eyebrow, title, icon, subtitle, progress: p}) => (
  <div
    style={{
      width: STAGE_W,
      display: 'flex',
      alignItems: 'center',
      gap: 24,
      marginBottom: 24,
      opacity: p,
      transform: `translateY(${(1 - p) * 18}px)`,
    }}
  >
    <div
      style={{
        width: 112,
        height: 112,
        borderRadius: 26,
        display: 'grid',
        placeItems: 'center',
        flexShrink: 0,
        color: colors.blueDark,
        background: colors.blueSoft,
        border: '2px solid #cbd9ff',
        boxShadow: '0 22px 56px rgba(23, 32, 51, 0.12)',
      }}
    >
      {icon}
    </div>
    <div>
      <div
        style={{
          marginBottom: 8,
          color: colors.blue,
          fontSize: 20,
          fontWeight: 820,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}
      >
        {eyebrow}
      </div>
      <div style={{color: colors.ink, fontSize: 56, lineHeight: 1.05, fontWeight: 850}}>{title}</div>
      {subtitle ? (
        <div style={{marginTop: 8, color: colors.muted, fontSize: 25, lineHeight: 1.28, fontWeight: 650}}>
          {subtitle}
        </div>
      ) : null}
    </div>
  </div>
);

const GraphStage: React.FC<{
  reveal: number;
  muted?: boolean;
  highlightedNodeIds?: string[];
  highlightedEdgeIds?: string[];
  stageHeight?: number;
  graphHeight?: number;
  showTypes?: boolean;
  softReveal?: boolean;
  overlay?: React.ReactNode;
}> = ({
  reveal,
  muted = false,
  highlightedNodeIds = [],
  highlightedEdgeIds = [],
  stageHeight = 650,
  graphHeight = GRAPH_H,
  showTypes = true,
  softReveal = false,
  overlay,
}) => (
  <div
    style={{
      position: 'relative',
      width: STAGE_W,
      height: stageHeight,
      borderRadius: 8,
      background: '#ffffff',
      border: '1px solid #dbe3ef',
      boxShadow: '0 24px 70px rgba(23, 32, 51, 0.1)',
      display: 'grid',
      placeItems: 'center',
      overflow: 'hidden',
    }}
  >
    <GraphView
      nodes={graphNodes}
      edges={syntheticGraphEdges}
      reveal={reveal}
      softReveal={softReveal}
      transparentBg
      showTypes={showTypes}
      typeOutside
      muted={muted}
      highlightedNodeIds={highlightedNodeIds}
      highlightedEdgeIds={highlightedEdgeIds}
      nodeRadius={68}
      nodeAspect={1.28}
      labelFontSize={25}
      typeFontSize={18}
      edgeFontSize={22}
      width={GRAPH_W}
      height={graphHeight}
    />
    {overlay}
  </div>
);

const endpoint = (
  sourceId: string,
  targetId: string,
  graphHeight: number,
  trimFromSource: boolean,
  offset: number,
) => {
  const source = nodeById.get(sourceId)!;
  const target = nodeById.get(targetId)!;
  const sx = source.x * GRAPH_W;
  const sy = source.y * graphHeight;
  const tx = target.x * GRAPH_W;
  const ty = target.y * graphHeight;
  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const rx = 68 * 1.28;
  const ry = 68;
  const trim = 1 / Math.sqrt((ux / rx) ** 2 + (uy / ry) ** 2);
  const baseX = trimFromSource ? sx + ux * trim : tx - ux * trim;
  const baseY = trimFromSource ? sy + uy * trim : ty - uy * trim;
  return {
    x: baseX + -uy * offset,
    y: baseY + ux * offset,
  };
};

const AnswerNodeOverlay: React.FC<{progress?: number; graphHeight: number; showType?: boolean}> = ({
  progress: p = 1,
  graphHeight,
  showType = true,
}) => {
  const x = answerNode.x * GRAPH_W;
  const y = answerNode.y * graphHeight;
  const lines = labelLines(answerNode.label);
  const scale = 0.92 + 0.08 * clamp(p);
  const haloOpacity = 0.12 + 0.2 * clamp(p);
  return (
    <g opacity={p}>
      <g transform={`translate(${x} ${y}) scale(${scale}) translate(${-x} ${-y})`}>
        <ellipse
          cx={x}
          cy={y}
          rx={118}
          ry={92}
          fill={answerColor}
          opacity={haloOpacity}
        />
        <ellipse
          cx={x}
          cy={y}
          rx={103}
          ry={82}
          fill="none"
          stroke={answerColor}
          strokeWidth={4}
          strokeDasharray="12 9"
          opacity={0.72}
        />
      </g>
      <ellipse
        cx={x}
        cy={y}
        rx={68 * 1.28}
        ry={68}
        fill="#ffffff"
        stroke={answerColor}
        strokeWidth={10}
        filter="url(#answerGlow)"
      />
      <text x={x} y={y - (lines.length > 1 ? 13 : -2)} textAnchor="middle" className="nodeLabel" style={{fontSize: 25}}>
        {lines.map((line, index) => (
          <tspan key={line} x={x} dy={index === 0 ? 0 : 26}>
            {line}
          </tspan>
        ))}
      </text>
      {showType && answerNode.type ? (
        <text x={x} y={y + 94} textAnchor="middle" className="nodeType" style={{fontSize: 18}}>
          {answerNode.type}
        </text>
      ) : null}
    </g>
  );
};

const GraphOverlaySvg: React.FC<{children: React.ReactNode; graphHeight: number}> = ({children, graphHeight}) => (
  <svg
    className="syntheticGraphOverlay"
    viewBox={`0 0 ${GRAPH_W} ${graphHeight}`}
    preserveAspectRatio="xMidYMid meet"
  >
    <defs>
      <filter id="answerGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="8" result="blur" />
        <feColorMatrix
          in="blur"
          type="matrix"
          values="0 0 0 0 0.478 0 0 0 0 0.361 0 0 0 0 1 0 0 0 0.45 0"
          result="colorBlur"
        />
        <feMerge>
          <feMergeNode in="colorBlur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
    {children}
  </svg>
);

const MultiPathOverlay: React.FC<{progress: number; graphHeight: number}> = ({progress: p, graphHeight}) => (
  <GraphOverlaySvg graphHeight={graphHeight}>
    {syntheticPaths.map((path, pathIndex) => {
      const color = pathColors[path.id];
      const offset = pathOffsets[path.id];
      return (
        <g key={path.id} opacity={clamp(p * 1.3 - pathIndex * 0.16)}>
          {path.edgeIds.map((edgeId) => {
            const edge = edgeById.get(edgeId);
            if (!edge) {
              return null;
            }
            const start = endpoint(edge.source, edge.target, graphHeight, true, offset);
            const end = endpoint(edge.source, edge.target, graphHeight, false, offset);
            const len = Math.hypot(end.x - start.x, end.y - start.y);
            return (
              <line
                key={`${path.id}-${edgeId}`}
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke={color}
                strokeWidth={6}
                strokeLinecap="round"
                strokeDasharray={len}
                strokeDashoffset={len * (1 - clamp(p * 1.25 - pathIndex * 0.12))}
                opacity={0.82}
              />
            );
          })}
        </g>
      );
    })}
    <AnswerNodeOverlay graphHeight={graphHeight} showType={false} />
  </GraphOverlaySvg>
);

const AnswerOnlyOverlay: React.FC<{progress: number; graphHeight: number}> = ({progress: p, graphHeight}) => (
  <GraphOverlaySvg graphHeight={graphHeight}>
    <AnswerNodeOverlay progress={p} graphHeight={graphHeight} />
  </GraphOverlaySvg>
);

const AnswerCallout: React.FC<{progress: number; bottom?: number}> = ({progress: p, bottom = 58}) => (
  <div
    style={{
      position: 'absolute',
      right: 74,
      bottom,
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '16px 20px',
      borderRadius: 12,
      background: colors.violetSoft,
      color: '#5637c8',
      fontSize: 28,
      fontWeight: 820,
      boxShadow: '0 18px 42px rgba(23, 32, 51, 0.12)',
      opacity: p,
      transform: `translateY(${(1 - p) * 18}px)`,
    }}
  >
    <Layers3 size={28} />
    <span>ответ: инженерные сети</span>
  </div>
);

const GraphPhase: React.FC = () => {
  const frame = useCurrentFrame();
  const headerIn = progress(frame, 6, 36);
  const answerIn = progress(frame, 82, 138);
  return (
    <Phase duration={PHASES.graph}>
      <Header
        eyebrow="Синтетические данные"
        title="Граф становится источником данных для обучения"
        subtitle="Выбираем вершину-ответ в графе знаний"
        icon={<DatabaseZap size={56} />}
        progress={headerIn}
      />
      <div style={{position: 'relative'}}>
        <GraphStage
          reveal={1}
          muted={answerIn > 0.25}
          overlay={<AnswerOnlyOverlay progress={answerIn} graphHeight={GRAPH_H} />}
        />
        <AnswerCallout progress={answerIn} />
      </div>
    </Phase>
  );
};

const PathPhase: React.FC = () => {
  const frame = useCurrentFrame();
  const headerIn = progress(frame, 6, 34);
  const pathsIn = progress(frame, 46, 118);
  const legendIn = progress(frame, 80, 136);
  const pathGraphHeight = 470;

  return (
    <Phase duration={PHASES.paths}>
      <Header
        eyebrow="Контроль сложности"
        title="Выбираем пути разной длины"
        subtitle="Один и тот же ответ можно получить через 1-hop, 2-hop или 3-hop пути"
        icon={<Waypoints size={58} />}
        progress={headerIn}
      />
      <div style={{position: 'relative', width: STAGE_W}}>
        <div style={{position: 'relative'}}>
          <GraphStage
            reveal={1}
            muted
            stageHeight={525}
            graphHeight={pathGraphHeight}
            showTypes={false}
            overlay={<MultiPathOverlay progress={pathsIn} graphHeight={pathGraphHeight} />}
          />
          <AnswerCallout progress={pathsIn} bottom={36} />
        </div>
        <div
          className="syntheticPathGrid"
          style={{
            opacity: legendIn,
            transform: `translateY(${(1 - legendIn) * 22}px)`,
            marginTop: 20,
          }}
        >
          {syntheticPaths.map((path) => (
            <div key={path.id} className="syntheticPathCard" style={{borderColor: pathColors[path.id]}}>
              <div className="syntheticPathCardTop">
                <span style={{background: pathColors[path.id]}} />
                <strong>{path.difficulty}</strong>
              </div>
              <PathHighlight path={path.path} progress={1} compact />
            </div>
          ))}
        </div>
      </div>
    </Phase>
  );
};

const DataCard: React.FC<{
  question: string;
  answer: string;
  path: string;
  difficulty: string;
  progress: number;
}> = ({question, answer, path, difficulty, progress: p}) => (
  <div
    className="syntheticHeroCard"
    style={{
      opacity: p,
      transform: `translateY(${(1 - p) * 34}px) scale(${0.96 + p * 0.04})`,
    }}
  >
    <div className="syntheticHeroCardTop">
      <span>{difficulty}</span>
      <Layers3 size={22} />
    </div>
    <div className="syntheticHeroQuestion">{question}</div>
    <div className="syntheticHeroAnswer">{answer}</div>
    <PathHighlight path={path} progress={p} compact />
  </div>
);

const DataPhase: React.FC = () => {
  const frame = useCurrentFrame();
  const headerIn = progress(frame, 6, 36);
  const cardsIn = progress(frame, 38, 142);
  const datasetIn = progress(frame, 138, 206);
  const modelIn = progress(frame, 190, 250);

  return (
    <Phase duration={PHASES.data}>
      <Header
        eyebrow="Генерация"
        title="Пути по графу превращаются в QA-примеры"
        subtitle="Путь объясняет ответ на вопрос"
        icon={<Network size={58} />}
        progress={headerIn}
      />
      <div style={{position: 'relative', width: STAGE_W, height: 690}}>
        <div
          className="syntheticHeroCards"
          style={{
            opacity: 1 - datasetIn * 0.35,
            paddingTop: 18,
            transform: `translateY(${-datasetIn * 12}px) scale(${1 - datasetIn * 0.04})`,
            transformOrigin: 'center top',
          }}
        >
          {naturalQACards.map((card, index) => (
            <DataCard
              key={card.question}
              {...card}
              progress={clamp(cardsIn * 1.35 - index * 0.2)}
            />
          ))}
        </div>

        <div
          className="syntheticDatasetFlow"
          style={{
            opacity: datasetIn,
            transform: `translateY(${(1 - datasetIn) * 40}px)`,
          }}
        >
          <div className="syntheticDatasetStack">
            <div className="syntheticDatasetTitle">
              <DatabaseZap size={30} />
              <span>Проверяемая синтетика</span>
            </div>
            <div className="syntheticDatasetLabels">
              {datasetLabels.map((label, index) => (
                <span key={label} style={{opacity: clamp(datasetIn * 1.4 - index * 0.16)}}>
                  {label}
                </span>
              ))}
            </div>
          </div>
          <Send className="syntheticFlowArrow" size={44} />
          <div
            className="syntheticModelHero"
            style={{
              opacity: modelIn,
              transform: `scale(${0.9 + modelIn * 0.1})`,
            }}
          >
            <Brain size={58} />
            <strong>малая доменная модель</strong>
          </div>
        </div>
      </div>
    </Phase>
  );
};

export const Animation3_SyntheticData: React.FC = () => {
  return (
    <AbsoluteFill style={{background: '#ffffff', fontFamily: font.family}}>
      <Sequence from={GRAPH_FROM} durationInFrames={PHASES.graph}>
        <GraphPhase />
      </Sequence>
      <Sequence from={PATHS_FROM} durationInFrames={PHASES.paths}>
        <PathPhase />
      </Sequence>
      <Sequence from={DATA_FROM} durationInFrames={PHASES.data}>
        <DataPhase />
      </Sequence>
    </AbsoluteFill>
  );
};
