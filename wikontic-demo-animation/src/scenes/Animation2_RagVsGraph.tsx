import React from 'react';
import {AbsoluteFill, Sequence, interpolate, useCurrentFrame} from 'remotion';
import {Search, Split, Waypoints} from 'lucide-react';
import {GraphView} from '../components/GraphView';
import {PathHighlight} from '../components/PathHighlight';
import {colors, font} from '../theme';
import {
  answerPathEdgeIds,
  answerPathNodeIds,
  internalPathText,
  ragChunks,
  ragQuestion,
  ragVsGraphEdges,
  ragVsGraphNodes,
} from '../data/animation2';

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const progress = (frame: number, from: number, to: number) =>
  interpolate(frame, [from, to], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

// Sequential phases. Each phase fades IN from white and OUT to white quickly,
// with a long hold so it reads on the first watch. Phases never overlap.
const FADE = 11; // ~0.37s — fast decay/appear to/from white
const PHASES = {
  rag: 580, // ~19.3s — links → concept highlight → flag missed → select → settle → takeaway
  wikontic: 620, // ~20.7s — same docs → unite → build graph → path → answer box
} as const;

const RAG_FROM = 0;
const WIKONTIC_FROM = RAG_FROM + PHASES.rag;

export const RAG_VS_WIKONTIC_FRAMES = WIKONTIC_FROM + PHASES.wikontic;

// Wraps phase content, handling the fast fade from/to white via opacity over
// the white composition background.
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
        padding: '64px 80px',
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

const BigIcon: React.FC<{tone: 'muted' | 'wikontic'; children: React.ReactNode}> = ({tone, children}) => {
  const accent = tone === 'wikontic' ? colors.blueDark : colors.muted;
  const bg = tone === 'wikontic' ? colors.blueSoft : '#eef1f6';
  return (
    <div
      style={{
        width: 132,
        height: 132,
        borderRadius: 28,
        background: bg,
        border: `2px solid ${tone === 'wikontic' ? '#cbd9ff' : '#dbe3ef'}`,
        display: 'grid',
        placeItems: 'center',
        color: accent,
        boxShadow: '0 22px 56px rgba(23, 32, 51, 0.12)',
      }}
    >
      {children}
    </div>
  );
};

// Fragment 2 is the relevant fragment RAG fails to retrieve — shown afterwards
// in pastel red to motivate the conclusion.
const missedChunk = ragChunks.find((chunk) => chunk.title === 'Фрагмент 2')!;

// Three semantic "themes" that emphasise the key words of the question.
type Tone = 'blue' | 'green' | 'violet';
const toneRgb: Record<Tone, {bg: string; line: string}> = {
  blue: {bg: '219,232,255', line: '47,109,246'},
  green: {bg: '223,248,236', line: '25,169,116'},
  violet: {bg: '238,233,255', line: '122,92,255'},
};

type Seg = {t: string; tone?: Tone};

// `on` (0→1) very gently fades in the tint + colored underline so the links
// read as a soft hint, not a hard highlight.
const SemText: React.FC<{segments: Seg[]; on: number; size: number; weight: number; color?: string}> = ({
  segments,
  on,
  size,
  weight,
  color,
}) => (
  <span style={{fontSize: size, lineHeight: 1.3, fontWeight: weight, color: color ?? colors.ink}}>
    {segments.map((seg, i) =>
      seg.tone ? (
        <span
          key={i}
          style={{
            borderRadius: 6,
            padding: '1px 5px',
            background: `rgba(${toneRgb[seg.tone].bg}, ${0.9 * on})`,
            boxShadow: `inset 0 -2px 0 rgba(${toneRgb[seg.tone].line}, ${on})`,
          }}
        >
          {seg.t}
        </span>
      ) : (
        <span key={i}>{seg.t}</span>
      ),
    )}
  </span>
);

const questionSegments: Seg[] = [
  {t: 'Что важно '},
  {t: 'проверить', tone: 'violet'},
  {t: ' перед вводом '},
  {t: 'жилого комплекса', tone: 'blue'},
  {t: ' в '},
  {t: 'эксплуатацию', tone: 'green'},
  {t: '?'},
];

// --- Filtering beat -------------------------------------------------------
// A corpus of identical grey document boxes (no titles), linked by arrows that
// show how the fragments relate. RAG then highlights the question's concepts
// inside them, flags the relevant fragment it will miss, picks the matching
// ones and discards the rest, and the kept fragments slide into a column.
const STAGE_W = 1400;
const STAGE_H = 700;
const COL_X = 80;
const COL_W = 1240;
const SLOT_Y = [20, 120, 220];
const MISSED_Y = 340;
const NOTE_Y = 500;
const BOX_H = 104; // assumed scatter height, used only for arrow anchoring

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

type CorpusBox = {
  text: string;
  col: number; // grid column (0..3)
  row: number; // grid row (0..2)
  slot?: number; // column slot if selected; undefined = discarded
  lost?: boolean; // the relevant fragment RAG fails to retrieve
  segments?: Seg[]; // question-concept highlighting (relevant fragments only)
  tx?: number; // throw direction x (discarded)
  ty?: number; // throw direction y (discarded)
  sx: number; // derived screen x
  sy: number; // derived screen y
  sw: number; // derived width
};

// Tidy 4×3 grid (like the Wikontic intro) so nothing overlaps.
const GRID_COLS = [20, 365, 710, 1055];
const GRID_ROWS = [20, 240, 460];
const GRID_W = 300;

// Relevant fragments sit on different rows AND columns; arrows connect the ones
// that share an entity (объект, система мониторинга, инженерные сети). The lost
// fragment is the bridge between фрагмент 1 and фрагмент 3.
const corpusGrid: Omit<CorpusBox, 'sx' | 'sy' | 'sw'>[] = [
  {text: 'Требования к объекту утверждены проектной командой в 2024 году.', col: 0, row: 0, slot: 2,
    segments: [{t: 'Требования', tone: 'violet'}, {t: ' к '}, {t: 'объекту', tone: 'blue'}, {t: ' утверждены проектной командой в 2024 году.'}]},
  {text: 'Объект включает систему мониторинга.', col: 1, row: 0, slot: 0,
    segments: [{t: 'Объект', tone: 'blue'}, {t: ' включает систему мониторинга.'}]},
  {text: 'Система мониторинга контролирует инженерные сети.', col: 2, row: 1, lost: true, tx: 1, ty: 1},
  {text: 'Инженерные сети связаны с эксплуатационными рисками.', col: 0, row: 2, slot: 1,
    segments: [{t: 'Инженерные сети связаны с '}, {t: 'эксплуатационными', tone: 'green'}, {t: ' рисками.'}]},
  {text: 'Подземный паркинг рассчитан на 120 машино-мест.', col: 2, row: 0, tx: 1, ty: -1},
  {text: 'Фасадные работы завершены в третьем квартале.', col: 3, row: 0, tx: 1, ty: -1},
  {text: 'Договор подряда подписан с генеральным подрядчиком.', col: 1, row: 1, tx: -1, ty: 0},
  {text: 'Высота типового этажа составляет три метра.', col: 3, row: 1, tx: 1, ty: 0},
  {text: 'Озеленение территории запланировано на весну.', col: 1, row: 2, tx: 1, ty: 1},
  {text: 'Гарантийный срок на кровлю — десять лет.', col: 3, row: 2, tx: 1, ty: 1},
  {text: 'Лифтовое оборудование поставлено зарубежным производителем.', col: 0, row: 1, tx: -1, ty: 0},
  {text: 'Система пожарной сигнализации принята в эксплуатацию.', col: 2, row: 2, tx: 1, ty: 1},
];

const corpus: CorpusBox[] = corpusGrid.map((b) => ({
  ...b,
  sx: GRID_COLS[b.col],
  sy: GRID_ROWS[b.row],
  sw: GRID_W,
}));

// Arrows connect entity-sharing fragments, clipped to box edges so the heads
// land in the gaps (фрагмент 4 → 1 → 2(lost) → 3).
const boxCenter = (b: CorpusBox) => ({x: b.sx + b.sw / 2, y: b.sy + BOX_H / 2});
const edgePoint = (b: CorpusBox, dx: number, dy: number) => {
  const c = boxCenter(b);
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const t = Math.min(
    ux !== 0 ? b.sw / 2 / Math.abs(ux) : Infinity,
    uy !== 0 ? BOX_H / 2 / Math.abs(uy) : Infinity,
  );
  return {x: c.x + ux * t, y: c.y + uy * t};
};
// Arrows by corpus index. `strong` marks the relevant chain through the lost
// bridge (slot2 → slot0 → lost → slot1); the rest is the surrounding mesh that
// shows the documents — and especially the lost one — are all connected.
const idxBySlot = (s: number) => corpus.findIndex((b) => b.slot === s);
const idxLost = corpus.findIndex((b) => b.lost);
const arrowSpecs: {from: number; to: number; strong?: boolean}[] = [
  // relevant chain — passes through the lost document
  {from: idxBySlot(2), to: idxBySlot(0), strong: true},
  {from: idxBySlot(0), to: idxLost, strong: true},
  {from: idxLost, to: idxBySlot(1), strong: true},
  // the lost document is connected to several neighbours
  {from: 4, to: idxLost},
  {from: 6, to: idxLost},
  {from: idxLost, to: 7},
  {from: idxLost, to: 11},
  // wider mesh so the corpus reads as an interconnected set
  {from: idxBySlot(0), to: 4},
  {from: 10, to: 6},
  {from: 5, to: 7},
  {from: 8, to: 11},
  {from: idxBySlot(1), to: 8},
];
const arrows = arrowSpecs.map(({from, to, strong}) => {
  const a = corpus[from];
  const b = corpus[to];
  const ca = boxCenter(a);
  const cb = boxCenter(b);
  const p1 = edgePoint(a, cb.x - ca.x, cb.y - ca.y);
  const p2 = edgePoint(b, ca.x - cb.x, ca.y - cb.y);
  return {p1, p2, len: Math.hypot(p2.x - p1.x, p2.y - p1.y), strong: strong === true};
});

const RagPhase: React.FC = () => {
  const frame = useCurrentFrame();
  const headerIn = progress(frame, 6, 30);
  const allIn = progress(frame, 24, 84); // all boxes appear (identical grey)
  const arrowsIn = progress(frame, 92, 150); // links between fragments
  const linkIn = progress(frame, 162, 220); // question concepts highlighted
  const lostIn = progress(frame, 232, 274); // relevant fragment we will miss — red outline
  const select = progress(frame, 288, 332); // relevant ones brighten
  const discard = progress(frame, 338, 382); // irrelevant (incl. lost) + arrows leave
  const settle = progress(frame, 348, 388); // relevant slide into column — fast, hides reflow
  const missedIn = progress(frame, 415, 458); // the relevant fragment RAG missed
  const noteIn = progress(frame, 480, 523); // takeaway, then holds
  return (
    <Phase duration={PHASES.rag}>
      {/* Header: RAG title + the question, mirroring the Wikontic slide. */}
      <div style={{display: 'flex', alignItems: 'center', gap: 36, width: STAGE_W, marginBottom: 26}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 22, flexShrink: 0}}>
          <BigIcon tone="muted">
            <Search size={64} />
          </BigIcon>
          <span style={{fontSize: 60, fontWeight: 840, color: colors.ink}}>RAG</span>
        </div>
        <div
          style={{
            flex: 1,
            padding: '18px 24px',
            borderRadius: 14,
            background: colors.panel,
            border: '1px solid #cbd9ff',
            boxShadow: '0 18px 44px rgba(23, 32, 51, 0.10)',
            opacity: headerIn,
            transform: `translateY(${(1 - headerIn) * 16}px)`,
          }}
        >
          <div
            style={{
              marginBottom: 8,
              color: colors.muted,
              fontSize: 17,
              fontWeight: 780,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            Вопрос
          </div>
          <SemText segments={questionSegments} on={linkIn} size={28} weight={760} />
        </div>
      </div>

      <div style={{position: 'relative', width: STAGE_W, height: STAGE_H}}>
        {corpus.map((box, i) => {
          const itemIn = clamp(allIn * 1.3 - i * 0.05);
          const isSel = box.slot !== undefined;
          const isLost = box.lost === true;
          let left = box.sx;
          let top = box.sy;
          let width = box.sw;
          let opacity = itemIn;
          let scale = 1;
          if (isSel) {
            left = lerp(box.sx, COL_X, settle);
            top = lerp(box.sy, SLOT_Y[box.slot!], settle);
            width = lerp(box.sw, COL_W, settle);
          } else {
            left = box.sx + (box.tx ?? 0) * 600 * discard;
            top = box.sy + (box.ty ?? 0) * 420 * discard;
            opacity = itemIn * (1 - 0.45 * select) * (1 - discard);
            scale = 1 - 0.12 * discard;
          }
          const sel = isSel ? select : 0;
          const lost = isLost ? lostIn : 0;
          const ch = (a: number, b: number, t: number) => Math.round(lerp(a, b, t));
          // All cards start as identical grey; selected brighten, lost goes red-outlined.
          const bg = isLost
            ? `rgb(${ch(238, 255, lost)}, ${ch(241, 225, lost)}, ${ch(246, 225, lost)})`
            : `rgb(${ch(238, 248, sel)}, ${ch(241, 250, sel)}, ${ch(246, 252, sel)})`;
          const borderCol = isSel
            ? `rgb(${ch(221, 47, sel)}, ${ch(227, 109, sel)}, ${ch(236, 246, sel)})`
            : isLost
              ? `rgb(${ch(221, 215, lost)}, ${ch(227, 40, lost)}, ${ch(236, 40, lost)})`
              : 'rgb(221, 227, 236)';
          const textCol = isLost
            ? `rgb(${ch(91, 178, lost)}, ${ch(102, 28, lost)}, ${ch(117, 28, lost)})`
            : `rgb(${ch(91, 23, sel)}, ${ch(102, 32, sel)}, ${ch(117, 51, sel)})`;
          const fontSize = isSel ? lerp(24, 29, settle) : 24;
          const weight = Math.round(lerp(600, 700, sel));
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left,
                top,
                width,
                padding: '18px 24px',
                borderRadius: 12,
                background: bg,
                border: `${1 + 2 * lost}px solid ${borderCol}`,
                boxShadow: `0 10px 26px rgba(23, 32, 51, 0.06), 0 ${18 * sel}px ${44 * sel}px rgba(47, 109, 246, ${0.18 * sel}), 0 0 0 ${2 * sel}px rgba(47, 109, 246, 0.55), 0 0 0 ${5 * lost}px rgba(215, 40, 40, ${0.95 * lost}), 0 12px 34px rgba(215, 40, 40, ${0.3 * lost})`,
                opacity,
                transform: `scale(${scale})`,
                color: textCol,
                fontSize,
                lineHeight: 1.3,
                fontWeight: weight,
                zIndex: isSel || isLost ? 2 : 1,
              }}
            >
              {box.segments ? (
                <SemText segments={box.segments} on={linkIn} size={fontSize} weight={weight} color={textCol} />
              ) : (
                box.text
              )}
            </div>
          );
        })}

        {/* Links between related fragments (drawn BEHIND the docs). */}
        <svg
          width={STAGE_W}
          height={STAGE_H}
          style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none', zIndex: 0}}
        >
          <defs>
            <marker id="ragArrowHead" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill={colors.steel} />
            </marker>
          </defs>
          {arrows.map((ar, i) => {
            const o = clamp(arrowsIn * 1.2 - i * 0.12) * (1 - discard);
            return (
              <line
                key={i}
                x1={ar.p1.x}
                y1={ar.p1.y}
                x2={ar.p2.x}
                y2={ar.p2.y}
                stroke={colors.steel}
                strokeWidth={2.5}
                markerEnd="url(#ragArrowHead)"
                strokeDasharray={ar.len}
                strokeDashoffset={ar.len * (1 - clamp(arrowsIn * 1.2 - i * 0.12))}
                opacity={o}
              />
            );
          })}
        </svg>

        {/* The relevant fragment RAG failed to retrieve. */}
        <div
          style={{
            position: 'absolute',
            left: COL_X,
            top: MISSED_Y,
            width: COL_W,
            padding: '20px 28px',
            border: `2px dashed ${colors.red}`,
            borderRadius: 12,
            background: colors.redSoft,
            opacity: missedIn,
            transform: `translateY(${(1 - missedIn) * 18}px)`,
            zIndex: 4,
          }}
        >
          <div style={{display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, color: '#a82626', fontSize: 26}}>
            <Split size={24} />
            <strong style={{fontWeight: 900}}>Релевантный фрагмент не найден</strong>
          </div>
          <p style={{margin: 0, color: '#a82626', fontSize: 29, lineHeight: 1.3, fontWeight: 700}}>
            {missedChunk.text}
          </p>
        </div>

        {/* Takeaway. */}
        <div
          style={{
            position: 'absolute',
            left: COL_X,
            top: NOTE_Y,
            width: COL_W,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '18px 24px',
            borderRadius: 12,
            background: colors.amberSoft,
            color: '#8a5c00',
            fontSize: 27,
            fontWeight: 780,
            opacity: noteIn,
            zIndex: 4,
          }}
        >
          <Split size={28} />
          <span>RAG не может найти все релевантные фрагменты.</span>
        </div>
      </div>
    </Phase>
  );
};

// Intro beat: the SAME corpus RAG saw, laid out as a tidy 4×3 scatter across the
// graph area. Wikontic then unites them all (converge to centre + text fades) and
// the graph grows out of the merged mass. Reuses `corpus` text so it visibly reads
// as "same documents, handled differently".
const DOC_COLS = [70, 490, 910, 1330];
const DOC_ROWS = [40, 235, 430];
const DOC_W = 320;
const docScatter = corpus.map((box, i) => ({
  text: box.text,
  x: DOC_COLS[i % 4],
  y: DOC_ROWS[Math.floor(i / 4)],
}));

// Clean, evenly-spaced layout tuned for this wide canvas (1180x470) so bubbles
// never overlap. Overrides only x/y; node identity/labels stay from the data.
const wikonticLayout: Record<string, {x: number; y: number}> = {
  requirements: {x: 0.08, y: 0.14},
  year: {x: 0.3, y: 0.14},
  project: {x: 0.05, y: 0.5},
  complex: {x: 0.265, y: 0.5},
  buildings: {x: 0.5, y: 0.14},
  monitoring: {x: 0.5, y: 0.5},
  parking: {x: 0.5, y: 0.86},
  networks: {x: 0.735, y: 0.5},
  risks: {x: 0.95, y: 0.5},
};
const wikonticGraphNodes = ragVsGraphNodes.map((node) => ({
  ...node,
  ...(wikonticLayout[node.id] ?? {}),
}));

const GRAPH_W = 1740;
const GRAPH_H = 560;

const GRAPH_CX = GRAPH_W / 2;
const GRAPH_CY = GRAPH_H / 2;

const WikonticPhase: React.FC = () => {
  const frame = useCurrentFrame();
  const headerIn = progress(frame, 6, 40);
  const docsIn = progress(frame, 24, 96); // same corpus appears, scattered & grey
  const uniteIn = progress(frame, 130, 200); // all docs converge to centre, text fades
  const graphIn = progress(frame, 200, 264); // graph grows out of the merged mass
  const pathIn = progress(frame, 300, 420);
  const internalIn = progress(frame, 320, 410);
  const answerIn = progress(frame, 510, 558); // answer box slides up from the bottom
  const pathOut = progress(frame, 510, 552); // path box slides out downward
  const visiblePathCount = Math.ceil(answerPathEdgeIds.length * pathIn);
  // The merged cluster lingers under the first frames of the graph, then dissolves.
  const docsGone = progress(frame, 200, 252);
  return (
    <Phase duration={PHASES.wikontic}>
      {/* Header: Wikontic title + the question, mirroring the RAG slide. */}
      <div style={{display: 'flex', alignItems: 'center', gap: 36, width: GRAPH_W, marginBottom: 26}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 22, flexShrink: 0}}>
          <BigIcon tone="wikontic">
            <Waypoints size={64} />
          </BigIcon>
          <span style={{fontSize: 60, fontWeight: 840, color: colors.blueDark}}>Wikontic</span>
        </div>
        <div
          style={{
            flex: 1,
            padding: '18px 24px',
            borderRadius: 14,
            background: colors.panel,
            border: '1px solid #cbd9ff',
            boxShadow: '0 18px 44px rgba(23, 32, 51, 0.10)',
            opacity: headerIn,
            transform: `translateY(${(1 - headerIn) * 16}px)`,
          }}
        >
          <div
            style={{
              marginBottom: 8,
              color: colors.muted,
              fontSize: 17,
              fontWeight: 780,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            Вопрос
          </div>
          <SemText segments={questionSegments} on={1} size={28} weight={760} />
        </div>
      </div>

      <div style={{position: 'relative', width: GRAPH_W, height: GRAPH_H}}>
        {/* Intro: the same corpus RAG saw, scattered — then united into the centre. */}
        {docsGone < 1 && (
          <div style={{position: 'absolute', inset: 0, opacity: 1 - docsGone, zIndex: 2}}>
            {docScatter.map((doc, i) => {
              const itemIn = clamp(docsIn * 1.3 - i * 0.05);
              const left = lerp(doc.x, GRAPH_CX - (DOC_W * 0.34) / 2, uniteIn);
              const top = lerp(doc.y, GRAPH_CY - 36, uniteIn);
              const scale = lerp(1, 0.34, uniteIn);
              const textOpacity = clamp(1 - uniteIn * 1.7);
              // Borders blend toward a single united block as they pile up.
              const united = uniteIn;
              return (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    left,
                    top,
                    width: DOC_W,
                    padding: '16px 20px',
                    borderRadius: 12,
                    background: `rgb(${Math.round(lerp(238, 224, united))}, ${Math.round(
                      lerp(241, 234, united),
                    )}, ${Math.round(lerp(246, 255, united))})`,
                    border: `1px solid rgba(${Math.round(lerp(221, 47, united))}, ${Math.round(
                      lerp(227, 109, united),
                    )}, ${Math.round(lerp(236, 246, united))}, ${lerp(1, 0.5, united)})`,
                    boxShadow: `0 10px 26px rgba(23, 32, 51, ${lerp(0.06, 0.14, united)})`,
                    opacity: itemIn,
                    transform: `scale(${scale})`,
                    transformOrigin: 'top left',
                    color: colors.steel,
                    fontSize: 22,
                    lineHeight: 1.3,
                    fontWeight: 600,
                  }}
                >
                  <span style={{opacity: textOpacity}}>{doc.text}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Caption that narrates the unite beat, fading out as the graph forms. */}
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: GRAPH_CY + 70,
            textAlign: 'center',
            zIndex: 3,
            opacity: clamp(uniteIn * 1.6) * (1 - progress(frame, 200, 224)),
            color: colors.blueDark,
            fontSize: 28,
            fontWeight: 820,
            letterSpacing: '0.01em',
          }}
        >
          Объединяем документы в граф
        </div>

        <GraphView
          nodes={wikonticGraphNodes}
          edges={ragVsGraphEdges}
          reveal={graphIn}
          softReveal
          showTypes
          typeOutside
          nodeRadius={78}
          nodeAspect={1.3}
          labelFontSize={29}
          typeFontSize={21}
          edgeFontSize={25}
          highlightedNodeIds={answerPathNodeIds.slice(0, visiblePathCount + 1)}
          highlightedEdgeIds={answerPathEdgeIds.slice(0, visiblePathCount)}
          width={GRAPH_W}
          height={GRAPH_H}
          muted={pathIn > 0.05}
        />
      </div>

      {/* Bottom slot: the path box, then the answer box slides up to replace it. */}
      <div style={{position: 'relative', width: GRAPH_W, height: 188, marginTop: 36}}>
        {/* Path-to-answer box */}
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 0,
            padding: '26px 36px',
            border: '1px solid #cbd9ff',
            borderRadius: 16,
            background: colors.panel,
            boxShadow: '0 18px 50px rgba(23, 32, 51, 0.1)',
            opacity: internalIn * (1 - pathOut),
            transform: `translateY(${pathOut * 230}px)`,
          }}
        >
          <div
            style={{
              marginBottom: 16,
              color: colors.blue,
              fontSize: 26,
              fontWeight: 800,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            В графе находим путь к ответу
          </div>
          <PathHighlight path={internalPathText} progress={pathIn} fontSize={30} />
        </div>

        {/* Answer box — slides up from the bottom to replace the path box */}
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 0,
            padding: '24px 36px',
            border: '1px solid rgba(25, 169, 116, 0.4)',
            borderRadius: 16,
            background: colors.panel,
            boxShadow: '0 22px 60px rgba(23, 32, 51, 0.14)',
            opacity: answerIn,
            transform: `translateY(${(1 - answerIn) * 230}px)`,
          }}
        >
          <div
            style={{
              marginBottom: 14,
              color: '#08724d',
              fontSize: 26,
              fontWeight: 800,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Ответ
          </div>
          <p style={{margin: 0, fontSize: 38, lineHeight: 1.3, fontWeight: 800, color: colors.ink}}>
            {answerSegments.map((seg, index) =>
              seg.tone ? (
                <span
                  key={index}
                  style={{
                    borderRadius: 8,
                    padding: '2px 10px',
                    background: seg.tone === 'blue' ? colors.blueSoft : colors.greenSoft,
                    color: seg.tone === 'blue' ? colors.blueDark : '#08724d',
                    boxDecorationBreak: 'clone',
                    WebkitBoxDecorationBreak: 'clone',
                  }}
                >
                  {seg.text}
                </span>
              ) : (
                <span key={index}>{seg.text}</span>
              ),
            )}
          </p>
        </div>
      </div>
    </Phase>
  );
};

// Answer with the two key phrases highlighted, per request.
const answerSegments: {text: string; tone?: 'blue' | 'green'}[] = [
  {text: 'Нужно проверить '},
  {text: 'систему мониторинга', tone: 'blue'},
  {text: ' инженерных сетей, потому что она связана с '},
  {text: 'эксплуатационным контролем', tone: 'green'},
  {text: ' объекта.'},
];

export const Animation2_RagVsGraph: React.FC = () => {
  return (
    <AbsoluteFill style={{background: '#ffffff', fontFamily: font.family}}>
      <Sequence from={RAG_FROM} durationInFrames={PHASES.rag}>
        <RagPhase />
      </Sequence>
      <Sequence from={WIKONTIC_FROM} durationInFrames={PHASES.wikontic}>
        <WikonticPhase />
      </Sequence>
    </AbsoluteFill>
  );
};
