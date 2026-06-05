import React from 'react';
import {colors} from '../theme';
import type {GraphEdge, GraphNode} from '../data/graphBefore';
import {refinedNodes} from '../data/graphAfter';

type GraphViewProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  reveal?: number;
  showTypes?: boolean;
  highlightedNodeIds?: string[];
  highlightedEdgeIds?: string[];
  scannerProgress?: number;
  mergeProgress?: number;
  width?: number;
  height?: number;
  muted?: boolean;
  nodeRadius?: number;
  nodeAspect?: number;
  typeOutside?: boolean;
  labelFontSize?: number;
  typeFontSize?: number;
  edgeFontSize?: number;
  softReveal?: boolean;
  onNodeSelect?: (node: GraphNode) => void;
};

const smoothstep = (t: number) => {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
};

const kindColor: Record<GraphNode['kind'], {fill: string; stroke: string}> = {
  project: {fill: colors.blueSoft, stroke: colors.blue},
  asset: {fill: colors.greenSoft, stroke: colors.green},
  system: {fill: colors.violetSoft, stroke: colors.violet},
  requirement: {fill: colors.amberSoft, stroke: colors.amber},
  time: {fill: '#f4f6f9', stroke: colors.steel},
  document: {fill: '#eef4ff', stroke: colors.blueDark},
};

const edgeColor = (edge: GraphEdge, highlighted: boolean) => {
  if (highlighted || edge.status === 'path') {
    return colors.blue;
  }
  if (edge.status === 'invalid') {
    return colors.red;
  }
  if (edge.status === 'warning') {
    return colors.amber;
  }
  return colors.steel;
};

const clamp = (value: number) => Math.max(0, Math.min(1, value));

const nodePosition = (node: GraphNode, width: number, height: number, mergeProgress: number) => {
  if (node.aliasOf && mergeProgress > 0) {
    const target = refinedNodes.find((candidate) => candidate.id === node.aliasOf);
    if (target) {
      const p = clamp(mergeProgress);
      return {
        x: (node.x * (1 - p) + target.x * p) * width,
        y: (node.y * (1 - p) + target.y * p) * height,
      };
    }
  }
  return {x: node.x * width, y: node.y * height};
};

const labelLines = (label: string) => {
  const words = label.split(' ');
  if (label.length < 14 || words.length === 1) {
    return [label];
  }
  const midpoint = Math.ceil(words.length / 2);
  return [words.slice(0, midpoint).join(' '), words.slice(midpoint).join(' ')];
};

export const GraphView: React.FC<GraphViewProps> = ({
  nodes,
  edges,
  reveal = 1,
  showTypes = false,
  highlightedNodeIds = [],
  highlightedEdgeIds = [],
  scannerProgress,
  mergeProgress = 0,
  width = 980,
  height = 620,
  muted = false,
  nodeRadius,
  nodeAspect = 1,
  typeOutside = false,
  labelFontSize,
  typeFontSize,
  edgeFontSize,
  softReveal = false,
  onNodeSelect,
}) => {
  const edgeFs = edgeFontSize ?? 17;
  const edgeHalf = edgeFs * 0.3;
  const edgeRectH = edgeFs + 15;
  const visibleNodeCount = Math.ceil(nodes.length * clamp(reveal));
  const visibleNodes = nodes.slice(0, visibleNodeCount);
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));

  // Soft, staggered reveal: each node gently fades + scales in over an
  // overlapping window so the graph "grows" smoothly instead of popping.
  const nodeIndex = new Map(nodes.map((node, i) => [node.id, i]));
  // Smaller window => more spacing between consecutive node appearances
  // (less simultaneous, less jittery). Per-node fade speed is kept constant by
  // lengthening the overall reveal window in the caller.
  const appearWindow = 0.2;
  const appear = (id: string): number => {
    if (!softReveal) {
      return 1;
    }
    const i = nodeIndex.get(id) ?? 0;
    const start = nodes.length > 1 ? (i / (nodes.length - 1)) * (1 - appearWindow) : 0;
    return smoothstep((clamp(reveal) - start) / appearWindow);
  };

  const renderNodes = softReveal ? nodes : visibleNodes;

  return (
    <svg className="graphSvg" viewBox={`0 0 ${width} ${height}`} role="img">
      <defs>
        <filter id="softGlow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="7" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill={colors.steel} />
        </marker>
        <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill={colors.blue} />
        </marker>
      </defs>
      <rect x="0" y="0" width={width} height={height} rx="26" fill={muted ? '#fbfcfe' : '#ffffff'} />
      {scannerProgress !== undefined ? (
        <rect
          x={scannerProgress * width - 22}
          y="18"
          width="44"
          height={height - 36}
          rx="22"
          fill={colors.blue}
          opacity="0.1"
        />
      ) : null}
      {edges.map((edge) => {
        if (!softReveal && (!visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target))) {
          return null;
        }
        const source = nodes.find((node) => node.id === edge.source);
        const target = nodes.find((node) => node.id === edge.target);
        if (!source || !target) {
          return null;
        }
        const edgeAppear = softReveal ? Math.min(appear(edge.source), appear(edge.target)) : 1;
        if (edgeAppear <= 0.001) {
          return null;
        }
        const start = nodePosition(source, width, height, mergeProgress);
        const end = nodePosition(target, width, height, mergeProgress);
        const highlighted = highlightedEdgeIds.includes(edge.id);
        const stroke = edgeColor(edge, highlighted);
        const midX = (start.x + end.x) / 2;
        const midY = (start.y + end.y) / 2;
        return (
          <g key={edge.id} opacity={(muted && !highlighted ? 0.36 : 1) * edgeAppear}>
            <line
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke={stroke}
              strokeWidth={highlighted ? 7 : 3.5}
              strokeDasharray={edge.status === 'warning' || edge.status === 'invalid' ? '10 9' : undefined}
              markerEnd={highlighted ? 'url(#arrowBlue)' : 'url(#arrow)'}
              filter={highlighted ? 'url(#softGlow)' : undefined}
            />
            <rect
              x={midX - edge.label.length * edgeHalf - 12}
              y={midY - edgeRectH / 2}
              width={edge.label.length * edgeHalf * 2 + 24}
              height={edgeRectH}
              rx={edgeRectH / 2}
              fill="#ffffff"
              stroke={stroke}
              opacity="0.96"
            />
            <text
              x={midX}
              y={midY + edgeFs * 0.34}
              textAnchor="middle"
              className="edgeLabel"
              style={{fontSize: edgeFs}}
              fill={stroke}
            >
              {edge.label}
            </text>
          </g>
        );
      })}
      {renderNodes.map((node) => {
        const pos = nodePosition(node, width, height, mergeProgress);
        const highlighted = highlightedNodeIds.includes(node.id) || Boolean(node.aliasOf && mergeProgress > 0.8);
        const color = kindColor[node.kind];
        const radius = nodeRadius ?? (node.label.length > 14 ? 68 : 54);
        const labelFs = labelFontSize ?? 22;
        const typeFs = typeFontSize ?? 15;
        const labelLineHeight = labelFs * 1.05;
        const aliasFade = node.aliasOf && mergeProgress > 0.75 ? 1 - mergeProgress : 1;
        const nodeAppear = appear(node.id);
        if (softReveal && nodeAppear <= 0.001) {
          return null;
        }
        const scale = softReveal ? 0.85 + 0.15 * nodeAppear : 1;
        const lines = labelLines(node.label);
        return (
          <g
            key={node.id}
            opacity={softReveal ? nodeAppear * aliasFade : Math.max(0.08, aliasFade)}
            filter={highlighted ? 'url(#softGlow)' : undefined}
            transform={softReveal ? `translate(${pos.x} ${pos.y}) scale(${scale}) translate(${-pos.x} ${-pos.y})` : undefined}
            onClick={onNodeSelect ? () => onNodeSelect(node) : undefined}
            style={{cursor: onNodeSelect ? 'pointer' : 'default'}}
          >
            <ellipse
              cx={pos.x}
              cy={pos.y}
              rx={radius * nodeAspect}
              ry={radius}
              fill={highlighted ? '#ffffff' : color.fill}
              stroke={highlighted ? colors.blue : color.stroke}
              strokeWidth={highlighted ? 6 : 3}
            />
            <text
              x={pos.x}
              y={
                pos.y -
                (showTypes && node.type && !typeOutside
                  ? labelLineHeight * 0.7
                  : lines.length > 1
                    ? labelLineHeight / 2
                    : -2)
              }
              textAnchor="middle"
              className="nodeLabel"
              style={{fontSize: labelFs}}
            >
              {lines.map((line, index) => (
                <tspan key={line} x={pos.x} dy={index === 0 ? 0 : labelLineHeight}>
                  {line}
                </tspan>
              ))}
            </text>
            {showTypes && node.type ? (
              <text
                x={pos.x}
                y={
                  typeOutside
                    ? pos.y + radius + typeFs + 8
                    : pos.y + (lines.length > 1 ? labelLineHeight + 16 : 24)
                }
                textAnchor="middle"
                className="nodeType"
                style={{fontSize: typeFs}}
              >
                {node.type}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
};
