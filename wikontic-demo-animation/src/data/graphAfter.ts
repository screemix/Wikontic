import type {GraphEdge, GraphNode} from './graphBefore';

export const refinedNodes: GraphNode[] = [
  {
    id: 'complex_clean',
    label: 'жилой комплекс',
    type: 'строительный объект',
    x: 0.22,
    y: 0.46,
    kind: 'project',
  },
  {
    id: 'buildings',
    label: 'три корпуса',
    type: 'здание',
    x: 0.52,
    y: 0.24,
    kind: 'asset',
  },
  {
    id: 'parking',
    label: 'подземный паркинг',
    type: 'инфраструктура',
    x: 0.57,
    y: 0.45,
    kind: 'asset',
  },
  {
    id: 'monitoring',
    label: 'система мониторинга',
    type: 'инженерная система',
    x: 0.55,
    y: 0.68,
    kind: 'system',
  },
  {
    id: 'networks_clean',
    label: 'инженерные сети',
    type: 'инженерная система',
    x: 0.8,
    y: 0.68,
    kind: 'system',
  },
  {
    id: 'requirements',
    label: 'требования',
    type: 'требование',
    x: 0.3,
    y: 0.2,
    kind: 'requirement',
  },
  {
    id: 'year',
    label: '2024',
    type: 'дата',
    x: 0.12,
    y: 0.2,
    kind: 'time',
  },
];

export const refinedEdges: GraphEdge[] = [
  {id: 'e1_clean', source: 'complex_clean', target: 'buildings', label: 'включает', status: 'valid'},
  {id: 'e2_clean', source: 'complex_clean', target: 'parking', label: 'включает', status: 'valid'},
  {id: 'e3_clean', source: 'complex_clean', target: 'monitoring', label: 'включает', status: 'path'},
  {id: 'e4_clean', source: 'monitoring', target: 'networks_clean', label: 'относится к', status: 'path'},
  {id: 'e5_clean', source: 'requirements', target: 'year', label: 'дата', status: 'valid'},
  {id: 'e6_clean', source: 'requirements', target: 'complex_clean', label: 'объект требований', status: 'valid'},
];

export const aliasMerges = [
  {from: ['ЖК', 'объект'], to: 'жилой комплекс'},
  {from: ['сети'], to: 'инженерные сети'},
];

export const qaPathEdgeIds = ['e3_clean', 'e4_clean'];
export const qaPathNodeIds = ['complex_clean', 'monitoring', 'networks_clean'];
