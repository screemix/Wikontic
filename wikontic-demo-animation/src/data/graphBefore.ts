export type GraphNode = {
  id: string;
  label: string;
  type?: string;
  x: number;
  y: number;
  kind: 'project' | 'asset' | 'system' | 'requirement' | 'time' | 'document';
  aliasOf?: string;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  status?: 'valid' | 'warning' | 'invalid' | 'path';
};

export const dirtyNodes: GraphNode[] = [
  {id: 'complex', label: 'жилой комплекс', x: 0.24, y: 0.38, kind: 'project', aliasOf: 'complex_clean'},
  {id: 'jk', label: 'ЖК', x: 0.17, y: 0.58, kind: 'project', aliasOf: 'complex_clean'},
  {id: 'object', label: 'объект', x: 0.31, y: 0.62, kind: 'project', aliasOf: 'complex_clean'},
  {id: 'buildings', label: 'три корпуса', x: 0.52, y: 0.23, kind: 'asset'},
  {id: 'parking', label: 'подземный паркинг', x: 0.61, y: 0.42, kind: 'asset'},
  {id: 'monitoring', label: 'система мониторинга', x: 0.55, y: 0.66, kind: 'system'},
  {id: 'networks', label: 'инженерные сети', x: 0.78, y: 0.58, kind: 'system', aliasOf: 'networks_clean'},
  {id: 'nets', label: 'сети', x: 0.82, y: 0.77, kind: 'system', aliasOf: 'networks_clean'},
  {id: 'requirements', label: 'требования', x: 0.28, y: 0.18, kind: 'requirement'},
  {id: 'year', label: '2024', x: 0.13, y: 0.21, kind: 'time'},
];

export const dirtyEdges: GraphEdge[] = [
  {id: 'e1', source: 'complex', target: 'buildings', label: 'включает', status: 'valid'},
  {id: 'e2', source: 'jk', target: 'parking', label: 'содержит', status: 'warning'},
  {id: 'e3', source: 'object', target: 'monitoring', label: 'имеет в составе', status: 'warning'},
  {id: 'e4', source: 'monitoring', target: 'nets', label: 'про сети', status: 'invalid'},
  {id: 'e5', source: 'requirements', target: 'year', label: 'дата', status: 'valid'},
  {id: 'e6', source: 'requirements', target: 'object', label: 'для', status: 'warning'},
];
