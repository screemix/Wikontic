import type {GraphEdge, GraphNode} from './graphBefore';

export type DocumentPart = {
  text: string;
  factId?: string;
};

export type DocumentLine = DocumentPart[];

export const methodDocument: DocumentLine[] = [
  [
    {text: 'В 2024 году проектная команда '},
    {text: 'утвердила требования к жилому комплексу', factId: 'requirements-date'},
    {text: '.'},
  ],
  [
    {text: 'Объект включает '},
    {text: 'три корпуса', factId: 'buildings'},
    {text: ', '},
    {text: 'подземный паркинг', factId: 'parking'},
    {text: ' и '},
    {text: 'систему мониторинга инженерных сетей', factId: 'monitoring-networks'},
    {text: '.'},
  ],
  [
    {text: 'Система мониторинга используется для '},
    {text: 'контроля эксплуатационных рисков', factId: 'risk-control'},
    {text: '.'},
  ],
];

export const highlightedFacts = [
  {id: 'requirements-date', text: 'требования утверждены в 2024 году', tone: 'amber'},
  {id: 'buildings', text: 'объект включает три корпуса', tone: 'blue'},
  {id: 'parking', text: 'объект включает подземный паркинг', tone: 'green'},
  {id: 'monitoring-networks', text: 'система мониторинга инженерных сетей', tone: 'violet'},
  {id: 'risk-control', text: 'мониторинг контролирует эксплуатационные риски', tone: 'blue'},
] as const;

export const methodTriplets = [
  {subject: 'жилой комплекс', relation: 'включает', object: 'три корпуса'},
  {subject: 'жилой комплекс', relation: 'включает', object: 'подземный паркинг'},
  {subject: 'жилой комплекс', relation: 'включает', object: 'система мониторинга'},
  {subject: 'система мониторинга', relation: 'контролирует', object: 'инженерные сети'},
  {subject: 'утверждение требований', relation: 'дата', object: '2024', qualifier: 'контекст: проект'},
];

export const ontologyChecks = [
  {source: 'ЖК', result: 'строительный объект', tone: 'blue'},
  {source: 'инженерные сети', result: 'инженерная система', tone: 'green'},
  {source: 'содержит / имеет в составе', result: 'включает', tone: 'amber'},
  {source: 'контроль рисков', result: 'допустимая связь', tone: 'green'},
] as const;

export const dedupGroups = [
  {aliases: ['ЖК', 'жилой комплекс', 'объект'], canonical: 'жилой комплекс'},
  {aliases: ['сети', 'инженерные сети'], canonical: 'инженерные сети'},
];

export const compactGraphNodes: GraphNode[] = [
  {id: 'project', label: 'проект', type: 'проект', x: 0.12, y: 0.48, kind: 'document'},
  {id: 'complex', label: 'жилой комплекс', type: 'строительный объект', x: 0.34, y: 0.48, kind: 'project'},
  {id: 'buildings', label: 'три корпуса', type: 'здание', x: 0.58, y: 0.24, kind: 'asset'},
  {id: 'parking', label: 'подземный паркинг', type: 'инфраструктура', x: 0.62, y: 0.48, kind: 'asset'},
  {id: 'monitoring', label: 'система мониторинга', type: 'инженерная система', x: 0.58, y: 0.72, kind: 'system'},
  {id: 'networks', label: 'инженерные сети', type: 'инженерная система', x: 0.84, y: 0.72, kind: 'system'},
  {id: 'requirements', label: 'требования', type: 'требование', x: 0.34, y: 0.2, kind: 'requirement'},
  {id: 'year', label: '2024', type: 'дата', x: 0.12, y: 0.2, kind: 'time'},
  {id: 'risks', label: 'эксплуатационные риски', type: 'риск', x: 0.86, y: 0.45, kind: 'requirement'},
];

export const compactGraphEdges: GraphEdge[] = [
  {id: 'project-complex', source: 'project', target: 'complex', label: 'содержит', status: 'valid'},
  {id: 'complex-buildings', source: 'complex', target: 'buildings', label: 'включает', status: 'valid'},
  {id: 'complex-parking', source: 'complex', target: 'parking', label: 'включает', status: 'valid'},
  {id: 'complex-monitoring', source: 'complex', target: 'monitoring', label: 'включает', status: 'valid'},
  {id: 'monitoring-networks', source: 'monitoring', target: 'networks', label: 'контролирует', status: 'valid'},
  {id: 'networks-risks', source: 'networks', target: 'risks', label: 'связаны с', status: 'valid'},
  {id: 'requirements-year', source: 'requirements', target: 'year', label: 'дата', status: 'valid'},
  {id: 'requirements-complex', source: 'requirements', target: 'complex', label: 'для объекта', status: 'valid'},
];

