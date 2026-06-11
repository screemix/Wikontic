import type {GraphEdge, GraphNode} from './graphBefore';
import type {Locale} from '../i18n/types';

export type DocumentPart = {
  text: string;
  factId?: string;
};

export type DocumentLine = DocumentPart[];

export const methodDocument: DocumentLine[] = [
  [
    {text: 'В '},
    {text: '2024', factId: 'requirements-date'},
    {text: ' году проектная команда '},
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
  {id: 'project', label: 'проект', type: 'документ', x: 0.12, y: 0.48, kind: 'document'},
  {id: 'complex', label: 'жилой комплекс', type: 'строительный объект', x: 0.34, y: 0.48, kind: 'project'},
  {id: 'buildings', label: 'три корпуса', type: 'здание', x: 0.58, y: 0.24, kind: 'asset'},
  {id: 'parking', label: 'подземный паркинг', type: 'строительный объект', x: 0.62, y: 0.48, kind: 'asset'},
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

const methodDocumentEn: DocumentLine[] = [
  [
    {text: 'In '},
    {text: '2024', factId: 'requirements-date'},
    {text: ', the project team '},
    {text: 'approved requirements for a residential complex', factId: 'requirements-date'},
    {text: '.'},
  ],
  [
    {text: 'The asset includes '},
    {text: 'three buildings', factId: 'buildings'},
    {text: ', '},
    {text: 'underground parking', factId: 'parking'},
    {text: ', and '},
    {text: 'a utility-network monitoring system', factId: 'monitoring-networks'},
    {text: '.'},
  ],
  [
    {text: 'The monitoring system is used for '},
    {text: 'controlling operational risks', factId: 'risk-control'},
    {text: '.'},
  ],
];

const highlightedFactsEn = [
  {id: 'requirements-date', text: 'requirements approved in 2024', tone: 'amber'},
  {id: 'buildings', text: 'asset includes three buildings', tone: 'blue'},
  {id: 'parking', text: 'asset includes underground parking', tone: 'green'},
  {id: 'monitoring-networks', text: 'utility-network monitoring system', tone: 'violet'},
  {id: 'risk-control', text: 'monitoring controls operational risks', tone: 'blue'},
] as const;

const methodTripletsEn = [
  {subject: 'residential complex', relation: 'includes', object: 'three buildings'},
  {subject: 'residential complex', relation: 'includes', object: 'underground parking'},
  {subject: 'residential complex', relation: 'includes', object: 'monitoring system'},
  {subject: 'monitoring system', relation: 'controls', object: 'utility networks'},
  {subject: 'requirements approval', relation: 'date', object: '2024', qualifier: 'context: project'},
];

const ontologyChecksEn = [
  {source: 'RC', result: 'construction asset', tone: 'blue'},
  {source: 'utility networks', result: 'engineering system', tone: 'green'},
  {source: 'contains / has part', result: 'includes', tone: 'amber'},
  {source: 'risk control', result: 'valid relation', tone: 'green'},
] as const;

const dedupGroupsEn = [
  {aliases: ['RC', 'residential complex', 'asset'], canonical: 'residential complex'},
  {aliases: ['networks', 'utility networks'], canonical: 'utility networks'},
];

const compactGraphNodesEn: GraphNode[] = [
  {id: 'project', label: 'project', type: 'document', x: 0.12, y: 0.48, kind: 'document'},
  {id: 'complex', label: 'residential complex', type: 'construction asset', x: 0.34, y: 0.48, kind: 'project'},
  {id: 'buildings', label: 'three buildings', type: 'building', x: 0.58, y: 0.24, kind: 'asset'},
  {id: 'parking', label: 'underground parking', type: 'construction asset', x: 0.62, y: 0.48, kind: 'asset'},
  {id: 'monitoring', label: 'monitoring system', type: 'engineering system', x: 0.58, y: 0.72, kind: 'system'},
  {id: 'networks', label: 'utility networks', type: 'engineering system', x: 0.84, y: 0.72, kind: 'system'},
  {id: 'requirements', label: 'requirements', type: 'requirement', x: 0.34, y: 0.2, kind: 'requirement'},
  {id: 'year', label: '2024', type: 'date', x: 0.12, y: 0.2, kind: 'time'},
  {id: 'risks', label: 'operational risks', type: 'risk', x: 0.86, y: 0.45, kind: 'requirement'},
];

const compactGraphEdgesEn: GraphEdge[] = [
  {id: 'project-complex', source: 'project', target: 'complex', label: 'contains', status: 'valid'},
  {id: 'complex-buildings', source: 'complex', target: 'buildings', label: 'includes', status: 'valid'},
  {id: 'complex-parking', source: 'complex', target: 'parking', label: 'includes', status: 'valid'},
  {id: 'complex-monitoring', source: 'complex', target: 'monitoring', label: 'includes', status: 'valid'},
  {id: 'monitoring-networks', source: 'monitoring', target: 'networks', label: 'controls', status: 'valid'},
  {id: 'networks-risks', source: 'networks', target: 'risks', label: 'linked to', status: 'valid'},
  {id: 'requirements-year', source: 'requirements', target: 'year', label: 'date', status: 'valid'},
  {id: 'requirements-complex', source: 'requirements', target: 'complex', label: 'for asset', status: 'valid'},
];

export type Animation1Content = {
  methodDocument: DocumentLine[];
  highlightedFacts: readonly {id: string; text: string; tone: 'amber' | 'blue' | 'green' | 'violet'}[];
  methodTriplets: {subject: string; relation: string; object: string; qualifier?: string}[];
  ontologyChecks: readonly {source: string; result: string; tone: 'blue' | 'green' | 'amber' | 'violet'}[];
  dedupGroups: {aliases: string[]; canonical: string}[];
  compactGraphNodes: GraphNode[];
  compactGraphEdges: GraphEdge[];
  titles: {
    document: string;
    triplets: string;
    ontology: string;
    dedup: string;
    final: string;
  };
  labels: {
    eyebrow: string;
    sourceText: string;
    graph: string;
    before: string;
    after: string;
    tokens: string;
    entitiesAnd: string;
    triplets: string;
    documentToolbar: string;
    documentHeading: string;
    ontologyTitle: string;
    dedupTitle: string;
    dedupAliasesPrefix: string;
    dedupFooter: string;
  };
  graphBenefits: string[];
};

export const animation1Content: Record<Locale, Animation1Content> = {
  ru: {
    methodDocument,
    highlightedFacts,
    methodTriplets,
    ontologyChecks,
    dedupGroups,
    compactGraphNodes,
    compactGraphEdges,
    titles: {
      document: 'Документы содержат факты',
      triplets: '1. Из текста извлекаются триплеты-кандидаты',
      ontology: '2. Верификация и согласование графа с онтологией',
      dedup: '3. Очистка и дедупликация графа',
      final: 'Информация сохранена в компактном и проверяемом графе знаний',
    },
    labels: {
      eyebrow: 'Метод',
      sourceText: 'Исходный текст',
      graph: 'Граф знаний',
      before: 'Было',
      after: 'Стало',
      tokens: 'токенов',
      entitiesAnd: 'сущностей и',
      triplets: 'триплетов',
      documentToolbar: 'Документ',
      documentHeading: 'Проектные требования к объекту',
      ontologyTitle: 'Онтологическая проверка',
      dedupTitle: 'Очистка и дедупликация',
      dedupAliasesPrefix: 'aliases:',
      dedupFooter: 'Синонимы объединены в одну сущность',
    },
    graphBenefits: ['Интерпретируемость', 'Верифицируемость', 'Редактируемость', 'Компактность'],
  },
  en: {
    methodDocument: methodDocumentEn,
    highlightedFacts: highlightedFactsEn,
    methodTriplets: methodTripletsEn,
    ontologyChecks: ontologyChecksEn,
    dedupGroups: dedupGroupsEn,
    compactGraphNodes: compactGraphNodesEn,
    compactGraphEdges: compactGraphEdgesEn,
    titles: {
      document: 'Documents contain facts',
      triplets: '1. Text becomes candidate triplets',
      ontology: '2. Ontology verification and graph alignment',
      dedup: '3. Cleaning and deduplication',
      final: 'Information is preserved in a compact, verifiable knowledge graph',
    },
    labels: {
      eyebrow: 'Method',
      sourceText: 'Source text',
      graph: 'Knowledge graph',
      before: 'Before',
      after: 'After',
      tokens: 'tokens',
      entitiesAnd: 'entities and',
      triplets: 'triplets',
      documentToolbar: 'Document',
      documentHeading: 'Project requirements for the asset',
      ontologyTitle: 'Ontology check',
      dedupTitle: 'Cleaning and deduplication',
      dedupAliasesPrefix: 'aliases:',
      dedupFooter: 'Aliases are merged into one entity',
    },
    graphBenefits: ['Interpretability', 'Verification', 'Editable', 'Compact data representation'],
  },
};

export const getAnimation1Content = (locale: Locale = 'ru') => animation1Content[locale];
