import {compactGraphEdges, compactGraphNodes, getAnimation1Content} from './animation1';
import type {Locale} from '../i18n/types';

export const syntheticGraphNodes = compactGraphNodes;
export const syntheticGraphEdges = compactGraphEdges;

export const syntheticAnswerNodeId = 'networks';

export const syntheticPaths = [
  {
    id: 'one-hop',
    difficulty: '1-hop',
    nodeIds: ['monitoring', 'networks'],
    edgeIds: ['monitoring-networks'],
    path: 'система мониторинга -> контролирует -> инженерные сети',
  },
  {
    id: 'two-hop',
    difficulty: '2-hop',
    nodeIds: ['complex', 'monitoring', 'networks'],
    edgeIds: ['complex-monitoring', 'monitoring-networks'],
    path: 'жилой комплекс -> включает -> система мониторинга -> контролирует -> инженерные сети',
  },
  {
    id: 'three-hop',
    difficulty: '3-hop',
    nodeIds: ['project', 'complex', 'monitoring', 'networks'],
    edgeIds: ['project-complex', 'complex-monitoring', 'monitoring-networks'],
    path:
      'проект -> содержит -> жилой комплекс -> включает -> система мониторинга -> контролирует -> инженерные сети',
  },
];

export const naturalQACards = [
  {
    question: 'Что контролирует система мониторинга?',
    answer: 'Инженерные сети.',
    path: syntheticPaths[0].path,
    difficulty: '1-hop',
  },
  {
    question: 'Что нужно контролировать при эксплуатации жилого комплекса?',
    answer: 'Инженерные сети.',
    path: syntheticPaths[1].path,
    difficulty: '2-hop',
  },
  {
    question: 'Какая часть инфраструктуры проекта требует мониторингового контроля?',
    answer: 'Инженерные сети.',
    path: syntheticPaths[2].path,
    difficulty: '3-hop',
  },
];

export const datasetLabels = ['QA pairs', 'reasoning chains', 'difficulty control', 'negative examples'];

export type Animation3Content = {
  syntheticGraphNodes: typeof compactGraphNodes;
  syntheticGraphEdges: typeof compactGraphEdges;
  syntheticAnswerNodeId: string;
  syntheticPaths: typeof syntheticPaths;
  naturalQACards: typeof naturalQACards;
  datasetLabels: string[];
  labels: {
    answerCallout: string;
    graphEyebrow: string;
    graphTitle: string;
    graphSubtitle: string;
    pathsEyebrow: string;
    pathsTitle: string;
    pathsSubtitle: string;
    dataEyebrow: string;
    dataTitle: string;
    dataSubtitle: string;
    syntheticDataset: string;
    smallModel: string;
  };
};

const enGraph = getAnimation1Content('en');

const syntheticPathsEn = [
  {
    id: 'one-hop',
    difficulty: '1-hop',
    nodeIds: ['monitoring', 'networks'],
    edgeIds: ['monitoring-networks'],
    path: 'monitoring system -> controls -> utility networks',
  },
  {
    id: 'two-hop',
    difficulty: '2-hop',
    nodeIds: ['complex', 'monitoring', 'networks'],
    edgeIds: ['complex-monitoring', 'monitoring-networks'],
    path: 'residential complex -> includes -> monitoring system -> controls -> utility networks',
  },
  {
    id: 'three-hop',
    difficulty: '3-hop',
    nodeIds: ['project', 'complex', 'monitoring', 'networks'],
    edgeIds: ['project-complex', 'complex-monitoring', 'monitoring-networks'],
    path: 'project -> contains -> residential complex -> includes -> monitoring system -> controls -> utility networks',
  },
];

const naturalQACardsEn = [
  {
    question: 'What does the monitoring system control?',
    answer: 'Utility networks.',
    path: syntheticPathsEn[0].path,
    difficulty: '1-hop',
  },
  {
    question: 'What should be monitored during operation of the residential complex?',
    answer: 'Utility networks.',
    path: syntheticPathsEn[1].path,
    difficulty: '2-hop',
  },
  {
    question: 'Which part of the project infrastructure requires monitoring control?',
    answer: 'Utility networks.',
    path: syntheticPathsEn[2].path,
    difficulty: '3-hop',
  },
];

export const animation3Content: Record<Locale, Animation3Content> = {
  ru: {
    syntheticGraphNodes,
    syntheticGraphEdges,
    syntheticAnswerNodeId,
    syntheticPaths,
    naturalQACards,
    datasetLabels,
    labels: {
      answerCallout: 'ответ: инженерные сети',
      graphEyebrow: 'Синтетические данные',
      graphTitle: 'Граф становится источником данных для обучения',
      graphSubtitle: 'Выбираем вершину-ответ в графе знаний',
      pathsEyebrow: 'Контроль сложности',
      pathsTitle: 'Выбираем пути разной длины',
      pathsSubtitle: 'Один и тот же ответ можно получить через 1-hop, 2-hop или 3-hop пути',
      dataEyebrow: 'Генерация',
      dataTitle: 'Пути по графу превращаются в QA-примеры',
      dataSubtitle: 'Путь объясняет ответ на вопрос',
      syntheticDataset: 'Проверяемая синтетика',
      smallModel: 'малая доменная модель',
    },
  },
  en: {
    syntheticGraphNodes: enGraph.compactGraphNodes,
    syntheticGraphEdges: enGraph.compactGraphEdges,
    syntheticAnswerNodeId,
    syntheticPaths: syntheticPathsEn,
    naturalQACards: naturalQACardsEn,
    datasetLabels,
    labels: {
      answerCallout: 'answer: utility networks',
      graphEyebrow: 'Synthetic data',
      graphTitle: 'The graph becomes a source of training data',
      graphSubtitle: 'Choose an answer node in the knowledge graph',
      pathsEyebrow: 'Complexity control',
      pathsTitle: 'Sample paths of different length',
      pathsSubtitle: 'The same answer can be reached through 1-hop, 2-hop, or 3-hop paths',
      dataEyebrow: 'Generation',
      dataTitle: 'Graph paths become QA examples',
      dataSubtitle: 'The path explains the answer to the question',
      syntheticDataset: 'Verifiable synthetic data',
      smallModel: 'small domain model',
    },
  },
};

export const getAnimation3Content = (locale: Locale = 'ru') => animation3Content[locale];
