import {compactGraphEdges, compactGraphNodes} from './animation1';

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
