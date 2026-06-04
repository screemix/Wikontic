import {compactGraphEdges, compactGraphNodes} from './animation1';

export const ragQuestion = 'Что важно проверить перед вводом жилого комплекса в эксплуатацию?';

export const ragChunks = [
  {
    title: 'Фрагмент 1',
    text: 'Объект включает систему мониторинга.',
  },
  {
    title: 'Фрагмент 2',
    text: 'Система мониторинга контролирует инженерные сети.',
  },
  {
    title: 'Фрагмент 3',
    text: 'Инженерные сети связаны с эксплуатационными рисками.',
  },
  {
    title: 'Фрагмент 4',
    text: 'Требования к объекту утверждены проектной командой в 2024 году.',
  },
];

export const ragVsGraphNodes = compactGraphNodes;
export const ragVsGraphEdges = compactGraphEdges;

export const answerPathNodeIds = ['complex', 'monitoring', 'networks', 'risks'];
export const answerPathEdgeIds = ['complex-monitoring', 'monitoring-networks', 'networks-risks'];

export const internalPathText =
  'жилой комплекс -> включает -> система мониторинга -> контролирует -> инженерные сети -> связаны с -> эксплуатационные риски';

export const ragVsWikonticAnswer =
  'Нужно проверить систему мониторинга инженерных сетей, потому что она связана с эксплуатационным контролем объекта.';

