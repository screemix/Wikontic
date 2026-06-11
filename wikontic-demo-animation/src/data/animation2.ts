import {compactGraphEdges, compactGraphNodes, getAnimation1Content} from './animation1';
import type {Locale} from '../i18n/types';

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

export type Animation2Tone = 'blue' | 'green' | 'violet';
export type Animation2Segment = {t: string; tone?: Animation2Tone};
export type Animation2AnswerSegment = {text: string; tone?: 'blue' | 'green'};

export type Animation2Content = {
  ragQuestion: string;
  ragChunks: {title: string; text: string}[];
  ragVsGraphNodes: typeof compactGraphNodes;
  ragVsGraphEdges: typeof compactGraphEdges;
  internalPathText: string;
  questionSegments: Animation2Segment[];
  answerSegments: Animation2AnswerSegment[];
  corpusGrid: {
    text: string;
    col: number;
    row: number;
    slot?: number;
    lost?: boolean;
    segments?: Animation2Segment[];
    tx?: number;
    ty?: number;
  }[];
  labels: {
    question: string;
    notFound: string;
    noDirectLink: string;
    relevantNotFound: string;
    ragNote: string;
    uniteDocuments: string;
    pathToAnswer: string;
    answer: string;
  };
};

const contentRu: Animation2Content = {
  ragQuestion,
  ragChunks,
  ragVsGraphNodes,
  ragVsGraphEdges,
  internalPathText,
  questionSegments: [
    {t: 'Что важно '},
    {t: 'проверить', tone: 'violet'},
    {t: ' перед вводом '},
    {t: 'жилого комплекса', tone: 'blue'},
    {t: ' в '},
    {t: 'эксплуатацию', tone: 'green'},
    {t: '?'},
  ],
  answerSegments: [
    {text: 'Нужно проверить '},
    {text: 'систему мониторинга', tone: 'blue'},
    {text: ' инженерных сетей, потому что она связана с '},
    {text: 'эксплуатационным контролем', tone: 'green'},
    {text: ' объекта.'},
  ],
  corpusGrid: [
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
  ],
  labels: {
    question: 'Вопрос',
    notFound: 'Не найден!',
    noDirectLink: 'Нет прямой связи с вопросом.',
    relevantNotFound: 'Релевантный фрагмент не найден',
    ragNote: 'RAG смотрит только на близость к вопросу, а не на взаимосвязь между фактами.',
    uniteDocuments: 'Объединяем документы в граф',
    pathToAnswer: 'В графе находим путь к ответу',
    answer: 'Ответ',
  },
};

const enGraph = getAnimation1Content('en');

const contentEn: Animation2Content = {
  ragQuestion: 'What should be checked before commissioning the residential complex?',
  ragChunks: [
    {title: 'Fragment 1', text: 'The asset includes a monitoring system.'},
    {title: 'Fragment 2', text: 'The monitoring system controls utility networks.'},
    {title: 'Fragment 3', text: 'Utility networks are linked to operational risks.'},
    {title: 'Fragment 4', text: 'Requirements for the asset were approved by the project team in 2024.'},
  ],
  ragVsGraphNodes: enGraph.compactGraphNodes,
  ragVsGraphEdges: enGraph.compactGraphEdges,
  internalPathText:
    'residential complex -> includes -> monitoring system -> controls -> utility networks -> linked to -> operational risks',
  questionSegments: [
    {t: 'What should be '},
    {t: 'checked', tone: 'violet'},
    {t: ' before commissioning the '},
    {t: 'residential complex', tone: 'blue'},
    {t: '?'},
  ],
  answerSegments: [
    {text: 'Check the '},
    {text: 'monitoring system', tone: 'blue'},
    {text: ' for utility networks, because it is tied to '},
    {text: 'operational control', tone: 'green'},
    {text: ' of the asset.'},
  ],
  corpusGrid: [
    {text: 'Requirements for the asset were approved by the project team in 2024.', col: 0, row: 0, slot: 2,
      segments: [{t: 'Requirements', tone: 'violet'}, {t: ' for the '}, {t: 'asset', tone: 'blue'}, {t: ' were approved by the project team in 2024.'}]},
    {text: 'The asset includes a monitoring system.', col: 1, row: 0, slot: 0,
      segments: [{t: 'The asset', tone: 'blue'}, {t: ' includes a monitoring system.'}]},
    {text: 'The monitoring system controls utility networks.', col: 2, row: 1, lost: true, tx: 1, ty: 1},
    {text: 'Utility networks are linked to operational risks.', col: 0, row: 2, slot: 1,
      segments: [{t: 'Utility networks are linked to '}, {t: 'operational', tone: 'green'}, {t: ' risks.'}]},
    {text: 'Underground parking is designed for 120 spaces.', col: 2, row: 0, tx: 1, ty: -1},
    {text: 'Facade work was completed in the third quarter.', col: 3, row: 0, tx: 1, ty: -1},
    {text: 'The construction contract was signed with the general contractor.', col: 1, row: 1, tx: -1, ty: 0},
    {text: 'The standard floor height is three meters.', col: 3, row: 1, tx: 1, ty: 0},
    {text: 'Landscaping is scheduled for spring.', col: 1, row: 2, tx: 1, ty: 1},
    {text: 'The roof warranty period is ten years.', col: 3, row: 2, tx: 1, ty: 1},
    {text: 'Elevator equipment was supplied by an overseas manufacturer.', col: 0, row: 1, tx: -1, ty: 0},
    {text: 'The fire alarm system has been accepted for operation.', col: 2, row: 2, tx: 1, ty: 1},
  ],
  labels: {
    question: 'Question',
    notFound: 'Not found',
    noDirectLink: 'No direct match to the question.',
    relevantNotFound: 'Relevant fragment not retrieved',
    ragNote: 'RAG uses similarity to the question, not the explicit relationships between facts.',
    uniteDocuments: 'Unifying documents into a graph',
    pathToAnswer: 'The graph finds a path to the answer',
    answer: 'Answer',
  },
};

export const animation2Content: Record<Locale, Animation2Content> = {
  ru: contentRu,
  en: contentEn,
};

export const getAnimation2Content = (locale: Locale = 'ru') => animation2Content[locale];
