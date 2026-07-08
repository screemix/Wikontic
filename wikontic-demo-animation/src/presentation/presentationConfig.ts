import type {ComponentType} from 'react';
import {Animation1_TextToGraph, TEXT_TO_GRAPH_FRAMES} from '../scenes/Animation1_TextToGraph';
import {Animation2_RagVsGraph, RAG_VS_WIKONTIC_FRAMES} from '../scenes/Animation2_RagVsGraph';
import {Animation3_SyntheticData, SYNTHETIC_DATA_FRAMES} from '../scenes/Animation3_SyntheticData';
import {FPS, VIDEO_HEIGHT, VIDEO_WIDTH} from '../theme';
import type {Locale} from '../i18n/types';

export type Chapter = {
  label: string;
  frame: number;
};

export type AnimationSlide = {
  kind: 'animation';
  id: string;
  title: string;
  subtitle: string;
  component: ComponentType<{locale?: Locale}>;
  durationInFrames: number;
  chapters: Chapter[];
};

export type MetricsSlide = {
  kind: 'metrics';
  id: string;
  title: string;
  subtitle: string;
  chapters: [];
};

export type PresentationSlide = AnimationSlide | MetricsSlide;

type LocalizedAnimationSlide = {
  kind: 'animation';
  id: string;
  title: string;
  subtitle: string;
  chapters: Chapter[];
};

type LocalizedMetricsSlide = MetricsSlide;

type LocalizedPresentationSlide = LocalizedAnimationSlide | LocalizedMetricsSlide;

export const presentationVideo = {
  width: VIDEO_WIDTH,
  height: VIDEO_HEIGHT,
  fps: FPS,
};

const localizedSlides: Record<Locale, LocalizedPresentationSlide[]> = {
  ru: [
    {
      kind: 'animation',
      id: 'text-to-graph',
      title: 'Wikontic',
      subtitle: 'Текст превращается в проверяемую базу фактов',
      chapters: [
        {label: 'Документ', frame: 0},
        {label: 'Триплеты', frame: 245},
        {label: 'Онтология', frame: 425},
        {label: 'Дедупликация', frame: 540},
        {label: 'Финальный граф', frame: 869},
      ],
    },
    {
      kind: 'animation',
      id: 'rag-vs-wikontic',
      title: 'RAG vs Wikontic',
      subtitle: 'Wikontic лучше подходит для сложных вопросов, где надо связать несколько фактов',
      chapters: [
        {label: 'Вопрос', frame: 25},
        {label: 'RAG-фрагменты', frame: 275},
        {label: 'RAG failure', frame: 350},
        {label: 'RAG итог', frame: 600},
        {label: 'Путь в графе', frame: 1050},
        {label: 'Ответ', frame: 1200},
      ],
    },
    {
      kind: 'animation',
      id: 'synthetic-data',
      title: 'Генерация синтетических данных',
      subtitle: 'Граф как генератор качественных синтетических данных для обучения моделей',
      chapters: [
        {label: 'Граф', frame: 50},
        {label: 'Ответ', frame: 140},
        {label: 'Пути', frame: 400},
        {label: 'QA-карточки', frame: 588},
        {label: 'Датасет', frame: 680},
      ],
    },
    {
      kind: 'metrics',
      id: 'metrics',
      title: 'Итог',
      subtitle: 'Ключевые результаты',
      chapters: [],
    },
  ],
  en: [
    {
      kind: 'animation',
      id: 'text-to-graph',
      title: 'Wikontic',
      subtitle: 'Text becomes a verifiable fact base',
      chapters: [
        {label: 'Document', frame: 0},
        {label: 'Triplets', frame: 245},
        {label: 'Ontology', frame: 425},
        {label: 'Deduplication', frame: 540},
        {label: 'Final graph', frame: 869},
      ],
    },
    {
      kind: 'animation',
      id: 'rag-vs-wikontic',
      title: 'RAG vs Wikontic',
      subtitle: 'For complex questions, Wikontic follows explicit relations between facts',
      chapters: [
        {label: 'Question', frame: 25},
        {label: 'RAG chunks', frame: 275},
        {label: 'RAF failure', frame: 350},
        {label: 'RAF summary', frame: 600},
        {label: 'Graph path', frame: 1050},
        {label: 'Answer', frame: 1200},
      ],
    },
    {
      kind: 'animation',
      id: 'synthetic-data',
      title: 'Synthetic data generation',
      subtitle: 'The graph becomes a source of high-quality synthetic data for model training',
      chapters: [
        {label: 'Graph', frame: 50},
        {label: 'Answer', frame: 140},
        {label: 'Paths', frame: 400},
        {label: 'QA cards', frame: 588},
        {label: 'Dataset', frame: 680},
      ],
    },
    {
      kind: 'metrics',
      id: 'metrics',
      title: 'Summary',
      subtitle: 'Key results',
      chapters: [],
    },
  ],
};

const slideComponents = [
  {component: Animation1_TextToGraph, durationInFrames: TEXT_TO_GRAPH_FRAMES},
  {component: Animation2_RagVsGraph, durationInFrames: RAG_VS_WIKONTIC_FRAMES},
  {component: Animation3_SyntheticData, durationInFrames: SYNTHETIC_DATA_FRAMES},
];

export const getPresentationSlides = (locale: Locale = 'ru'): PresentationSlide[] =>
  localizedSlides[locale].map((slide, index) => {
    if (slide.kind === 'metrics') {
      return slide;
    }
    return {
      ...slide,
      ...slideComponents[index],
    };
  });

export const presentationSlides = getPresentationSlides('ru');
