import type {ComponentType} from 'react';
import {Animation1_TextToGraph, TEXT_TO_GRAPH_FRAMES} from '../scenes/Animation1_TextToGraph';
import {Animation2_RagVsGraph, RAG_VS_WIKONTIC_FRAMES} from '../scenes/Animation2_RagVsGraph';
import {Animation3_SyntheticData, SYNTHETIC_DATA_FRAMES} from '../scenes/Animation3_SyntheticData';
import {FPS, VIDEO_HEIGHT, VIDEO_WIDTH} from '../theme';

export type Chapter = {
  label: string;
  frame: number;
};

export type AnimationSlide = {
  kind: 'animation';
  id: string;
  title: string;
  subtitle: string;
  component: ComponentType<Record<string, never>>;
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

export const presentationVideo = {
  width: VIDEO_WIDTH,
  height: VIDEO_HEIGHT,
  fps: FPS,
};

export const presentationSlides: PresentationSlide[] = [
  {
    kind: 'animation',
    id: 'text-to-graph',
    title: 'Wikontic',
    subtitle: 'Текст превращается в проверяемую базу фактов',
    component: Animation1_TextToGraph,
    durationInFrames: TEXT_TO_GRAPH_FRAMES,
    chapters: [
      {label: 'Документ', frame: 0},
      {label: 'Факты', frame: 130},
      {label: 'Триплеты', frame: 245},
      {label: 'Онтология', frame: 390},
      {label: 'Дедупликация', frame: 540},
      {label: 'Финальный граф', frame: 700},
    ],
  },
  {
    kind: 'animation',
    id: 'rag-vs-wikontic',
    title: 'RAG vs Wikontic',
    subtitle: 'Wikonitc лучше подходит для сложных вопросов, где надо связать несколько фактов',
    component: Animation2_RagVsGraph,
    durationInFrames: RAG_VS_WIKONTIC_FRAMES,
    chapters: [
      {label: 'Вопрос', frame: 0},
      {label: 'RAG-фрагменты', frame: 180},
      {label: 'Путь в графе', frame: 330},
      {label: 'Ответ', frame: 520},
    ],
  },
  {
    kind: 'animation',
    id: 'synthetic-data',
    title: 'Генерация синтетических данных',
    subtitle: 'Граф как генератор качественных синтетических данных для обучения моделей',
    component: Animation3_SyntheticData,
    durationInFrames: SYNTHETIC_DATA_FRAMES,
    chapters: [
      {label: 'Граф', frame: 0},
      {label: 'Ответ', frame: 140},
      {label: 'Пути', frame: 330},
      {label: 'QA-карточки', frame: 530},
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
];
