import React from 'react';
import {Brain, Layers3} from 'lucide-react';
import {PathHighlight} from './PathHighlight';

type DatasetCardProps = {
  question: string;
  answer: string;
  path: string;
  difficulty: string;
  progress?: number;
};

export const DatasetCard: React.FC<DatasetCardProps> = ({
  question,
  answer,
  path,
  difficulty,
  progress = 1,
}) => (
  <div
    className="datasetCard"
    style={{
      opacity: progress,
      transform: `translateY(${(1 - progress) * 26}px) scale(${0.97 + progress * 0.03})`,
    }}
  >
    <div className="datasetCardTop">
      <span>{difficulty}</span>
      <Layers3 size={18} />
    </div>
    <div className="datasetQuestion">{question}</div>
    <div className="datasetAnswer">{answer}</div>
    <PathHighlight path={path} compact progress={progress} />
  </div>
);

export const SmallModelIcon: React.FC<{progress?: number}> = ({progress = 1}) => (
  <div
    className="smallModel"
    style={{
      opacity: progress,
      transform: `scale(${0.92 + progress * 0.08})`,
    }}
  >
    <Brain size={52} />
    <strong>малая доменная модель</strong>
  </div>
);

