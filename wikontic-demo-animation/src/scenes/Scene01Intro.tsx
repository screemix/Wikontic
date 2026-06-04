import React from 'react';
import {interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {FileSearch, Network, ShieldCheck} from 'lucide-react';
import {DocumentPanel} from '../components/DocumentPanel';
import {Panel, SceneLayout} from '../components/SceneLayout';
import {sampleDocument} from '../data/sampleDocument';

export const Scene01Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const reveal = spring({frame, fps, config: {damping: 30, stiffness: 70}});
  const rightProgress = interpolate(frame, [80, 170], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <SceneLayout
      eyebrow="01 / Problem"
      title="Знания компании лежат в документах"
      subtitle="Важные факты есть в тексте, но их трудно проверить, связать и переиспользовать."
      frameLabel="Text -> facts"
    >
      <div style={{display: 'grid', gridTemplateColumns: '760px 1fr', gap: 58, alignItems: 'center', height: '100%'}}>
        <DocumentPanel paragraphs={sampleDocument} reveal={reveal} highlightProgress={reveal} />
        <div style={{display: 'grid', gap: 22, opacity: rightProgress, transform: `translateX(${(1 - rightProgress) * 42}px)`}}>
          <Panel style={{padding: 30}}>
            <div className="pipelinePill">
              <FileSearch size={24} />
              Скрытые сущности и факты
            </div>
            <p style={{fontSize: 29, lineHeight: 1.35, margin: '24px 0 0', color: '#172033', fontWeight: 720}}>
              Документ отвечает на вопросы только после ручного чтения или поиска похожих фрагментов.
            </p>
          </Panel>
          <Panel style={{padding: 30}}>
            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16}}>
              <div className="pipelinePill">
                <Network size={24} />
                граф фактов
              </div>
              <div className="pipelinePill">
                <ShieldCheck size={24} />
                проверка онтологией
              </div>
            </div>
            <p style={{fontSize: 28, lineHeight: 1.35, margin: '24px 0 0', color: '#647084'}}>
              Wikontic делает знание явным: сущности, связи, контекст и источник каждого ребра.
            </p>
          </Panel>
        </div>
      </div>
    </SceneLayout>
  );
};
