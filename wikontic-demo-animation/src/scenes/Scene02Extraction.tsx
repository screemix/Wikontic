import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {DocumentPanel} from '../components/DocumentPanel';
import {SceneLayout} from '../components/SceneLayout';
import {TripletCard} from '../components/TripletCard';
import {extractedTriplets, sampleDocument} from '../data/sampleDocument';

export const Scene02Extraction: React.FC = () => {
  const frame = useCurrentFrame();
  const docScale = interpolate(frame, [0, 90], [1, 0.92], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const arrowProgress = interpolate(frame, [60, 150], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <SceneLayout
      eyebrow="02 / Candidate extraction"
      title="LLM извлекает факты и контекст"
      subtitle="Фрагменты текста превращаются в кандидаты: subject, relation, object и qualifier."
      frameLabel="candidate triplets"
    >
      <div style={{display: 'grid', gridTemplateColumns: '650px 180px 1fr', alignItems: 'center', gap: 30, height: '100%'}}>
        <div style={{transform: `scale(${docScale})`, transformOrigin: 'center'}}>
          <DocumentPanel paragraphs={sampleDocument} reveal={1} highlightProgress={1} />
        </div>
        <svg viewBox="0 0 180 220" style={{width: 180, opacity: arrowProgress}}>
          <path d="M22 110 H146" stroke="#2f6df6" strokeWidth="8" strokeLinecap="round" />
          <path d="M122 75 L158 110 L122 145" fill="none" stroke="#2f6df6" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
          <text x="90" y="190" textAnchor="middle" style={{fontSize: 22, fill: '#647084', fontWeight: 760}}>
            LLM
          </text>
        </svg>
        <div style={{display: 'grid', gap: 18}}>
          {extractedTriplets.map((triplet, index) => {
            const progress = interpolate(frame, [70 + index * 32, 138 + index * 32], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            });
            return <TripletCard key={triplet.id} {...triplet} progress={progress} compact />;
          })}
        </div>
      </div>
    </SceneLayout>
  );
};
