import React from 'react';
import {AbsoluteFill, Img, Sequence, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {MetricBadge} from './components/MetricBadge';
import {SceneLayout} from './components/SceneLayout';
import {Scene01Intro} from './scenes/Scene01Intro';
import {Scene02Extraction} from './scenes/Scene02Extraction';
import {Scene03DirtyGraph} from './scenes/Scene03DirtyGraph';
import {Scene04OntologyRefinement} from './scenes/Scene04OntologyRefinement';
import {Scene05Normalization} from './scenes/Scene05Normalization';
import {Scene06QuestionAnswering} from './scenes/Scene06QuestionAnswering';
import {Scene07SyntheticData} from './scenes/Scene07SyntheticData';
import {sceneFrames} from './theme';

const FadeScene: React.FC<{from: number; duration: number; children: React.ReactNode}> = ({
  from,
  duration,
  children,
}) => {
  const frame = useCurrentFrame();
  const local = frame - from;
  const opacity = Math.min(
    interpolate(local, [0, 18], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    interpolate(local, [duration - 20, duration], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
  );
  return (
    <Sequence from={from} durationInFrames={duration}>
      <AbsoluteFill style={{opacity}}>{children}</AbsoluteFill>
    </Sequence>
  );
};

const FinalFrame: React.FC = () => {
  const frame = useCurrentFrame();
  const reveal = interpolate(frame, [8, 48], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <SceneLayout
      eyebrow="Wikontic"
      title="из документов в проверяемую карту фактов"
      subtitle="Интерпретируемость · Multi-hop reasoning · Синтетические данные"
    >
      <div style={{height: '100%', display: 'grid', placeItems: 'center'}}>
        <div style={{display: 'grid', justifyItems: 'center', gap: 34, opacity: reveal, transform: `translateY(${(1 - reveal) * 28}px)`}}>
          <Img src={staticFile('assets/wikontic.png')} style={{width: 360, height: 'auto'}} />
          <div style={{display: 'flex', gap: 16}}>
            <MetricBadge value="84-86%" label="MINE-1 retention" tone="blue" />
            <MetricBadge value="76.0 F1" label="HotpotQA triplets-only" tone="green" />
            <MetricBadge value="59.8 F1" label="MuSiQue triplets-only" tone="amber" />
          </div>
        </div>
      </div>
    </SceneLayout>
  );
};

export const HeroVideo: React.FC = () => (
  <AbsoluteFill>
    <FadeScene {...sceneFrames.intro}>
      <Scene01Intro />
    </FadeScene>
    <FadeScene {...sceneFrames.extraction}>
      <Scene02Extraction />
    </FadeScene>
    <FadeScene {...sceneFrames.dirtyGraph}>
      <Scene03DirtyGraph />
    </FadeScene>
    <FadeScene {...sceneFrames.ontology}>
      <Scene04OntologyRefinement />
    </FadeScene>
    <FadeScene {...sceneFrames.normalization}>
      <Scene05Normalization />
    </FadeScene>
    <FadeScene {...sceneFrames.qa}>
      <Scene06QuestionAnswering />
    </FadeScene>
    <FadeScene {...sceneFrames.synthetic}>
      <Scene07SyntheticData />
    </FadeScene>
    <Sequence from={sceneFrames.finale.from} durationInFrames={sceneFrames.finale.duration}>
      <FinalFrame />
    </Sequence>
  </AbsoluteFill>
);
