import React from 'react';
import {Composition} from 'remotion';
import {Animation1_TextToGraph, TEXT_TO_GRAPH_FRAMES} from './scenes/Animation1_TextToGraph';
import {Animation2_RagVsGraph, RAG_VS_WIKONTIC_FRAMES} from './scenes/Animation2_RagVsGraph';
import {Animation3_SyntheticData, SYNTHETIC_DATA_FRAMES} from './scenes/Animation3_SyntheticData';
import {FPS, VIDEO_HEIGHT, VIDEO_WIDTH} from './theme';
import type {Locale} from './i18n/types';
import './styles/global.css';

const localeProps = (locale: Locale) => ({locale});

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="TextToGraph"
        component={Animation1_TextToGraph}
        durationInFrames={TEXT_TO_GRAPH_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('en')}
      />
      <Composition
        id="TextToGraphRU"
        component={Animation1_TextToGraph}
        durationInFrames={TEXT_TO_GRAPH_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('ru')}
      />
      <Composition
        id="TextToGraphEN"
        component={Animation1_TextToGraph}
        durationInFrames={TEXT_TO_GRAPH_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('en')}
      />
      <Composition
        id="RagVsWikontic"
        component={Animation2_RagVsGraph}
        durationInFrames={RAG_VS_WIKONTIC_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('en')}
      />
      <Composition
        id="RagVsWikonticRU"
        component={Animation2_RagVsGraph}
        durationInFrames={RAG_VS_WIKONTIC_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('ru')}
      />
      <Composition
        id="RagVsWikonticEN"
        component={Animation2_RagVsGraph}
        durationInFrames={RAG_VS_WIKONTIC_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('en')}
      />
      <Composition
        id="SyntheticDataFactory"
        component={Animation3_SyntheticData}
        durationInFrames={SYNTHETIC_DATA_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('en')}
      />
      <Composition
        id="SyntheticDataFactoryRU"
        component={Animation3_SyntheticData}
        durationInFrames={SYNTHETIC_DATA_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('ru')}
      />
      <Composition
        id="SyntheticDataFactoryEN"
        component={Animation3_SyntheticData}
        durationInFrames={SYNTHETIC_DATA_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={localeProps('en')}
      />
    </>
  );
};
