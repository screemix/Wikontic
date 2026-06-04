import React from 'react';
import {Composition} from 'remotion';
import {Animation1_TextToGraph, TEXT_TO_GRAPH_FRAMES} from './scenes/Animation1_TextToGraph';
import {Animation2_RagVsGraph, RAG_VS_WIKONTIC_FRAMES} from './scenes/Animation2_RagVsGraph';
import {Animation3_SyntheticData, SYNTHETIC_DATA_FRAMES} from './scenes/Animation3_SyntheticData';
import {FPS, VIDEO_HEIGHT, VIDEO_WIDTH} from './theme';
import './styles/global.css';

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
        defaultProps={{}}
      />
      <Composition
        id="RagVsWikontic"
        component={Animation2_RagVsGraph}
        durationInFrames={RAG_VS_WIKONTIC_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={{}}
      />
      <Composition
        id="SyntheticDataFactory"
        component={Animation3_SyntheticData}
        durationInFrames={SYNTHETIC_DATA_FRAMES}
        fps={FPS}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        defaultProps={{}}
      />
    </>
  );
};
