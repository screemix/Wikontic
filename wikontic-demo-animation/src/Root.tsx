import React from 'react';
import {Composition} from 'remotion';
import {HeroVideo} from './HeroVideo';
import {DURATION_IN_FRAMES, FPS, VIDEO_HEIGHT, VIDEO_WIDTH} from './theme';
import './styles/global.css';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="WikonticHero"
      component={HeroVideo}
      durationInFrames={DURATION_IN_FRAMES}
      fps={FPS}
      width={VIDEO_WIDTH}
      height={VIDEO_HEIGHT}
      defaultProps={{}}
    />
  );
};
