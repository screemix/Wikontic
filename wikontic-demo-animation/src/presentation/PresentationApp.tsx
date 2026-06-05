import React, {useCallback, useEffect, useRef, useState} from 'react';
import {Player} from '@remotion/player';
import type {CallbackListener, PlayerRef} from '@remotion/player';
import {ChevronLeft, ChevronRight, Maximize2, Pause, Play, RotateCcw, SkipBack, SkipForward} from 'lucide-react';
import {MetricBadge} from '../components/MetricBadge';
import {presentationSlides, presentationVideo} from './presentationConfig';
import type {AnimationSlide} from './presentationConfig';

const clampFrame = (frame: number, slide: AnimationSlide) =>
  Math.max(0, Math.min(slide.durationInFrames - 1, Math.round(frame)));

const formatTime = (frame: number, duration: number) => `${frame} / ${duration - 1}`;

const MetricsSlide: React.FC = () => (
  <div className="metricsSlide">
    <div className="metricsGridTexture" aria-hidden />
    <div className="metricsContent">
      <img src="/assets/wikontic.png" alt="Wikontic" className="metricsLogo" />
      <div>
        <h1>из документов в проверяемую базу фактов</h1>
        <p>Интерпретируемость · Multi-hop reasoning · Синтетические данные</p>
      </div>
      <div className="metricsBadges">
        <MetricBadge value="84-86%" label="MINE-1 retention" tone="blue" />
        <MetricBadge value="76.0 F1" label="HotpotQA triplets-only" tone="green" />
        <MetricBadge value="59.8 F1" label="MuSiQue triplets-only" tone="amber" />
      </div>
    </div>
  </div>
);

export const PresentationApp: React.FC = () => {
  const [activeIndex, setActiveIndex] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playerRef = useRef<PlayerRef>(null);
  const [playerHandle, setPlayerHandle] = useState<PlayerRef | null>(null);
  const activeSlide = presentationSlides[activeIndex];
  const activeAnimation = activeSlide.kind === 'animation' ? activeSlide : null;

  const setPlayerRef = useCallback((node: PlayerRef | null) => {
    playerRef.current = node;
    setPlayerHandle(node);
  }, []);

  const pause = useCallback(() => {
    playerRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const play = useCallback(() => {
    if (!activeAnimation) {
      return;
    }
    playerRef.current?.play();
    setIsPlaying(true);
  }, [activeAnimation]);

  const seekTo = useCallback(
    (frame: number) => {
      if (!activeAnimation) {
        return;
      }
      const nextFrame = clampFrame(frame, activeAnimation);
      playerRef.current?.seekTo(nextFrame);
      setCurrentFrame(nextFrame);
    },
    [activeAnimation],
  );

  const restart = useCallback(() => {
    if (!activeAnimation) {
      return;
    }
    seekTo(0);
    playerRef.current?.play();
    setIsPlaying(true);
  }, [activeAnimation, seekTo]);

  const goToSlide = useCallback(
    (nextIndex: number) => {
      pause();
      setActiveIndex(Math.max(0, Math.min(presentationSlides.length - 1, nextIndex)));
      setCurrentFrame(0);
    },
    [pause],
  );

  const togglePlayback = useCallback(() => {
    if (!activeAnimation) {
      return;
    }
    if (playerRef.current?.isPlaying()) {
      pause();
    } else {
      play();
    }
  }, [activeAnimation, pause, play]);

  useEffect(() => {
    if (!activeAnimation || !playerHandle) {
      return;
    }

    const timer = window.setTimeout(() => {
      playerHandle.seekTo(0);
      setCurrentFrame(0);
      playerHandle.play();
      setIsPlaying(true);
    }, 0);

    return () => window.clearTimeout(timer);
  }, [activeAnimation?.id, playerHandle]);

  useEffect(() => {
    if (!activeAnimation) {
      setIsPlaying(false);
      setCurrentFrame(0);
      return;
    }

    const player = playerRef.current;
    if (!player) {
      return;
    }

    const onFrame: CallbackListener<'frameupdate'> = ({detail}) => setCurrentFrame(detail.frame);
    const onPlay: CallbackListener<'play'> = () => setIsPlaying(true);
    const onPause: CallbackListener<'pause'> = () => setIsPlaying(false);
    const onEnded: CallbackListener<'ended'> = () => setIsPlaying(false);

    player.addEventListener('frameupdate', onFrame);
    player.addEventListener('play', onPlay);
    player.addEventListener('pause', onPause);
    player.addEventListener('ended', onEnded);

    return () => {
      player.removeEventListener('frameupdate', onFrame);
      player.removeEventListener('play', onPlay);
      player.removeEventListener('pause', onPause);
      player.removeEventListener('ended', onEnded);
    };
  }, [activeAnimation?.id]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) {
        return;
      }

      if (event.key === ' ') {
        event.preventDefault();
        togglePlayback();
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        goToSlide(activeIndex + 1);
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        goToSlide(activeIndex - 1);
      } else if (event.key.toLowerCase() === 'r') {
        event.preventDefault();
        restart();
      } else if (event.key === ',') {
        event.preventDefault();
        seekTo(currentFrame - 15);
      } else if (event.key === '.') {
        event.preventDefault();
        seekTo(currentFrame + 15);
      } else if (/^[1-9]$/.test(event.key) && activeAnimation) {
        const chapter = activeAnimation.chapters[Number(event.key) - 1];
        if (chapter) {
          event.preventDefault();
          pause();
          seekTo(chapter.frame);
        }
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeAnimation, activeIndex, currentFrame, goToSlide, pause, restart, seekTo, togglePlayback]);

  return (
    <main className="presentationShell">
      <aside className="presentationSidebar">
        <div className="presentationBrand">
          <img src="/assets/wikotic-wo-text.png" alt="Wikontic" />
          <div>
            <strong>Wikontic</strong>
            <span>Demo slide deck</span>
          </div>
        </div>

        <nav className="slideList" aria-label="Presentation slides">
          {presentationSlides.map((slide, index) => (
            <button
              key={slide.id}
              type="button"
              className={index === activeIndex ? 'activeSlideButton' : undefined}
              onClick={() => goToSlide(index)}
            >
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{slide.title}</strong>
              <small>{slide.subtitle}</small>
            </button>
          ))}
        </nav>

        <div className="keyboardHints">
          <span>Space: play/pause</span>
          <span>← / →: slide</span>
          <span>R: restart</span>
          <span>, / .: ±15 frames</span>
        </div>
      </aside>

      <section className="presentationMain">
        <header className="presentationTopbar">
          <div>
            <div className="presentationEyebrow">
              {activeIndex + 1} / {presentationSlides.length}
            </div>
            <h1>{activeSlide.title}</h1>
            <p>{activeSlide.subtitle}</p>
          </div>
          {activeAnimation ? (
            <div className="frameCounter">{formatTime(currentFrame, activeAnimation.durationInFrames)}</div>
          ) : null}
        </header>

        <div className="stageOuter">
          <div className="stageFrame">
            {activeAnimation ? (
              <Player
                key={activeAnimation.id}
                ref={setPlayerRef}
                component={activeAnimation.component}
                durationInFrames={activeAnimation.durationInFrames}
                fps={presentationVideo.fps}
                compositionWidth={presentationVideo.width}
                compositionHeight={presentationVideo.height}
                controls={false}
                autoPlay
                initialFrame={0}
                initiallyMuted
                clickToPlay={false}
                spaceKeyToPlayOrPause={false}
                doubleClickToFullscreen={false}
                moveToBeginningWhenEnded={false}
                style={{width: '100%', height: '100%'}}
              />
            ) : (
              <MetricsSlide />
            )}
          </div>
        </div>

        <div className="presentationControls">
          <div className="transportControls">
            <button type="button" onClick={() => goToSlide(activeIndex - 1)} disabled={activeIndex === 0}>
              <ChevronLeft size={20} />
              Назад
            </button>
            <button type="button" onClick={togglePlayback} disabled={!activeAnimation}>
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              {isPlaying ? 'Пауза' : 'Продолжить'}
            </button>
            <button type="button" onClick={restart} disabled={!activeAnimation}>
              <RotateCcw size={20} />
              Сначала
            </button>
            <button type="button" onClick={() => seekTo(currentFrame - 15)} disabled={!activeAnimation}>
              <SkipBack size={20} />
              15f
            </button>
            <button type="button" onClick={() => seekTo(currentFrame + 15)} disabled={!activeAnimation}>
              <SkipForward size={20} />
              15f
            </button>
            <button type="button" onClick={() => playerRef.current?.requestFullscreen()} disabled={!activeAnimation}>
              <Maximize2 size={20} />
              Fullscreen
            </button>
            <button
              type="button"
              onClick={() => goToSlide(activeIndex + 1)}
              disabled={activeIndex === presentationSlides.length - 1}
            >
              Далее
              <ChevronRight size={20} />
            </button>
          </div>

          {activeAnimation ? (
            <>
              <input
                className="frameScrubber"
                type="range"
                min={0}
                max={activeAnimation.durationInFrames - 1}
                value={currentFrame}
                onChange={(event) => {
                  pause();
                  seekTo(Number(event.currentTarget.value));
                }}
              />
              <div className="chapterControls">
                {activeAnimation.chapters.map((chapter, index) => (
                  <button
                    key={`${activeAnimation.id}-${chapter.label}`}
                    type="button"
                    onClick={() => {
                      pause();
                      seekTo(chapter.frame);
                    }}
                    className={Math.abs(currentFrame - chapter.frame) < 24 ? 'activeChapter' : undefined}
                  >
                    <span>{index + 1}</span>
                    {chapter.label}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <div className="chapterControls finalSlideNote">
              Финальный слайд статичен: используйте стрелки, чтобы вернуться к анимациям.
            </div>
          )}
        </div>
      </section>
    </main>
  );
};
