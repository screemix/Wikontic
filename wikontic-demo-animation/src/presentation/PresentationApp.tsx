import React, {useCallback, useEffect, useRef, useState} from 'react';
import {Player} from '@remotion/player';
import type {CallbackListener, PlayerRef} from '@remotion/player';
import {ChevronLeft, ChevronRight, Maximize2, Pause, Play, RotateCcw, SkipBack, SkipForward} from 'lucide-react';
import {MetricBadge} from '../components/MetricBadge';
import {getPresentationSlides, presentationVideo} from './presentationConfig';
import type {AnimationSlide} from './presentationConfig';
import {normalizeLocale} from '../i18n/types';
import type {Locale} from '../i18n/types';

const clampFrame = (frame: number, slide: AnimationSlide) =>
  Math.max(0, Math.min(slide.durationInFrames - 1, Math.round(frame)));

const formatTime = (frame: number, duration: number) => `${frame} / ${duration - 1}`;

const presentationCopy: Record<Locale, {
  metrics: {
    title: string;
    subtitle: string;
    badges: {value: string; label: string; tone: 'blue' | 'green' | 'amber'}[];
  };
  controls: {
    back: string;
    pause: string;
    continue: string;
    restart: string;
    next: string;
    finalNote: string;
    hints: string[];
  };
}> = {
  ru: {
    metrics: {
      title: 'из документов в проверяемую базу фактов',
      subtitle: 'Интерпретируемость · Multi-hop reasoning · Генерация синтетических данных',
      badges: [
        {value: '84-86%', label: 'извлеченных фактов на MINE-1', tone: 'blue'},
        {value: '76.0 F1', label: 'QA только по графу на HotpotQA', tone: 'green'},
        {value: 'x20 экономия токенов', label: 'по сравнению с GraphRAG', tone: 'blue'},
      ],
    },
    controls: {
      back: 'Назад',
      pause: 'Пауза',
      continue: 'Продолжить',
      restart: 'Сначала',
      next: 'Далее',
      finalNote: 'Финальный слайд статичен: используйте стрелки, чтобы вернуться к анимациям.',
      hints: ['Space: play/pause', '← / →: slide', 'R: restart', ', / .: ±15 frames'],
    },
  },
  en: {
    metrics: {
      title: 'from documents to a verifiable fact base',
      subtitle: 'Interpretability · Multi-hop reasoning · Synthetic data generation',
      badges: [
        {value: '84-86%', label: 'fact retention on MINE-1', tone: 'blue'},
        {value: '76.0 F1', label: 'graph-only QA on HotpotQA', tone: 'green'},
        {value: 'x20 token economy', label: 'compared with GraphRAG', tone: 'blue'},
      ],
    },
    controls: {
      back: 'Back',
      pause: 'Pause',
      continue: 'Continue',
      restart: 'Restart',
      next: 'Next',
      finalNote: 'The final slide is static: use arrows to return to the animations.',
      hints: ['Space: play/pause', '← / →: slide', 'R: restart', ', / .: ±15 frames'],
    },
  },
};

const MetricsSlide: React.FC<{locale: Locale}> = ({locale}) => {
  const copy = presentationCopy[locale].metrics;
  return (
    <div className="metricsSlide">
      <div className="metricsGridTexture" aria-hidden />
      <div className="metricsContent">
        <img src="/assets/wikontic.png" alt="Wikontic" className="metricsLogo" />
        <div>
          <h1>{copy.title}</h1>
          <p>{copy.subtitle}</p>
        </div>
        <div className="metricsBadges">
          {copy.badges.map((badge) => (
            <MetricBadge key={`${badge.value}-${badge.label}`} {...badge} />
          ))}
        </div>
      </div>
    </div>
  );
};

const StaticStage: React.FC<{children: React.ReactNode}> = ({children}) => {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({
    width: presentationVideo.width,
    height: presentationVideo.height,
  });

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const updateSize = () => {
      const rect = viewport.getBoundingClientRect();
      setSize({width: rect.width, height: rect.height});
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);

  const scale = Math.min(size.width / presentationVideo.width, size.height / presentationVideo.height);

  return (
    <div className="staticStageViewport" ref={viewportRef}>
      <div
        className="staticStageCanvas"
        style={{
          width: presentationVideo.width,
          height: presentationVideo.height,
          transform: `translate(-50%, -50%) scale(${scale})`,
        }}
      >
        {children}
      </div>
    </div>
  );
};

export const PresentationApp: React.FC = () => {
  const [locale, setLocale] = useState<Locale>(() => normalizeLocale(new URLSearchParams(window.location.search).get('lang')));
  const presentationSlides = getPresentationSlides(locale);
  const copy = presentationCopy[locale].controls;
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

  const changeLocale = useCallback(
    (nextLocale: Locale) => {
      if (nextLocale === locale) {
        return;
      }
      pause();
      setCurrentFrame(0);
      setLocale(nextLocale);

      const url = new URL(window.location.href);
      if (nextLocale === 'en') {
        url.searchParams.delete('lang');
      } else {
        url.searchParams.set('lang', nextLocale);
      }
      window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    },
    [locale, pause],
  );

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
    [pause, presentationSlides.length],
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
            {/* <span>Demo slide deck</span> */}
          </div>
        </div>

        <div className="localeToggle" aria-label="Presentation language">
          <button
            type="button"
            className={locale === 'en' ? 'activeLocale' : undefined}
            onClick={() => changeLocale('en')}
          >
            EN
          </button>
          <button
            type="button"
            className={locale === 'ru' ? 'activeLocale' : undefined}
            onClick={() => changeLocale('ru')}
          >
            RU
          </button>
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
          {copy.hints.map((hint) => (
            <span key={hint}>{hint}</span>
          ))}
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
                key={`${activeAnimation.id}-${locale}`}
                ref={setPlayerRef}
                component={activeAnimation.component}
                inputProps={{locale}}
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
              <StaticStage>
                <MetricsSlide locale={locale} />
              </StaticStage>
            )}
          </div>
        </div>

        <div className="presentationControls">
          <div className="transportControls">
            <button type="button" onClick={() => goToSlide(activeIndex - 1)} disabled={activeIndex === 0}>
              <ChevronLeft size={20} />
              {copy.back}
            </button>
            <button type="button" onClick={togglePlayback} disabled={!activeAnimation}>
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              {isPlaying ? copy.pause : copy.continue}
            </button>
            <button type="button" onClick={restart} disabled={!activeAnimation}>
              <RotateCcw size={20} />
              {copy.restart}
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
              {copy.next}
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
              {copy.finalNote}
            </div>
          )}
        </div>
      </section>
    </main>
  );
};
