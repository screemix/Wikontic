import React from 'react';
import {colors} from '../theme';

type SceneLayoutProps = {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  frameLabel?: string;
};

export const SceneLayout: React.FC<SceneLayoutProps> = ({
  eyebrow,
  title,
  subtitle,
  children,
  frameLabel,
}) => {
  return (
    <div className="scene">
      <div className="sceneHeader">
        <div>
          {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
          <h1>{title}</h1>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {frameLabel ? <div className="frameLabel">{frameLabel}</div> : null}
      </div>
      <div className="sceneBody">{children}</div>
      <div className="gridTexture" aria-hidden />
      <div className="cornerMark" style={{borderColor: colors.blueSoft}} aria-hidden />
    </div>
  );
};

export const Panel: React.FC<{
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}> = ({children, className, style}) => (
  <div className={`panel ${className ?? ''}`} style={style}>
    {children}
  </div>
);
