import React from 'react';
import type {SourceParagraph} from '../data/sampleDocument';

const highlightClass = {
  blue: 'hlBlue',
  green: 'hlGreen',
  amber: 'hlAmber',
  violet: 'hlViolet',
};

type DocumentPanelProps = {
  paragraphs: SourceParagraph[];
  reveal: number;
  highlightProgress?: number;
};

const renderHighlighted = (paragraph: SourceParagraph, visible: boolean) => {
  let remaining = paragraph.text;
  const parts: React.ReactNode[] = [];
  paragraph.spans.forEach((span) => {
    const index = remaining.indexOf(span.text);
    if (index < 0) {
      return;
    }
    const before = remaining.slice(0, index);
    if (before) {
      parts.push(before);
    }
    parts.push(
      <span key={span.id} className={`docHighlight ${visible ? highlightClass[span.color] : ''}`}>
        {span.text}
      </span>,
    );
    remaining = remaining.slice(index + span.text.length);
  });
  if (remaining) {
    parts.push(remaining);
  }
  return parts;
};

export const DocumentPanel: React.FC<DocumentPanelProps> = ({
  paragraphs,
  reveal,
  highlightProgress = 1,
}) => {
  return (
    <div className="documentPanel">
      <div className="docTop">
        <div className="docDot" />
        <div className="docDot" />
        <div className="docDot" />
        <span>domain-specification.txt</span>
      </div>
      <div className="docTitle">Техническое описание проекта</div>
      {paragraphs.map((paragraph, index) => {
        const visible = reveal > index / paragraphs.length;
        return (
          <p
            key={paragraph.text}
            style={{
              opacity: visible ? 1 : 0.2,
              transform: `translateY(${visible ? 0 : 18}px)`,
            }}
          >
            {renderHighlighted(paragraph, highlightProgress > index / paragraphs.length)}
          </p>
        );
      })}
    </div>
  );
};
