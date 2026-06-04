import React from 'react';
import {BadgeCheck, ShieldCheck} from 'lucide-react';

type OntologyScannerProps = {
  progress: number;
};

export const OntologyScanner: React.FC<OntologyScannerProps> = ({progress}) => {
  const x = -8 + progress * 116;
  return (
    <div className="ontologyLayer">
      <div className="ontologyHeader">
        <ShieldCheck size={24} />
        <span>Wikidata ontology layer</span>
      </div>
      <div className="ontologyRules">
        <span>тип субъекта</span>
        <span>допустимая связь</span>
        <span>тип объекта</span>
      </div>
      <div className="scannerTrack">
        <div className="scannerBeam" style={{left: `${x}%`}} />
      </div>
      <div className="ontologyChecks">
        <div>
          <BadgeCheck size={18} />
          строительный объект {'->'} включает {'->'} инженерная система
        </div>
        <div>
          <BadgeCheck size={18} />
          дата хранится как qualifier / time entity
        </div>
      </div>
    </div>
  );
};
