export type HighlightSpan = {
  id: string;
  text: string;
  color: 'blue' | 'green' | 'amber' | 'violet';
};

export type SourceParagraph = {
  text: string;
  spans: HighlightSpan[];
};

export const sampleDocument: SourceParagraph[] = [
  {
    text: 'В 2024 году проектная команда утвердила требования к жилому комплексу.',
    spans: [
      {id: 'date', text: 'В 2024 году', color: 'amber'},
      {id: 'team', text: 'проектная команда', color: 'violet'},
      {id: 'requirements', text: 'утвердила требования', color: 'green'},
      {id: 'complex', text: 'жилому комплексу', color: 'blue'},
    ],
  },
  {
    text: 'Объект включает три корпуса, подземный паркинг и систему мониторинга инженерных сетей.',
    spans: [
      {id: 'object', text: 'Объект', color: 'blue'},
      {id: 'relation', text: 'включает', color: 'green'},
      {id: 'buildings', text: 'три корпуса', color: 'violet'},
      {id: 'parking', text: 'подземный паркинг', color: 'violet'},
      {id: 'monitoring', text: 'систему мониторинга', color: 'blue'},
      {id: 'networks', text: 'инженерных сетей', color: 'blue'},
    ],
  },
  {
    text: 'Для приемки подрядчик должен предоставить журнал проверок и отчеты по датчикам.',
    spans: [
      {id: 'contractor', text: 'подрядчик', color: 'violet'},
      {id: 'logs', text: 'журнал проверок', color: 'green'},
      {id: 'sensors', text: 'отчеты по датчикам', color: 'blue'},
    ],
  },
];

export const extractedTriplets = [
  {
    id: 't1',
    subject: 'жилой комплекс',
    relation: 'включает',
    object: 'три корпуса',
    qualifier: 'контекст: требования',
  },
  {
    id: 't2',
    subject: 'жилой комплекс',
    relation: 'включает',
    object: 'подземный паркинг',
    qualifier: 'контекст: требования',
  },
  {
    id: 't3',
    subject: 'система мониторинга',
    relation: 'относится к',
    object: 'инженерным сетям',
    qualifier: 'источник: абзац 2',
  },
  {
    id: 't4',
    subject: 'утверждение требований',
    relation: 'дата',
    object: '2024',
    qualifier: 'тип: время',
  },
];

export const question = 'Какие инженерные элементы связаны с жилым комплексом через систему мониторинга?';

export const answer = 'Система мониторинга связана с инженерными сетями жилого комплекса.';
