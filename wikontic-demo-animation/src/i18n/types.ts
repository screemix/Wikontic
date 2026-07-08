export type Locale = 'ru' | 'en';

export const normalizeLocale = (value: string | null | undefined): Locale => (value === 'ru' ? 'ru' : 'en');
