import React from 'react';
import {createRoot} from 'react-dom/client';
import {PresentationApp} from './PresentationApp';
import '../styles/global.css';
import './presentation.css';

createRoot(document.getElementById('presentation-root') as HTMLElement).render(
  <React.StrictMode>
    <PresentationApp />
  </React.StrictMode>,
);
