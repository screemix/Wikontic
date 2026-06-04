import React from 'react';
import {createRoot} from 'react-dom/client';
import {GraphDemo} from './interactive/GraphDemo';

createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <GraphDemo />
  </React.StrictMode>,
);
