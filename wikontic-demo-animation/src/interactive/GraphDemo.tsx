import React, {useMemo, useState} from 'react';
import {GraphView} from '../components/GraphView';
import type {GraphNode} from '../data/graphBefore';
import {qaPathEdgeIds, qaPathNodeIds, refinedEdges, refinedNodes} from '../data/graphAfter';
import {ragQuestion as question, ragVsWikonticAnswer as answer} from '../data/animation2';
import '../styles/global.css';

const aliases: Record<string, string[]> = {
  complex_clean: ['ЖК', 'объект', 'жилой комплекс'],
  networks_clean: ['сети', 'инженерные сети'],
  monitoring: ['система мониторинга'],
};

export const GraphDemo: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<GraphNode>(refinedNodes[0]);
  const [showPath, setShowPath] = useState(true);
  const selectedEdges = useMemo(
    () => refinedEdges.filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id),
    [selectedNode.id],
  );

  return (
    <main className="interactivePage">
      <header className="interactiveHeader">
        <img src="/assets/wikotic-wo-text.png" alt="Wikontic" />
        <div>
          <h1>Wikontic backup graph demo</h1>
          <p style={{margin: '6px 0 0', color: '#647084'}}>Детерминированный локальный граф без API и backend-запросов</p>
        </div>
      </header>
      <section className="interactiveGrid">
        <div className="panel" style={{padding: 18}}>
          <GraphView
            nodes={refinedNodes}
            edges={refinedEdges}
            reveal={1}
            showTypes
            highlightedNodeIds={showPath ? qaPathNodeIds : [selectedNode.id]}
            highlightedEdgeIds={showPath ? qaPathEdgeIds : selectedEdges.map((edge) => edge.id)}
            muted
            onNodeSelect={(node) => {
              setSelectedNode(node);
              setShowPath(false);
            }}
            width={1120}
            height={700}
          />
        </div>
        <aside className="sidePanel">
          <h2>{selectedNode.label}</h2>
          <p><strong>Тип:</strong> {selectedNode.type}</p>
          <p><strong>Алиасы:</strong> {(aliases[selectedNode.id] ?? [selectedNode.label]).join(', ')}</p>
          <p><strong>Связи:</strong> {selectedEdges.map((edge) => edge.label).join(', ') || 'нет'}</p>
          <p><strong>Источник:</strong> техническое описание проекта, абзац 2</p>
          <hr style={{border: 0, borderTop: '1px solid #dbe3ef', margin: '20px 0'}} />
          <p><strong>Вопрос:</strong> {question}</p>
          <p><strong>Ответ:</strong> {answer}</p>
          <button onClick={() => setShowPath(true)}>Показать QA-путь</button>
        </aside>
      </section>
    </main>
  );
};
