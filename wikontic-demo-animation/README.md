# Wikontic Demo Animation

Local React + Remotion project for the Wikontic demo animations.

The project contains three independently renderable short animations and a separate presentation-style web viewer. Each animation starts automatically when its slide opens, and the viewer lets you pause, restart, jump between chapters, and move through the animations like a slide deck.

## Requirements

- Node.js 20+ recommended.
- npm 10+ recommended.
- A local Chromium-compatible environment for Remotion rendering. Remotion downloads/uses its own browser tooling through the installed packages.
- No backend services are required.
- No API keys are required.
- No live LLM/API calls are made.

## Install

From this folder:

```bash
cd wikontic-demo-animation
npm install
```

The source of truth for dependency versions is `package-lock.json`.

## Run The Presentation Viewer

This is the recommended mode for a live demo.

```bash
npm run present
```

Open:

```text
http://localhost:5173/presentation.html
```

If port `5173` is already busy, Vite may choose another port. You can also specify one explicitly:

```bash
npm run present -- --host=127.0.0.1 --port=5174
```

Presentation controls:

- animations start automatically when their slide opens
- `Space`: play / pause current animation
- `R`: restart current animation
- `Left` / `Right`: previous / next slide
- `,` / `.`: step backward / forward by 15 frames
- number keys: jump to chapter markers in the current animation
- on-screen buttons: play, pause, restart, frame step, fullscreen, chapter jump

The presentation viewer uses the Remotion React compositions directly through `@remotion/player`. It does not depend on the exported MP4 files.

## Run Remotion Studio

Use this mode for authoring and inspecting Remotion compositions directly.

```bash
npm run preview
```

This opens Remotion Studio for the compositions registered in `src/Root.tsx`.

Current composition IDs:

- `TextToGraph`
- `RagVsWikontic`
- `SyntheticDataFactory`

## Render Videos

Render all three standalone MP4 files:

```bash
npm run render:all
```

Render one animation at a time:

```bash
npm run render:text-to-graph
npm run render:rag-vs-wikontic
npm run render:synthetic-data
```

Outputs are written to:

```text
out/text-to-graph.mp4
out/rag-vs-wikontic.mp4
out/synthetic-data-factory.mp4
```

The `out/` folder is treated as generated output and should not be committed.

## Optional Graph Demo

There is also a simple backup graph demo:

```bash
npm run graph:preview
```

This runs the Vite app entry in `index.html`, which mounts `src/interactive/GraphDemo.tsx`.

## Code Organization

Important entrypoints:

- `src/index.ts`: Remotion registration entrypoint.
- `src/Root.tsx`: registers the three independent Remotion compositions.
- `presentation.html`: separate Vite entry for the presentation viewer.
- `src/presentation/main.tsx`: mounts the presentation viewer.
- `src/main.tsx`: mounts the optional graph demo.

Animation scenes:

- `src/scenes/Animation1_TextToGraph.tsx`: method animation, text to structured graph.
- `src/scenes/Animation2_RagVsGraph.tsx`: RAG comparison animation.
- `src/scenes/Animation3_SyntheticData.tsx`: synthetic data generation animation.

Presentation viewer:

- `src/presentation/PresentationApp.tsx`: slide-deck UI, Remotion Player, controls, final metrics slide.
- `src/presentation/presentationConfig.ts`: slide list, component references, durations, chapter frame markers.
- `src/presentation/presentation.css`: presentation-only styling.

Data:

- `src/data/animation1.ts`: deterministic document, facts, triplets, ontology checks, compact graph.
- `src/data/animation2.ts`: natural business question, RAG chunks, internal graph path, answer.
- `src/data/animation3.ts`: answer node, sampled paths, natural QA cards, dataset labels.

Shared visual components:

- `src/components/GraphView.tsx`: SVG graph rendering.
- `src/components/DocumentView.tsx`: document text with highlightable fact spans.
- `src/components/TripletCard.tsx`: extracted fact card.
- `src/components/OntologyPass.tsx`: ontology validation panel.
- `src/components/DedupMerge.tsx`: alias/deduplication merge visualization.
- `src/components/DatasetCard.tsx`: synthetic QA card.
- `src/components/MetricBadge.tsx`: metric badges used in final slide.

Legacy files:

- `src/HeroVideo.tsx` and `src/scenes/Scene01*` through `Scene07*` belong to the earlier single 90-second hero video version. The active v2 Remotion compositions are the three `Animation*` scene files listed above.

## Assets

Assets are stored in:

```text
assets/
public/assets/
```

The presentation and Remotion components use the public assets path where needed, for example `/assets/wikontic.png`.

## Validation

Run TypeScript checks:

```bash
npm run check
```

Expected constraints:

- deterministic hardcoded data only
- no backend requests
- no external API calls
- Russian labels remain readable at 1920x1080
- presentation viewer does not modify the animation scene files
