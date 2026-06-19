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

English is the default. Open the Russian version with:

```text
http://localhost:5173/presentation.html?lang=ru
```

The viewer also has an `EN` / `RU` language toggle in the sidebar.

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
- `TextToGraphRU`
- `TextToGraphEN`
- `RagVsWikontic`
- `RagVsWikonticRU`
- `RagVsWikonticEN`
- `SyntheticDataFactory`
- `SyntheticDataFactoryRU`
- `SyntheticDataFactoryEN`

## Render Videos

Render all three standalone MP4 files in English:

```bash
npm run render:all
```

Render all three standalone MP4 files in Russian:

```bash
npm run render:all:ru
```

Render one animation at a time:

```bash
npm run render:text-to-graph
npm run render:text-to-graph:en
npm run render:text-to-graph:ru
npm run render:rag-vs-wikontic
npm run render:rag-vs-wikontic:en
npm run render:rag-vs-wikontic:ru
npm run render:synthetic-data
npm run render:synthetic-data:en
npm run render:synthetic-data:ru
```

Outputs are written to:

```text
out/text-to-graph.mp4
out/text-to-graph-en.mp4
out/text-to-graph-ru.mp4
out/rag-vs-wikontic.mp4
out/rag-vs-wikontic-en.mp4
out/rag-vs-wikontic-ru.mp4
out/synthetic-data-factory.mp4
out/synthetic-data-factory-en.mp4
out/synthetic-data-factory-ru.mp4
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
- `src/presentation/presentationConfig.ts`: localized slide list, component references, durations, chapter frame markers.
- `src/presentation/presentation.css`: presentation-only styling.
- `src/i18n/types.ts`: shared locale type and locale normalization helper.

Data:

- `src/data/animation1.ts`: localized deterministic document, facts, triplets, ontology checks, compact graph.
- `src/data/animation2.ts`: localized natural business question, RAG chunks, internal graph path, answer.
- `src/data/animation3.ts`: localized answer node, sampled paths, natural QA cards, dataset labels.

## Maintaining Two Languages

English is the default language. Russian is selected in the viewer with `?lang=ru` or the `RU` toggle.

When changing demo content:

- Edit animation copy/data in `src/data/animation1.ts`, `src/data/animation2.ts`, and `src/data/animation3.ts`.
- Edit slide titles, subtitles, and chapter labels in `src/presentation/presentationConfig.ts`.
- Edit final metrics slide copy and viewer UI labels in `src/presentation/PresentationApp.tsx`.
- Keep graph node IDs, edge IDs, path IDs, and card counts identical across `en` and `ru`; translate only visible labels, questions, answers, paths, and display text.
- If changing layout, timing, or animation behavior, do it once in the scene/CSS files and verify both languages still fit.

After bilingual edits, run:

```bash
npm run check
```

For visual changes, spot-check representative stills in both English and Russian.

Shared visual components:

- `src/components/GraphView.tsx`: SVG graph rendering.
- `src/components/DocumentView.tsx`: document text with highlightable fact spans.
- `src/components/TripletCard.tsx`: extracted fact card.
- `src/components/OntologyPass.tsx`: ontology validation panel.
- `src/components/DedupMerge.tsx`: alias/deduplication merge visualization.
- `src/components/DatasetCard.tsx`: synthetic QA card.
- `src/components/MetricBadge.tsx`: metric badges used in final slide.

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
- Russian and English labels remain readable at 1920x1080
- presentation viewer does not modify the animation scene files
