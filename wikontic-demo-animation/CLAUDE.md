# CLAUDE.md — Wikontic Demo Animation

Guidance for building a **clean, executive-facing visual demo** of the Wikontic method.
The current code in this folder is a **rough first pass — use it for inspiration only, do not rely on it.** We are building a prettier, more deliberate version.

---

## 1. What we are building

Two short, independently-rendered animations that explain Wikontic to a **non-technical, high-level executive** as clearly as possible.

- **Animation 1 — Method (text → structured graph):** how a fact/knowledge flows out of unstructured text into a structured knowledge graph.
- **Animation 2 — Wikontic vs RAG (multi-hop):** for a complex multi-hop question, Wikontic finds an explicit *path* through facts; ordinary RAG only retrieves scattered chunks.

**Animation 3 (synthetic data) is DROPPED.** Do not build it. (The old code has `Animation3_SyntheticData.tsx` / `animation3.ts` / `Scene07SyntheticData.tsx` — ignore them.)

### Hard constraints (decided with the author)
- **On-screen language: Russian.** All labels, captions, document text, questions, answers in Russian.
- **Example content: keep exactly the existing construction / real-estate example.** Do NOT invent a new domain. The story is the residential complex (жилой комплекс) with a monitoring system, engineering networks, and operational risks. Exact data is in §6.
- **Audience: executive.** Modern, minimal, calm motion. No academic clutter, no cartoon style. Clear visual hierarchy. Each animation must read in 15–30s.
- **Deterministic, hardcoded data. No live APIs, no LLM calls, no keys.**

> Note: the author of the Wikontic paper (Aydar Bulatov) is the user. Keep the algorithm depiction faithful; when in doubt about method details, ask rather than guess.

---

## 2. The stack (verified)

| Layer | Tech |
|---|---|
| Video composition / timeline / render | **Remotion 4** (`remotion`, `@remotion/cli`, `@remotion/renderer`, `@remotion/transitions`) |
| UI / scene components | **React 19** |
| Types | **TypeScript 5.7** (`npm run check` = `tsc --noEmit`, currently passes) |
| Graphics | **SVG + CSS** (graph nodes/edges, scanners, highlights, cards) — see `components/GraphView.tsx` |
| Icons | **lucide-react** |
| Optional interactive page | **Vite 7** + `@vitejs/plugin-react` (`src/main.tsx`, `src/interactive/GraphDemo.tsx`) — not part of the rendered videos |

**Video format:** 1920×1080, **30 fps**. See `src/theme.ts` (`VIDEO_WIDTH`, `VIDEO_HEIGHT`, `FPS`, `DURATION_IN_FRAMES`, `colors`, `font`, `sceneFrames`).

### Remotion patterns used here
- `src/index.ts` → `registerRoot(RemotionRoot)`.
- `src/Root.tsx` declares `<Composition id=... component=... durationInFrames fps width height />`. **Currently there is only ONE composition: `WikonticHero`.** The render scripts in `package.json` reference `TextToGraph` / `RagVsWikontic` / `SyntheticDataFactory`, which **do not exist** in the current `Root.tsx` — those scripts will fail until we register matching compositions. When we build the new animations, register **two** compositions (suggested ids: `Method` / `RagVsWikontic`, or keep `TextToGraph` / `RagVsWikontic`) and fix `package.json` accordingly.
- Animation primitives: `useCurrentFrame()`, `interpolate(frame, [inFrames], [outValues], {extrapolateLeft/Right: 'clamp'})`, `spring()`, `<Sequence from= durationInFrames=>`, `<AbsoluteFill>`, `staticFile('assets/...')`, `<Img>`.
- Scenes are time-sliced with `<Sequence>` and cross-faded (see `HeroVideo.tsx` `FadeScene`).
- All timing is **frame-based** (frame = second × 30). Keep a single source of truth for scene timings like the existing `sceneFrames` map.

### Environment (already set up)
- Node 26, npm 11, Git, system FFmpeg 8.1 all present.
- `npm install` done (0 vulnerabilities). Remotion's bundled Chrome Headless Shell is downloaded.

### Commands
```bash
npm run preview              # Remotion Studio at http://localhost:3000
npm run check                # tsc --noEmit (must stay green)
npm run render:text-to-graph # render one composition to out/*.mp4 (h264, yuv420p)
npm run render:rag-vs-wikontic
npm run render:all
npm run graph:preview        # optional Vite page at http://localhost:5173 (not the videos)
```
Rendered MP4s land in `out/`.

---

## 3. THE ALGORITHM (Wikontic) — get this right

Paper: **“Wikontic: Constructing Wikidata-Aligned, Ontology-Aware Knowledge Graphs with Large Language Models”** (Chepurova, Bulatov, Burtsev, Kuratov). arXiv:2512.00590.

**One-line:** an LLM-driven, ontology-guided pipeline that turns text into a **compact, Wikidata-aligned, de-duplicated, ontology-consistent** knowledge graph — and then reasons over the *triplets alone* (no source text) for multi-hop QA.

### Pipeline = 3 stages (this is Figure 1)

**Stage 1 — Candidate triplet extraction (with qualifiers).**
- An LLM reads a text paragraph and emits candidate triplets as JSON. Each triplet has:
  `subject`, `relation`, `object`, `subject_type`, `object_type`, `qualifiers`.
- `relation` is a **Wikidata-style predicate**. `subject_type`/`object_type` are candidate classes.
- **Qualifiers** are contextual metadata attached to a triplet (e.g. time, location), each itself a `{relation, object}` pair. *Qualifiers are always attached to a triplet, never standalone.*
  - Canonical example: “In 2010, Christopher Nolan directed Inception” → `(Nolan, directed, Inception)` + qualifier `{point in time: 2010}`.
- Output at this stage is **raw / “dirty”**: relation wording is inconsistent, types are guesses, entities are surface forms.

**Stage 2 — Ontology-aware refinement (Wikidata constraints).** Three sub-steps:
1. **Entity typing.** For each subject and object, retrieve **top-10 candidate types** from a dense retrieval index over Wikidata types; the LLM picks the most plausible one. Supertypes are added by **recursive taxonomy expansion** along Wikidata `instance of` (P31) and `subclass of` (P279).
2. **Relation validation.** Using Wikidata constraints, enumerate **all relations that can legally connect the chosen subject/object types** (including inverse directions). Candidate relations are ranked by **cosine similarity** to the extracted relation. (Domain/range constraints define which classes a property may connect.)
3. **Triplet backbone reconstruction.** The LLM is given the text + the triplet + the valid relation set, and **selects the most plausible ontology-valid configuration** → a refined triplet “backbone”.
- **Final verification per triplet:** (i) subject type + object type + relation are all defined in the ontology, and (ii) the relation’s **domain & range** constraints hold. Triplets that fail are **flagged as “ontology-misaligned” but RETAINED** (so an alignment score can be computed) — they are *not* silently deleted.
- Backing data: a custom ontology DB of **2,464 factual Wikidata properties** with subject/object type constraints; **Contriever** embeddings in **MongoDB** for dense retrieval of entity/relation candidates.

**Stage 3 — Entity normalization & alias-aware deduplication.**
- For each refined triplet, link subject/object surface forms to **existing KG entities of the same/compatible parent type**.
- Using precomputed **alias embeddings**, retrieve **top-10 candidates**, rank by cosine similarity to the mention; the LLM decides **match vs. new**.
- On match: replace mention with the **canonical KG label**, store the surface form as an **alias**. Otherwise: keep a **new entity**, add its surface form to the alias collection.
- Net effect: duplicate surface forms (e.g. “ЖК” / “жилой комплекс” / “объект”) collapse into one canonical node with aliases. Supports **incremental** graph updates.

**Result:** a graph that is *compact, ontology-consistent, well-connected, de-duplicated, ready for downstream tasks.*

### Multi-hop QA over the graph (for Animation 2)
- **Iterative subquestion decomposition.** Given a question, the LLM:
  1. forms the **first 1-hop subquestion**;
  2. identifies relevant entities, **links them to KG nodes**, selects the most relevant;
  3. given the **subgraph from those nodes’ neighborhoods**, generates an answer;
  4. conditioned on that answer, forms the **next subquestion**.
- Repeats for **up to 5 subquestions**, then produces the final answer.
- **Triplets-only setting:** the LLM sees **only KG triplets, not the original text.** This is the key contrast with retrieval baselines (HippoRAG, AriGraph, GraphRAG) that still feed text chunks.

### Headline numbers (use sparingly, only where they land for executives)
- **HotpotQA:** 64.5 EM / **76.0 F1** (gpt-4.1, triplets-only).
- **MuSiQue:** 46.8 EM / **59.8 F1** (gpt-4.1, triplets-only) — matches/surpasses several RAG baselines.
- **Answer coverage (MuSiQue):** the correct answer entity appears in **96%** of generated triplets (96.2% in full KG; 68.8% within 10-hop neighborhood).
- **Efficiency (tokens per paragraph):** Wikontic **≈881 completion tokens** vs AriGraph ≈2,500 (**3×** more) vs GraphRAG ≈20,000 (**~22×** more, “<1/20”). Prompt tokens: 12.7k vs 11k vs 115k.
- **MINE-1 retention:** **86%** (gpt-4.1-mini) vs GraphRAG 47.8% vs KGGen 66%.

---

## 4. Algorithm → visual mapping

The demo simplifies the algorithm. Keep these mappings **faithful in spirit**; do not over-claim.

| Stage | What the animation shows | Faithful? |
|---|---|---|
| Document | Russian corporate document, factual spans highlighted one by one | ✅ |
| Stage 1 | Highlighted facts flow into **candidate triplet cards** `subject → relation → object` (+ a qualifier like `дата: 2024`) | ✅ (qualifiers are real; show at least one) |
| Stage 2 | A **scanner/checker** passes over cards; entity **types** appear (e.g. «ЖК» → «строительный объект»); relation variants («содержит» / «имеет в составе») **canonicalize** to «включает»; ✓ verified badges | ✅ — this is entity-typing + relation-validation + backbone reconstruction, compressed visually |
| Stage 2 (nuance, optional) | An off-ontology triplet shown **flagged amber, but kept** (not deleted) | ✅ matches “retained, not dropped”. Optional; only if it doesn’t confuse executives |
| Stage 3 | Duplicate nodes («ЖК», «жилой комплекс», «объект») **slide together and merge** into «жилой комплекс»; aliases shown as small tags | ✅ |
| Final graph | Compact, stable, readable graph; document fades back; optional metric badge | ✅ |
| Anim 2 — QA | Natural question → graph path lights up **edge by edge** → grounded answer | ✅ = iterative hop traversal, shown as a single highlighted path |
| Anim 2 — RAG side | 3–5 retrieved **chunks**, relevant but **fragmented**; the relation chain is implicit (dashed/“assemble it yourself”) | ✅ in spirit; see fairness rule below |

**Simplifications to be aware of (don’t claim otherwise):** real entity typing/relation validation use top-10 dense retrieval + cosine ranking + an LLM choice — the demo collapses this into a clean “scan → type → canonical relation” beat. The real QA loop is iterative subquestion decomposition (≤5 hops); the demo shows the resulting path as one continuous highlight.

---

## 5. Story & scene breakdown

### Communication rules (critical)
- **User-facing questions must sound like normal business questions** — never expose graph structure. The graph **path is Wikontic’s internal explanation**, shown visually, not as the question wording.
  - ❌ “Какая сущность связана с X через Y?”, “Что находится на 2-hop пути?”
  - ✅ “Что важно проверить перед вводом жилого комплекса в эксплуатацию?”
- **Do NOT make RAG look stupid.** The claim is narrow: *for multi-hop questions, explicit graph structure is more controllable than raw retrieved chunks.* Prefer “fragmented / less controllable” over “hard failure”.

### Animation 1 — Method (target 20–30s)
Beats: **A1.1** document close-up, highlight factual spans one by one → **A1.2** non-essential text fades, only facts remain → **A1.3** facts flow into **candidate triplet cards** (label «1. Кандидатные триплеты») → **A1.4** ontology scan: types + canonical relations + ✓ badges («2. Проверка на онтологию / верификация») → **A1.5** duplicate nodes merge with alias tags («3. Очистка и дедупликация») → **A1.6** final compact graph («Вся информация представлена компактно в графе»), optional metric badge.

### Animation 2 — Wikontic vs RAG (target 15–25s)
Beats: **A2.1** natural complex question («Сложный вопрос на несколько фактов») → **A2.2** left panel “RAG”: 3–5 fragmented chunks, dashed links, note «Связь между фактами нужно собрать заново» → **A2.3** right panel “Wikontic”: same question maps to nodes, path lights up step by step («В графе находим путь к ответу») → **A2.4** answer card + conclusion «RAG ищет фрагменты. Wikontic ищет путь между фактами.»

---

## 6. Canonical demo data (Russian — keep exactly)

Lives in `src/data/animation1.ts` and `animation2.ts`. **Reuse this content; only the presentation is being rebuilt.**

**Document (3 lines):**
> В 2024 году проектная команда **утвердила требования к жилому комплексу**.
> Объект включает **три корпуса**, **подземный паркинг** и **систему мониторинга инженерных сетей**.
> Система мониторинга используется для **контроля эксплуатационных рисков**.

**Highlighted facts:** требования утверждены в 2024 году · объект включает три корпуса · объект включает подземный паркинг · система мониторинга инженерных сетей · мониторинг контролирует эксплуатационные риски.

**Candidate triplets:**
- жилой комплекс → включает → три корпуса
- жилой комплекс → включает → подземный паркинг
- жилой комплекс → включает → система мониторинга
- система мониторинга → контролирует → инженерные сети
- утверждение требований → дата → 2024  *(qualifier: контекст: проект)*

**Ontology checks (Stage 2):** «ЖК» → тип «строительный объект» · «инженерные сети» → «инженерная система» · «содержит / имеет в составе» → канонично «включает» · «контроль рисков» → допустимая связь.

**Dedup groups (Stage 3):** {ЖК, жилой комплекс, объект} → **жилой комплекс**; {сети, инженерные сети} → **инженерные сети**.

**Anim 2 question:** «Что важно проверить перед вводом жилого комплекса в эксплуатацию?»
**RAG chunks:** «Объект включает систему мониторинга.» · «Система мониторинга контролирует инженерные сети.» · «Инженерные сети связаны с эксплуатационными рисками.» · «Требования к объекту утверждены проектной командой в 2024 году.»
**Graph path (internal):** жилой комплекс → включает → система мониторинга → контролирует → инженерные сети → связаны с → эксплуатационные риски.
**Answer:** «Нужно проверить систему мониторинга инженерных сетей, потому что она связана с эксплуатационным контролем объекта.»

Graph node/edge geometry (normalized x/y, kind, type) is already defined in `animation1.ts` (`compactGraphNodes`, `compactGraphEdges`) — good starting layout.

---

## 7. Visual style

Executive/minimal. Smooth, calm motion. Readable Russian (Inter / system sans, see `theme.ts`).

Color semantics (from `theme.ts` palette):
- Source document: neutral gray / white.
- Highlighted facts: amber or light blue.
- Candidate triplets: light blue (`blue`/`blueSoft`).
- Ontology verification: blue / green; **warnings / ambiguity / off-ontology: amber**; invalid: red.
- Final graph: stable blue / green.
- Node kinds already color-coded in `GraphView.tsx` (`project`/`asset`/`system`/`requirement`/`time`/`document`).

---

## 8. Reuse vs rebuild (the existing code is INSPO only)

**Worth mining for ideas / data:**
- `src/theme.ts` — palette, fps, frame helpers.
- `src/data/animation1.ts`, `animation2.ts` — keep the data (see §6).
- `src/components/GraphView.tsx` — SVG graph with node kinds, edge labels, `softGlow` filter, arrow markers, scanner rect, **merge animation via `mergeProgress`**, path highlight via `highlightedNodeIds`/`highlightedEdgeIds`. Solid reference.
- Components: `HighlightedFact`, `TripletCard`, `OntologyScanner`/`OntologyPass`, `DedupMerge`, `PathHighlight`, `MetricBadge`, `SceneLayout`.

**Ignore / delete-later:** anything synthetic-data (`Animation3_*`, `Scene07*`, `animation3.ts`, `syntheticCards.ts`, `DatasetCard.tsx`), and the 7-scene `HeroVideo` stitching — we want **two standalone compositions**, not one stitched hero video.

**When building the new version:**
1. Register two `<Composition>`s in `Root.tsx`; align `package.json` render script ids to them.
2. Keep `npm run check` green.
3. Keep all data deterministic and in `src/data/`.
4. Verify in `npm run preview` (Studio) before rendering.
