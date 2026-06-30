# Wikontic Streamlit Demo

This folder contains the Streamlit demo application. The Wikontic engine stays in `../src/wikontic`; the demo imports it as the Python package `wikontic`.

## Folder Structure

```text
demo_app/
├── Wikontic.py                 # Streamlit entrypoint
├── streamlit_app_config.py     # Env config, repo/media paths, DB defaults
├── streamlit_session.py        # Shared Mongo, aligner, extractor, inference setup
├── streamlit_navigation.py     # Streamlit page registration
├── streamlit_i18n.py           # UI translations
├── streamlit_examples.py       # Backend-language example texts and prompts
├── streamlit_inference.py      # Structured/dynamic extraction wrapper
├── streamlit_ui.py             # Shared header/footer/sidebar helpers
├── streamlit_kg_viz.py         # PyVis graph rendering
├── streamlit_token_stats.py    # Token comparison helpers
└── app_pages/                  # Streamlit pages
```

The app resolves `media/` and `.env` from the repository root, so it can run from either the root folder or `demo_app/`.

## Run

From the repository root:

```bash
PYTHONPATH=src:demo_app streamlit run demo_app/Wikontic.py
```

From `demo_app/`:

```bash
PYTHONPATH=../src:. streamlit run Wikontic.py
```

Using the helper script from the repository root:

```bash
./run_streamlit.sh
```

Default URL: `http://localhost:8501`.

## Required Environment

Create `.env` in the repository root. The app reads this file regardless of the current working directory.

```bash
MONGO_URI=mongodb://localhost:27018/?directConnection=true
OPENROUTER_KEY=...        # or KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

For OpenAI directly, use `KEY` or `OPENROUTER_KEY` with the default OpenAI base URL. `PROXY_URL` is optional.

## Language And Mode Flags

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `WIKONTIC_FRONTEND_LANGUAGE` | `en`, `ru` | `en` | Streamlit interface language |
| `WIKONTIC_BACKEND_LANGUAGE` | `en`, `ru` | `en` | Wikontic prompts, examples, transliteration, and default DB names |
| `WIKONTIC_USE_ONTOLOGY` | `true`, `false` | `true` | Structured ontology mode vs dynamic mode |
| `WIKONTIC_MODEL` | model id | `gpt-4.1` | LLM model used by the demo |
| `WIKONTIC_TRIPLETS_DB_NAME` | database name | mode/language default | Override demo KG database |
| `WIKONTIC_ONTOLOGY_DB_NAME` | database name | language default | Override ontology DB in ontology mode |

Examples from the repository root:

```bash
# English UI, English backend, ontology on
PYTHONPATH=src:demo_app streamlit run demo_app/Wikontic.py

# Russian UI, Russian backend, ontology on
WIKONTIC_FRONTEND_LANGUAGE=ru \
WIKONTIC_BACKEND_LANGUAGE=ru \
WIKONTIC_USE_ONTOLOGY=true \
PYTHONPATH=src:demo_app streamlit run demo_app/Wikontic.py

# English UI, Russian backend, ontology off
WIKONTIC_FRONTEND_LANGUAGE=en \
WIKONTIC_BACKEND_LANGUAGE=ru \
WIKONTIC_USE_ONTOLOGY=false \
PYTHONPATH=src:demo_app streamlit run demo_app/Wikontic.py
```

## Database Defaults

| Backend language | Ontology mode | Ontology DB | Triplets DB |
|------------------|---------------|-------------|-------------|
| `en` | on | `wikidata_ontology` | `demo` |
| `ru` | on | `wikidata_ontology_ru` | `demo_ru` |
| `en` | off | not used | `demo_dynamic` |
| `ru` | off | not used | `demo_ru_dynamic` |

## Initialize Databases

Start MongoDB Atlas Local first, for example from the repository root:

```bash
docker pull mongodb/mongodb-atlas-local:latest
docker run --name text2kg_mongo -d -p 27018:27017 mongodb/mongodb-atlas-local:latest
```

English ontology mode:

```bash
python -m wikontic.create_wikidata_ontology_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --database wikidata_ontology \
  --language en

python -m wikontic.create_ontological_triplets_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --db_name demo
```

Russian ontology mode:

```bash
python -m wikontic.create_wikidata_ontology_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --database wikidata_ontology_ru \
  --language ru

python -m wikontic.create_ontological_triplets_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --db_name demo_ru
```

English dynamic mode:

```bash
python -m wikontic.create_triplets_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --db_name demo_dynamic
```

Russian dynamic mode:

```bash
python -m wikontic.create_triplets_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --db_name demo_ru_dynamic
```

If the selected database is missing required collections, the app shows the command needed for the active configuration.

## Pages

| Page | Description |
|------|-------------|
| Home | Overview and links |
| KG Extraction | Extract triplets from text and visualize the graph |
| QA | Ask questions over the session graph |
| Current KG | Browse and delete the current session graph |
| Personal KG | Build a graph from web-searched person information |

`Personal KG` requires the OpenAI Responses API with `web_search`. OpenRouter or other OpenAI-compatible endpoints may not support this API; in that case the page shows an error.

## Docker

From the repository root:

```bash
docker build -t wikontic .
docker run -p 8501:8501 --env-file .env wikontic
```

## Troubleshooting

- `ModuleNotFoundError: wikontic`: include `src` in `PYTHONPATH`, or install the package with `pip install -e .` from the repository root.
- `ModuleNotFoundError: streamlit_session`: include `demo_app` in `PYTHONPATH`, or run from `demo_app` with `PYTHONPATH=../src:.`.
- Missing collection errors: initialize the database for the selected language and ontology mode.
- Missing images: run from this repository checkout; assets are loaded from `../media`.
- Mongo connection errors: confirm Atlas Local is running and `MONGO_URI` points to the exposed host port.
