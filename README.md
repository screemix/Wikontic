![Wikontic logo](/media/wikotic-wo-text.png)

# Wikontic

**Build ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs**

Paper: [arXiv:2512.00590](https://arxiv.org/abs/2512.00590) · Tutorial: [`tutorial.ipynb`](tutorial.ipynb)

---

## Overview

Knowledge graphs (KGs) provide structured, verifiable representations of knowledge. Building them from open-domain text is hard: extracted facts are often redundant, inconsistent, and not aligned with a formal schema.

**Wikontic** is a multi-stage pipeline that:

1. **Extracts** candidate `(subject, relation, object)` triplets from text with an LLM (optionally with qualifiers and entity types).
2. **Refines** them via embedding-based entity/relation linking and LLM reranking.
3. **Validates** triplets against a Wikidata-derived ontology (structured mode).
4. **Stores** results in a vector database for retrieval, QA, and visualization.

Two inference modes are supported:

| Mode | Class | Aligner | Ontology | Best for |
|------|--------|---------|----------|----------|
| **Structured** (default in research scripts) | `StructuredInferenceWithDB` | `structured_aligner.Aligner` | Wikidata types & property constraints | Wikidata-aligned KGs, QA benchmarks |
| **Dynamic** | `InferenceWithDB` | `dynamic_aligner.Aligner` | None (learned aliases only) | Open-domain graphs without ontology |

![Pipeline overview](/media/KG+LM-pipeline-with-background.png)

Triplet stages written to the database:

| Collection | Description |
|------------|-------------|
| `initial_triplets` | Raw LLM extraction before refinement |
| `triplets` | Final accepted triplets after refinement / deduplication |
| `filtered_triplets` | Triplets removed during refinement (e.g. invalid names) |
| `ontology_filtered_triplets` | Structured mode only — triplets that violate ontology constraints |

![Demo overview](/media/demo-with-background.png)

---

## Repository structure

```
Wikontic/
├── src/wikontic/              # Main Python package
│   ├── create_wikidata_ontology_db.py   # Populate Wikidata ontology DB
│   ├── create_ontological_triplets_db.py # KG DB schema (structured mode)
│   ├── create_triplets_db.py            # KG DB schema (dynamic mode)
│   ├── db/                    # Storage backends (MongoDB, Qdrant)
│   └── utils/
│       ├── openai_utils.py              # LLMTripletExtractor
│       ├── dynamic_aligner.py           # Entity/relation linking (no ontology)
│       ├── structured_aligner.py        # Ontology-aware alignment
│       ├── inference_with_db.py         # Dynamic extraction + QA
│       ├── structured_inference_with_db.py
│       ├── base_inference_with_db.py    # Shared QA logic
│       ├── ontology_mappings/           # Wikidata JSON mappings
│       ├── ontology_mappings_en_en/     # English ontology variant
│       ├── ontology_mappings_ru_en/     # Russian mappings
│       └── prompts/ / prompts_ru/       # LLM prompt templates
├── pages/                     # Streamlit multipage app
├── inference_and_eval/        # KG construction & QA evaluation
├── analysis/                  # KG dump, stats, visualization helpers
├── preprocessing/             # Dataset preprocessing scripts
├── tests/                     # Pytest suite (Mongo + Qdrant)
├── Wikontic.py                # Streamlit home page
├── tutorial.ipynb             # LangChain integration example
├── conftest.py                # Shared test fixtures
├── requirements.txt
├── pyproject.toml
├── Dockerfile                 # Dockerfile for Wikontic demo 
└── setup_mongo_db.sh          # Quick MongoDB Atlas Local bootstrap
```

---

## Requirements

- Python **≥ 3.9**
- **MongoDB Atlas Local** (recommended for production/demo vector search) **or** **Qdrant** (`:memory:` or remote)
- OpenAI-compatible API access (OpenAI, OpenRouter, local endpoint, etc.)
- GPU optional (embedding model `facebook/contriever` runs on CPU by default in tests)

---

## Installation

```bash
git clone https://github.com/screemix/Wikontic.git
cd Wikontic

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .           # install wikontic package from src/
```

For development / tests:

```bash
pip install pytest
```

---

## Environment variables

Create a `.env` file in the repository root (loaded automatically via `python-dotenv`):

| Variable | Purpose |
|----------|---------|
| `MONGO_URI` | MongoDB connection string (default: `mongodb://localhost:27018/?directConnection=true`) |
| `KEY` or `OPENROUTER_KEY` | API key for the LLM provider |
| `OPENROUTER_BASE_URL` | OpenAI-compatible API base URL (e.g. `https://openrouter.ai/api/v1`) |
| `PROXY_URL` | Optional HTTP proxy for API calls |
| `WIKONTIC_LOG_LEVEL` | Global log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |
| `OSS_URL` | Example custom base URL used in some configs |

Scripts read API settings from env; batch configs let you choose which env var names to use (`api_key_env_var`, `base_url_env_var`).

---

## Database setup

Wikontic uses a **storage backend abstraction** (`src/wikontic/db/`) with two implementations:

- **`mongodb`** — MongoDB with Atlas Vector Search indexes (local Atlas image or cloud)
- **`qdrant`** — Qdrant vector DB (`:memory:` for tests/ephemeral runs, or a remote URL)

### Option A — MongoDB Atlas Local

```bash
# Start MongoDB Atlas Local (see setup_mongo_db.sh)
docker pull mongodb/mongodb-atlas-local:latest
docker run --name text2kg_mongo -d -p 27018:27018 mongodb/mongodb-atlas-local:latest
```

Initialize databases from the repo root (requires `pip install -e .`):

```bash
# 1. Wikidata ontology (required for structured inference)
python -m wikontic.create_wikidata_ontology_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --database wikidata_ontology

# 2a. Structured KG database (with ontology_filtered_triplets)
python -m wikontic.create_ontological_triplets_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --db_name triplets_db

# 2b. OR dynamic KG database (no ontology collection)
python -m wikontic.create_triplets_db \
  --backend mongodb \
  --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
  --db_name triplets_db
```

Or use the helper script:

```bash
./setup_mongo_db.sh
```

### Option B — Qdrant in-memory (tests & batch jobs without Mongo)

```bash
python -m wikontic.create_wikidata_ontology_db --backend qdrant --qdrant_url :memory:
python -m wikontic.create_ontological_triplets_db --backend qdrant --qdrant_url :memory:
```

> **Note:** Qdrant `:memory:` data is lost when the process exits. Inference script automatically dumps the KG to JSON in that case (see [KG dump](#kg-dump)).

### Database CLI arguments

All three `create_*` scripts support:

| Argument | Description |
|----------|-------------|
| `--backend` | `mongodb` or `qdrant` |
| `--mongo_uri` | MongoDB URI (mongodb backend) |
| `--qdrant_url` | Qdrant URL or `:memory:` |
| `--qdrant_api_key` | Optional Qdrant API key |
| `--database` / `--db_name` | Database / collection namespace name |
| `--drop_collections` | Drop and recreate collections |
| `--embedding_dimensions` | Vector size (default `768`) |

See `--help` on each script for collection and index name overrides.

---

## Streamlit web app

Launch the interactive demo:

```bash
streamlit run Wikontic.py
```

Default URL: `http://localhost:8501`

### Pages

| Page | File | Description |
|------|------|-------------|
| Home | `Wikontic.py` | Overview and links |
| KG Extraction | `pages/1_KG_Extraction.py` | Extract triplets from text, visualize initial vs enriched graph |
| QA | `pages/2_QA.py` | Ask questions over the session KG |
| Current KG | `pages/3_Current_KG.py` | Browse triplets stored in the demo database |
| Personal KG | `pages/4_Personal_KG.py` | Build a personal knowledge graph |
| Wikipedia vs Wikidata | `pages/5_Wikipedia_vs_Wikidata.py` | Compare extraction variants |

The demo uses MongoDB databases `wikidata_ontology` and `demo` (or page-specific DB names). Ensure databases are initialized and `.env` contains `MONGO_URI` and `KEY`.

### Docker

```bash
docker build -t wikontic .
docker run -p 8501:8501 --env-file .env wikontic
```

---

## Batch KG construction

`inference_and_eval/dataset_inference.py` runs extraction over a JSON dataset and stores triplets in the configured backend.

### Run

From the repo root (after `pip install -e .`):

```bash
python inference_and_eval/dataset_inference.py \
  --config inference_and_eval/configs/musique_inference_with_db.yaml
```

Or from `inference_and_eval/` (config paths like `dataset_path` are relative to the working directory):

```bash
cd inference_and_eval
python dataset_inference.py --config configs/musique_inference_with_db.yaml
```

Or set `KG_CONSTRUCTION_CONFIG` to a YAML path.

### Config file (`configs/*.yaml`)

All keys below can be set in YAML. Unspecified keys use defaults from `CONFIG_DEFAULTS` in `dataset_inference.py`.

| Key | Default | Description |
|-----|---------|-------------|
| `mongo_uri` | `mongodb://localhost:27018/?directConnection=true` | MongoDB URI |
| `vector_db_backend` | `mongodb` | `mongodb` or `qdrant` |
| `qdrant_url` | `:memory:` | Qdrant URL; use `:memory:` for in-process |
| `qdrant_api_key` | `null` | Qdrant API key (remote) |
| `ontology_db_name` | `wikidata_ontology` | Ontology database name |
| `triplets_db_name` | `triplets_db` | Base name for triplets DB (suffixes added automatically) |
| `model_name` | `gpt-4o-mini` | LLM model id |
| `dataset_path` | `datasets/musique_200_test_preprocessed.json` | Input JSON dataset |
| `preprocessing` | `musique` | Dataset label (informational) |
| `sample_start_index` | `0` | Start index into dataset keys |
| `num_samples` | `50` | Number of samples to process |
| `structured_inference` | `true` | Use `StructuredInferenceWithDB` if true |
| `dump_kg` | `false` | Write `kg_dump/kg_dump_{db_name}.json` after the run (always on for Qdrant `:memory:`) |
| `api_key_env_var` | `KEY` | Env var for API key |
| `base_url_env_var` | `OPENROUTER_BASE_URL` | Env var for API base URL |
| `proxy_env_var` | `null` | Env var name for proxy URL |

The script derives the actual triplets database name:

```
{triplets_db_name}_{model_with_slashes_replaced}_{onto|non_onto}
```

Example: `triplets_db_gpt-4o-mini_onto`

### Dataset format

JSON object mapping `sample_id` → list of text passages:

```json
{
  "sample_1": ["First paragraph...", "Second paragraph..."],
  "sample_2": ["Another document..."]
}
```

Each passage is stored with `source_text_id` = its index in the list. Extraction is skipped if triplets already exist for `(sample_id, source_text_id)`.

### KG JSON dump

After inference, the script can export triplets to (always under the **repository root**, regardless of where you run the command from):

```
kg_dump/kg_dump_{triplets_db_name}.json
```

Example with default config names: `kg_dump/kg_dump_triplets_db_gpt-4o-mini_onto.json`

- **Qdrant `:memory:`** — dump runs automatically at the end of a successful run (in-memory data is not persisted elsewhere).
- **MongoDB or remote Qdrant** — set `dump_kg: true` in config to enable.

The dump path is printed to stdout when finished. Note: `kg_dump/` and `*.json` are in `.gitignore`, so the file exists on disk but won't appear in `git status`.

See [analysis/dump_kg.py](analysis/dump_kg.py) for the JSON schema.

---

## QA evaluation

After building KGs, evaluate question answering on MuSiQue or HotpotQA:

```bash
python inference_and_eval/qa_eval_musique.py \
  --triplets_db_name triplets_db_gpt-4o-mini_onto \
  --dataset_path datasets/musique_200_test.json \
  --structured_inference \
  --use_qualifiers
```

### `qa_eval_musique.py` / `qa_eval_hotpot.py` arguments

| Argument | Description |
|----------|-------------|
| `--mongo_uri` | MongoDB URI |
| `--ontology_db_name` | Ontology DB name |
| `--triplets_db_name` | Triplets DB with constructed KGs |
| `--model_name` | LLM for QA |
| `--dataset_path` | QA dataset JSON |
| `--structured_inference` / `--no_structured_inference` | Ontology-aware QA path |
| `--multi-step-qa` | Enable multi-step decomposition |
| `--use_qualifiers` / `--no_use_qualifiers` | Include qualifier nodes in retrieval |
| `--use_filtered_triplets` / `--no_use_filtered_triplets` | Include filtered triplets in context |
| `--run_number` | Run id for output files |

---

## KG dump

Export stored triplets to JSON for analysis or backup.

### From MongoDB

```bash
cd analysis
python dump_kg.py --db_name triplets_db_gpt-4o-mini_onto
```

### From any backend (Python)

```python
from wikontic.db.factory import create_backend
from dump_kg import dump_kg_from_backend

backend = create_backend("qdrant", qdrant_url=":memory:")
dump_kg_from_backend(backend, "my_run", include_ontology_filtered=True)
```

Output structure:

```json
{
  "sample_id": {
    "source_text_id": {
      "initial_triplets": [...],
      "triplets": [...],
      "ontology_filtered_triplets": [...],
      "filtered_triplets": [...]
    }
  }
}
```

Each triplet includes: `subject`, `relation`, `object`, `subject_type`, `object_type`, `qualifiers`, `sample_id`, `source_text_id`.

---

## Analysis utilities

| Script | Purpose |
|--------|---------|
| `analysis/dump_kg.py` | Export KGs from MongoDB or `StorageBackend` |
| `analysis/graph_analysis_stats.py` | Graph metrics (nodes, edges, clustering, components) |
| `analysis/visualize_knowledge_graph.py` | PyVis graph rendering from MongoDB |
| `analysis/wikidata_vs_wikipedia_utils.py` | Helpers for Wikipedia vs Wikidata comparison |

---

## Python API

Minimal structured extraction example:

```python
from pymongo import MongoClient
from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner
from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB

client = MongoClient("mongodb://localhost:27018/?directConnection=true")
ontology_db = client["wikidata_ontology"]
triplets_db = client["my_triplets_db"]

extractor = LLMTripletExtractor(model="gpt-4o-mini", api_key="...")
aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)
inference = StructuredInferenceWithDB(extractor, aligner, triplets_db)

initial, final, filtered, onto_filtered = (
    inference.extract_triplets_with_ontology_filtering_and_add_to_db(
        text="Paris is the capital of France.",
        sample_id="demo_1",
        source_text_id=0,
    )
)
```

For LangChain tool bindings, see [`tutorial.ipynb`](tutorial.ipynb).

### Key classes

| Module | Class | Role |
|--------|-------|------|
| `openai_utils` | `LLMTripletExtractor` | LLM triplet extraction & QA prompts |
| `dynamic_aligner` | `Aligner` | Embedding search over entity/property aliases |
| `structured_aligner` | `Aligner` | Wikidata type/property alignment |
| `inference_with_db` | `InferenceWithDB` | Dynamic pipeline + QA |
| `structured_inference_with_db` | `StructuredInferenceWithDB` | Ontology-aware pipeline + QA |
| `db.factory` | `create_backend` | Create MongoDB or Qdrant backend |

## Backend Selection (`mongodb` or `qdrant`)

`inference_and_eval/configs/*.yaml` now supports:

- `vector_db_backend`: `mongodb` or `qdrant`
- `qdrant_url`: set to `:memory:` for pure in-memory mode
- `qdrant_api_key`: optional (for remote Qdrant)

Example (in-memory Qdrant):

```yaml
vector_db_backend: "qdrant"
qdrant_url: ":memory:"
qdrant_api_key: null
```

---

## Tests

```bash
# Requires MongoDB on MONGO_URI and OPENROUTER_KEY (or KEY) in .env for LLM tests
pytest

# Subset examples
pytest tests/test_01_db_setup.py -k qdrant -v
pytest tests/test_02_backend.py -v
pytest tests/test_05_inference_pipeline.py -v   # calls live LLM API
```

| Test file | Coverage |
|-----------|----------|
| `test_01_db_setup.py` | DB collections, indexes (Mongo + Qdrant) |
| `test_02_backend.py` | Upsert, match (`$or`, `$and`, `$in`), vector search (both backends) |
| `test_03_dynamic_aligner.py` | Dynamic aligner CRUD & retrieval |
| `test_04_structured_aligner.py` | Structured aligner & ontology retrieval |
| `test_05_inference_pipeline.py` | End-to-end extraction + DB storage |

---

## Preprocessing

| Script | Purpose |
|--------|---------|
| `preprocessing/preprocess_dataset.py` | General dataset preprocessing |
| `preprocessing/constraint-preprocessing_batch.py` | Wikidata constraint collection |
| `preprocessing/edgar_chunking.py` | SEC EDGAR text chunking |

---

## Citation

If you use Wikontic in research, please cite the [arXiv paper](https://arxiv.org/abs/2512.00590).

---

## License

MIT — see [LICENSE](LICENSE).
