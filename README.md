![Wikontic logo](/media/wikotic-wo-text.png)

# Wikontic

**Build ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs**

---

## 🚀 Overview

Knowledge Graphs (KGs) provide structured, verifiable representations of knowledge, enabling fact grounding and empowering large language models (LLMs) with up-to-date, real-world information. However, creating high-quality KGs from open-domain text is challenging due to issues like redundancy, inconsistency, and lack of alignment with formal ontologies.

**Wikontic** is a multi-stage pipeline for constructing ontology-aligned KGs from unstructured text using LLMs and Wikidata. It extracts candidate triples from raw text, then refines them through ontology-based typing, schema validation, and entity deduplication—resulting in compact, semantically coherent graphs.

![Pipeline overview](/media/KG+LM-pipeline-with-background.png)

---

## 📁 Repository Structure

- `preprocessing/constraint-preprocessing.ipynb`  
  Jupyter notebook for collecting constraint rules from Wikidata.

- `utils/`  
  Utilities for LLM-based triple extraction and alignment with Wikidata ontology rules.


- `utils/openai_utils.py`  
  `LLMTripletExtractor` class for LLM-based triple extraction.


### To use ontology:

- `utils/ontology_mappings/`  
  JSON files containing ontology mappings from Wikidata.

- `utils/structured_inference_with_db.py`  
  - `StructuredInferenceWithDB` class: triple extraction and qa functions

- `utils/structured_aligner.py`
  -  `Aligner` class: ontology alignment and entity name refinement


### Not to use ontology:
- `utils/inference_with_db.py`
  - `InferenceWithDB` class: triple extraction and qa functions

- `utils/dynamic_aligner.py`
  -  `Aligner` class: entity and relation name refinement

### Evaluation:
- `inference_and_eval/`
	- Scripts for building KGs for MuSiQue and HotPot datasets and evaluation of QA performance
- `analysis/`
  - Notebooks with downstream analysis of the resulted KG

### Use Wikontic as a service:

- `pages/` and `Wikontic.py`  
  Code for the web service for knowledge graph extraction and visualization.

- `Dockerfile`  
  For building a containerized web service.

![Demo overview](/media/demo-with-background.png)

---

## 🏁 Getting Started

1. **Set up the ontology and KG databases:**
   ```
   ./setup_db.sh
   ```

2. **Launch the web service:**
   ```
   streamlit run Wikontic.py
   ```

---

Enjoy building knowledge graphs with Wikontic!
