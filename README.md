![Wikontic logo](/media/wikontic.png)

# Wikontic - building ontology-aware, Wikidata-aligned knowledge graphs from raw text with LLMs

## Overview 

Knowledge Graphs (KGs) offer accurate structured knowledge representation, enabling verifiable fact grounding and providing large language models (LLMs) with the latest real-world information. However, constructing high-quality KGs from open-domain text remains challenging due to redundancy, inconsistency, and lack of grounding in formal ontologies.

Wikontic, a tool with a multi-stage pipeline for building ontology-aligned KGs from unstructured text using LLMs and Wikidata. Wikontic extracts candidate triples from raw text, then refines them through ontology-based typing, schema validation, and entity deduplication, producing compact, semantically coherent graphs. 

## Repository structure

```preprocessing/constraint-preprocessing.ipynb``` - a notebook with a code for downloading constraints rules from Wikidata

```utils/``` - contains utilities for LLM-based triplet extraction and triplet alignment with Wikidata ontology rules

```utils/ontology_mappings``` - contains json files with ontology mappings from Wikidata

```utils/structured_dynamic_index_utils_with_db.py ``` - contains Aligner class for ontology aligner

```utils/structured_dynamic_index_utils_with_db.py ``` - contains LLMTripletExtractor class for triplet extraction 

```pages/``` and ```Wikontic.py``` - contains a code for a web-service for Knowledge Graph extraction and visualization


## How to launch

```
./setup_db.sh
streamlit run Wikontic.py 
```
