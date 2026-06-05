FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/wikontic/utils/ ./src/wikontic/utils/
COPY src/wikontic/__init__.py ./src/wikontic/__init__.py
COPY src/wikontic/create_ontological_triplets_db.py ./src/wikontic/create_ontological_triplets_db.py
COPY src/wikontic/create_triplets_db.py ./src/wikontic/create_triplets_db.py
COPY src/wikontic/create_wikidata_ontology_db.py ./src/wikontic/create_wikidata_ontology_db.py

COPY Wikontic.py .
COPY streamlit_ui.py .
COPY pages/ ./pages/

COPY media/ ./media/

EXPOSE 8501

CMD ["streamlit", "run", "Wikontic.py", "--server.port=8501", "--server.address=0.0.0.0"]
