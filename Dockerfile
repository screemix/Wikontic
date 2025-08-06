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

COPY utils/openai_utils.py ./utils/
COPY utils/structured_dynamic_index_utils_with_db.py ./utils/
COPY utils/structured_inference_with_db.py ./utils/
COPY utils/prompts/ ./utils/prompts/

COPY Wikontic.py .
COPY pages/ ./pages/

COPY media/ ./media/

EXPOSE 8501

CMD ["streamlit", "run", "Wikontic.py", "--server.port=8501", "--server.address=0.0.0.0"]
