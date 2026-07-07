FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app/demo_app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY demo_app/ ./demo_app/
COPY scripts/inject_streamlit_head.py ./scripts/inject_streamlit_head.py
COPY .streamlit/ ./.streamlit/
COPY media/ ./media/

RUN python scripts/inject_streamlit_head.py

EXPOSE 8501

CMD ["streamlit", "run", "demo_app/Wikontic.py", "--server.port=8501", "--server.address=0.0.0.0"]
