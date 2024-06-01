# Base image
# FROM python:3.8-slim
FROM arm64v8/python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_project/ mlops_project/
COPY data/ data/


# Create the models directory
RUN mkdir -p models
RUN mkdir -p reports/figures/


WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=~/.cache/pip pip install -r requirements.txt --no-cache-dir

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_project/train_model.py"]