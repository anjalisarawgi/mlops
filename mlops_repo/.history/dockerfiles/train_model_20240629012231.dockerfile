# Base image
FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY mlops_repo/requirements.txt mlops_repo/requirements.txt
COPY mlops_repo/pyproject.toml mlops_repo/pyproject.toml
COPY mlops_repo/mlops_project/ mlops_repo/mlops_project/
COPY mlops_repo/data/ mlops_repo/data/

WORKDIR /
RUN pip install -r mlops_repo/requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_project/train_model.py"]