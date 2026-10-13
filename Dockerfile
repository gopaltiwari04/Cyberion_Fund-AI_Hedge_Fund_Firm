# Airflow orchestrator image.
#
# Airflow 2.x pins SQLAlchemy 1.4, while the pipelines use SQLAlchemy 2 /
# pandas 2.3 / torch. The pipelines therefore run in an isolated virtualenv
# (/opt/pipeline-venv) invoked by the DAGs through $PIPELINE_PYTHON, so the
# orchestrator's own dependencies are never modified.
FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/pipeline-venv && chown -R airflow: /opt/pipeline-venv

USER airflow
COPY --chown=airflow:root requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN /opt/pipeline-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/pipeline-venv/bin/pip install --no-cache-dir -r /tmp/requirements-pipeline.txt

ENV PIPELINE_PYTHON=/opt/pipeline-venv/bin/python \
    PROJECT_DIR=/opt/airflow/dags/project \
    HF_HOME=/opt/airflow/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/opt/airflow/.cache/fastembed

# Model caches (FinBERT, fastembed) live on a named volume; make sure the airflow
# user owns the mount point so a fresh volume inherits writable permissions.
USER root
RUN mkdir -p /opt/airflow/.cache/huggingface /opt/airflow/.cache/fastembed && chown -R airflow:root /opt/airflow/.cache
USER airflow
