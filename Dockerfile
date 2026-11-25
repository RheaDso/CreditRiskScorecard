# Run the pipeline first: artifacts/ is copied into the image.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
COPY config ./config
COPY artifacts ./artifacts
ENV PYTHONPATH=/app/src
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "scorecard.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
