FROM python:3.9-slim

WORKDIR /app

# LightGBM requires libgomp (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy model artifacts and data
COPY model/ ./model/
COPY data/zipcode_demographics.csv ./data/zipcode_demographics.csv
COPY data/future_unseen_examples.csv ./data/future_unseen_examples.csv
COPY data/kc_house_data.csv ./data/kc_house_data.csv

# Run as non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]