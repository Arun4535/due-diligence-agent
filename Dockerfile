FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for reportlab
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]