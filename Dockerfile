# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables preventing Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for compiling certain packages like FAISS if using source, though faiss-cpu has wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run as a non-root user for safer defaults in shared environments
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port for FastAPI
EXPOSE 8000

# Healthcheck aligned with FastAPI liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Run the API server using Uvicorn
CMD ["uvicorn", "autonomous_sre.interfaces.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
