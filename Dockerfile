# Dockerfile for Brain Tumor YOLO Detection
# Phase 7: Production Docker image

FROM pytorch/pytorch:2.0.1-cuda12.1-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY data.yaml .

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD python -c "from ultralytics import YOLO; print('OK')" || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
