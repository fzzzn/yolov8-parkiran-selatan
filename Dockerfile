# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
# Use CPU-only torch to reduce image size and build time (no GPU needed for inference)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY config.json .
COPY telegram_bot.py .

# Pre-download YOLOv8 model during build to avoid downloading on every start
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Create directory for temporary files
RUN mkdir -p /tmp

# Set environment variables (will be overridden by .env or docker-compose)
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]
