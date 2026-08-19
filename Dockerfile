FROM python:3.11-slim

# Install system dependencies for OpenCV
# Note: libgl1-mesa-glx was renamed to libgl1 in newer Debian releases (Trixie+)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgstreamer1.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
