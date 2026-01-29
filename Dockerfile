# 1. Start with Python 3.9 slim
FROM python:3.9-slim

# 2. Install System Dependencies (Tesseract + OpenGL)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Setup U-2-Net model path
ENV U2NET_HOME=/app/.u2net

# 4. Install Python Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code
COPY . .

# 6. Start the server (Cloud-Ready Command)
# If $PORT is missing (running locally), it defaults to 8080.
# 6. Start the server
# We use ["/bin/sh", "-c", "..."] to safely use variables like ${PORT}
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]