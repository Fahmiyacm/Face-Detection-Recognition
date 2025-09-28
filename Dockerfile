FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies first to use Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
