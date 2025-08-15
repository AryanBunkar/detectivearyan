# Base image with Python and build tools
FROM python:3.11-slim

# Install system dependencies for dlib & face-recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libssl-dev \
    libopenmpi-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
    dlib==19.24.4 \
    face_recognition==1.3.0 \
    face_recognition_models==0.3.0 \
    numpy==1.26.4 \
    opencv-contrib-python==4.11.0.86 \
    opencv-python==4.11.0.86 \
    pandas==2.3.1 \
    streamlit==1.48.1 \
    streamlit-webrtc==0.63.4

# Copy your app into the container
WORKDIR /app
COPY . /app

# Streamlit specific settings
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
