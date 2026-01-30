# Imagen base compatible con tu código CUDA
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar compiladores, herramientas y librerías (OpenCV + Python)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-12 g++-12 \
    gdb cmake git pkg-config wget \
    libopencv-dev \
    libssl-dev \
    python3 python3-pip python3-dev \
    libexif-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar dependencias de Python
 COPY python/requirements.txt /tmp/pip-tmp/
 RUN pip3 install --no-cache-dir -r /tmp/pip-tmp/requirements.txt

# 3. Configurar GCC 12 como el compilador por defecto
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

WORKDIR /workspace
