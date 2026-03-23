FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    libeigen3-dev \
    libglew-dev \
    libepoxy-dev \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libx11-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    python3-dev \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Add the NVIDIA repository keys and sources for CUDA and TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update

# Install TensorRT 10 and cuDNN 9
RUN apt-get install -y --no-install-recommends \
    tensorrt-dev \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && ldconfig

# Build Pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin && \
    cd /opt/Pangolin && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_PANGOLIN_VIDEO=OFF && \
    make -j$(nproc) && \
    make install

# Install ONNX Runtime 1.24.4 (GPU version)
RUN cd /tmp && wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-gpu-1.24.4.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu-1.24.4.tgz -C /opt \
    && ln -s /opt/onnxruntime-linux-x64-gpu-1.24.4/lib /opt/onnxruntime-linux-x64-gpu-1.24.4/lib64 \
    && rm onnxruntime-linux-x64-gpu-1.24.4.tgz \
    && ldconfig

WORKDIR /workspace
ENV GALLIUM_DRIVER=llvmpipe