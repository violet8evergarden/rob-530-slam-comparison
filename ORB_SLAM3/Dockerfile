FROM ubuntu:22.04

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

# Build Pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin && \
    cd /opt/Pangolin && \
    git checkout v0.6 && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_PANGOLIN_VIDEO=OFF && \
    make -j$(nproc) && \
    make install

WORKDIR /workspace
ENV GALLIUM_DRIVER=llvmpipe