FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libpcl-dev \
    libeigen3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    ros-humble-rviz2 \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-rosbag2-storage-mcap \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Build and Install Livox-SDK2
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 && \
    cd /tmp/Livox-SDK2 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/Livox-SDK2

# Setup Workspace
WORKDIR /workspace/fast_lio_ws

# Initialize rosdep
RUN if [ ! -d /etc/ros/rosdep/sources.list.d ]; then rosdep init; fi && rosdep update

# Environment setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "export PCL_ROOT=/usr/include/pcl-1.12" >> ~/.bashrc