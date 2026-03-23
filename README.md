# To D or Not to D. That is the SLAM Question: A Comparative Analysis of Monocular SLAM

This repository evaluates the performance and accuracy trade-offs between a Transformer-based Monocular SLAM system (**DropD-SLAM**) and a high-performance LiDAR-based LIO system (**FAST-LIO**).

## Repository Structure
- `/dropd-slam`: Source code and Dockerfile for Vision-based SLAM.
- `/fast-lio-ros2`: ROS 2 workspace for LiDAR-based SLAM.
- `/data`: External dataset mount (KITTI + WeatherKITTI).
- `/results`: Trajectory files (.txt) and `evo` comparison plots.

## Hardware Requirements
- **GPU:** NVIDIA RTX GPU.
- **Host OS:** Ubuntu 22.04 or WSL2.

## Setup Instructions

### 1. Prerequisites
1. Install `git-lfs`:
```bash
sudo apt update && sudo apt install git-lfs
git lfs install
```

2. Clone this repository:
```
git clone https://github.com/tannermi/rob-530-slam-comparison.git
```

3. Install Docker (https://docs.docker.com/engine/install/ubuntu/):
```bash
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

# Install the Docker packages:
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

4. Install the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit):
```bash
# Install the prerequisites:
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   ca-certificates \
   curl \
   gnupg2

# Configure the production repository:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list from the repository:
sudo apt-get update

# Install the NVIDIA Container Toolkit packages:
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

5. Configure Docker (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration):
```bash
# Configure the container runtime by using the nvidia-ctk command:
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon:
sudo systemctl restart docker
```

### 2. Build and Run
Use Docker Compose to build and launch the specific environment you wish to evaluate:
```bash
# Build both images
docker-compose build

# Launch DropD-SLAM (Vision)
docker-compose run dropd-slam

# Launch FAST-LIO (LiDAR)
docker-compose run fast-lio
```

### 3. Compilation (inside containers)
Once inside the `dropd-slam` container, compile the project:
```bash
./build.sh
```

### 4. Run (inside containers)
Once inside the `dropd-slam` container, run the code:
```bash
# Run DropD-SLAM with monocular depth estimation
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml /data/rgbd_dataset_freiburg1_desk2 /data/rgbd_dataset_freiburg1_desk2/associations.txt unidepth /workspace/models/unidepthv2_with_cam.onnx
```

## Evaluation Metrics
We use the `evo` package to analyze the Absolute Pose Error (APE) and Relative Pose Error (RPE).

### Comparison Commands
To evaluate the Vision SLAM against Ground Truth:
```bash
evo_ape tum data/ground_truth.txt results/dropd_traj.txt -va --plot
```

To compare Vision SLAM directly against LiDAR SLAM:
```bash
evo_ape tum results/fast_lio_traj.txt results/dropd_traj.txt -va --plot
```
