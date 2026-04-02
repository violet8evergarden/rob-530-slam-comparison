# To D or Not to D. That is the SLAM Question: A Comparative Analysis of Visual and LiDAR SLAM

This repository evaluates the performance and accuracy trade-offs between a Transformer-based Monocular SLAM system (**[DropD-SLAM](https://github.com/tum-pf/dropd-slam)**), a robust Visual-Inertial SLAM system (**[ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)**), and a high-performance LiDAR-based LIO system (**[FAST-LIO2](https://github.com/Ericsii/FAST_LIO_ROS2)**).

## Repository Structure
- `/data`: External dataset mount (KITTI + WeatherKITTI).
- `/dropd-slam`: Source code and Dockerfile for Transformer-based Monocular SLAM.
- `/fast-lio-ros2`: ROS 2 workspace for LiDAR-based SLAM.
- `/ORB_SLAM3` Source code and Dockerfile for Vision-based SLAM.
- `/results`: Trajectory files (.txt) and `evo` comparison plots.
- `/scripts`: Scripts to compare results.

## Hardware Requirements
- **GPU:** NVIDIA RTX GPU (for DropD SLAM).
- **Host OS:** Ubuntu 22.04 or WSL2.

## Setup Instructions

### 1. Prerequisites
1. Install `git-lfs`:
```bash
sudo apt update && sudo apt install git-lfs
git lfs install
```

2. Clone this repository:
```bash
git clone https://github.com/tannermi/rob-530-slam-comparison.git
```

3. Install Python + Dependencies:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-tk
pip install evo --upgrade --no-binary evo
```

3. Install [Docker](https://docs.docker.com/engine/install/ubuntu/):
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

4. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit):
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

5. [Configure Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration):
```bash
# Configure the container runtime by using the nvidia-ctk command:
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon:
sudo systemctl restart docker
```

### 2. Build container
Use Docker Compose to build and launch the specific environment you wish to evaluate:
```bash
# Build both images
docker compose build
```

## DropD SLAM Instructions
1. Start the Docker container:
```bash
docker compose run dropd-slam
```

2. Once inside the `dropd-slam` container, compile the project:
```bash
./build.sh
```

3. Run the code using the desired arguments:
```bash
./Examples/RGB-D/rgbd_tum <vocabulary> <config> <sequence_path> <associations> <depth_model_type> <depth_model_path>
```
* `vocabulary`: Path to ORB vocabulary file
* `config`: Path to YAML configuration file
* `sequence_path`: Path to TUM RGB-D dataset sequence
* `associations`: Path to associations file
* `depth_model_type`: Type of depth model (unidepth or depthanything)
* `depth_model_path`: Path to ONNX depth model

For example:
```bash
# Run DropD-SLAM with monocular depth estimation
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml /data/TUM/rgbd_dataset_freiburg1_desk2 /data/TUM/rgbd_dataset_freiburg1_desk2/associations.txt unidepth /workspace/models/unidepthv2_with_cam.onnx
```

## ORB-SLAM3 Instructions
1. Start the Docker container:
```bash
docker compose run orb-slam
```

2. Once inside the `orb-slam` container, compile the project:
```bash
./build.sh
```

3. Run the code on your desired sequence:
```bash
# Run ORB-SLAM3 with stereo cameras on sequence 0
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml /data/KITTI/odometry/00
```

## FAST-LIO2 Instructions
1. Start the Docker container:
```bash
docker compose run --name fast_lio_active fast-lio
```

2. Once inside the container, compile the project:
```bash
cd src/livox_ros_driver2
./build.sh humble
cd /workspace/fast_lio_ws
source install/setup.bash
```

3. Run the code
```bash
# In the first terminal (make sure config_path and config_file are correct)
ros2 launch fast_lio mapping.launch.py config_path:=/workspace/fast_lio_ws/src/FAST_LIO_ROS2/config config_file:=kitti.yaml

# Open a second terminal (make sure the ROSbag path is correct)
docker exec -it fast_lio_active bash
source install/setup.bash
ros2 bag play /data/KITTI/rosbags/2011_09_30/2011_09_30_drive_0018_extract_ros2
```

## Evaluation
We use the `evo` package to analyze the Absolute Pose Error (APE) and Relative Pose Error (RPE).

### Comparison Commands
To evaluate DropD SLAM against Ground Truth:
```bash
evo_ape tum data/ground_truth.txt results/dropd_traj.txt -va --plot
```

To compare DropD SLAM directly against FAST-LIO2:
```bash
evo_ape tum results/fast_lio_traj.txt results/dropd_traj.txt -va --plot
```

## Datasets
1. [TUM RGBD](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download): fr1/desk2, fr2/desk, and fr3/structure_texture_near
2. [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php): Scenes 00-10
3. [KITTI ROS2 bags](https://github.com/Jakubach/kitti_to_ros): Scenes 00-10
