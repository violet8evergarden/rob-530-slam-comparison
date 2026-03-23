# To D or Not to D. That is the SLAM Question: A Comparative Analysis of Monocular SLAM

This repository evaluates the performance and accuracy trade-offs between a Transformer-based Monocular SLAM system (**DropD-SLAM**) and a high-performance LiDAR-based LIO system (**FAST-LIO**).

## Repository Structure
- `/dropd-slam`: Source code and Dockerfile for Vision-based SLAM.
- `/fast-lio-ros2`: ROS 2 workspace for LiDAR-based SLAM.
- `/data`: External dataset mount (KITTI + WeatherKITTI).
- `/results`: Trajectory files (.txt) and `evo` comparison plots.

## Hardware Requirements
- **GPU:** NVIDIA RTX GPU.
\- **Host OS:** Ubuntu 22.04 or WSL2.

## Setup Instructions

### 1. Prerequisites
Install the NVIDIA Container Toolkit on your host:
```bash
sudo apt-get install -y nvidia-container-toolkit
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
sudo chmod +x build.sh
./build.sh
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
