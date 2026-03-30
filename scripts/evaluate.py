import os
import re
import json
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

class SLAMEvaluator:
    def __init__(self, est_path, gt_path):
        self.est = os.path.abspath(est_path)
        self.gt = os.path.abspath(gt_path)
        self.results_dir = os.path.dirname(self.est)
        self.results = {}
        self.identifier = self._generate_identifier()
        
        # 1. Standardize FAST-LIO2 data before starting
        self._convert_fast_lio_if_needed()
        self.format = self._prepare_formats()

    def _generate_identifier(self):
        path_parts = self.est.split(os.sep)
        if len(path_parts) >= 4:
            return "_".join(path_parts[-4:-1])
        return os.path.basename(self.results_dir)

    def _convert_fast_lio_if_needed(self):
        """Converts FAST-LIO2 State Vector (pos_log.txt) to TUM format."""
        pos_log = os.path.join(self.results_dir, "pos_log.txt")
        if os.path.exists(pos_log) and (not os.path.exists(self.est) or "pos_log" in self.est):
            print(f"   [{self.identifier}] Converting FAST-LIO2 pos_log.txt...")
            try:
                data = np.loadtxt(pos_log)
                timestamps = data[:, 0]
                positions = data[:, 4:7]
                quats = R.from_rotvec(data[:, 1:4]).as_quat()
                tum_data = np.column_stack((timestamps, positions, quats))
                self.est = os.path.join(self.results_dir, "CameraTrajectory.txt")
                np.savetxt(self.est, tum_data, fmt='%.6f')
            except Exception as e:
                print(f"   Error in FAST-LIO conversion: {e}")

    def _detect_format(self, path):
        if not os.path.exists(path): return "unknown"
        with open(path, 'r') as f:
            for l in f:
                if l.strip() and not l.startswith('#'):
                    line = l.strip().split()
                    return "tum" if len(line) == 8 else "kitti" if len(line) == 12 else "tum"
        return "tum"

    def _prepare_formats(self):
        gt_fmt = self._detect_format(self.gt)
        est_fmt = self._detect_format(self.est)
        if gt_fmt != est_fmt and gt_fmt != "unknown":
            subprocess.run(["evo_traj", est_fmt, self.est, f"--save_as_{gt_fmt}"], capture_output=True)
            gen_ext = ".tum" if gt_fmt == "tum" else ".kitti"
            generated_file = self.est.replace(".txt", gen_ext)
            if os.path.exists(generated_file): self.est = generated_file
        return gt_fmt

    def calculate_success_rate(self):
        def count_lines(file_path):
            if not file_path or not os.path.exists(file_path): return 0
            with open(file_path, 'r') as f:
                # Count non-empty, non-comment lines
                return len([l for l in f.readlines() if l.strip() and not l.startswith('#')])

        try:
            num_est = count_lines(self.est)
            
            parent_dir = os.path.dirname(self.results_dir)
            assoc_path = os.path.join(parent_dir, "associations.txt")
            
            if os.path.exists(assoc_path):
                num_possible = count_lines(assoc_path)
            else:
                # Fallback to GT if no associations.txt
                num_possible = count_lines(self.gt)

            # 2. Calculate percentage
            self.results["tracking_success_percent"] = round(min((num_est / num_possible) * 100, 100.0), 2)
            self.results["frames_tracked"] = num_est
            self.results["frames_total_possible"] = num_possible
            
        except Exception as e:
            print(f"Error calculating success rate: {e}")
            self.results["tracking_success_percent"] = 0.0

    def run_all_metrics(self):
        """APE (Metric & Aligned), RPE, and Success Rate."""
        # 1. Success Rate
        self.calculate_success_rate()

        # 2. SE(3) APE
        proc_se3 = subprocess.run(["evo_ape", self.format, self.gt, self.est, "-va", "--silent"], capture_output=True, text=True)
        rmse_se3 = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc_se3.stdout, re.IGNORECASE)
        if rmse_se3: self.results["ape_rmse_se3_metric"] = float(rmse_se3.group(1))

        # 3. Sim(3) APE & Scale Factor
        proc_sim3 = subprocess.run(["evo_ape", self.format, self.gt, self.est, "-va", "-s", "--silent"], capture_output=True, text=True)
        rmse_sim3 = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc_sim3.stdout, re.IGNORECASE)
        scale_val = re.search(r"scale factor\s+[\W]*\s*([\d.]+)", proc_sim3.stdout, re.IGNORECASE)
        if rmse_sim3: self.results["ape_rmse_sim3_aligned"] = float(rmse_sim3.group(1))
        if scale_val: 
            self.results["scale_factor"] = float(scale_val.group(1))
            self.results["scale_drift_percent"] = round(abs(1.0 - float(scale_val.group(1))) * 100, 2)
        else:
            self.results["scale_factor"] = 1.0

        # 4. RPE
        proc_rpe = subprocess.run(["evo_rpe", self.format, self.gt, self.est, "-va", "--silent"], capture_output=True, text=True)
        rpe_match = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc_rpe.stdout, re.IGNORECASE)
        if rpe_match: self.results["rpe_rmse_metric"] = float(rpe_match.group(1))

    def parse_logs_and_complexity(self):
        """Parses every possible stat from DropD-SLAM, ORB-SLAM3, or FAST-LIO2 logs."""
        
        # DropD-SLAM/ORB_SLAM3 logs
        exec_path = os.path.join(self.results_dir, "ExecMean.txt")
        if os.path.exists(exec_path):
            with open(exec_path, 'r') as f:
                content = f.read()
                # Timing
                track = re.search(r"Total Tracking:\s+([\d.]+)", content)
                mapping = re.search(r"Total Local Mapping:\s+([\d.]+)", content)
                loop = re.search(r"Total Loop Closing:\s+([\d.]+)", content)
                if track: self.results["mean_tracking_ms"] = float(track.group(1))
                if mapping: self.results["mean_mapping_ms"] = float(mapping.group(1))
                if loop: self.results["mean_loop_closure_ms"] = float(loop.group(1))
                # Complexity
                kf = re.search(r"KFs in map:\s+(\d+)", content)
                mp = re.search(r"MPs in map:\s+(\d+)", content)
                if kf: self.results["keyframes_in_map"] = int(kf.group(1))
                if mp: self.results["points_in_map"] = int(mp.group(1))

        orb_stats_path = os.path.join(self.results_dir, "TrackingTimeStats.txt")
        if os.path.exists(orb_stats_path):
            try:
                data = np.genfromtxt(orb_stats_path, delimiter=',', comments='#')
                total_times = data[:, -1] if data.ndim > 1 else np.array([data[-1]])
                self.results["max_latency_ms"] = float(np.max(total_times))
                self.results["latency_std_dev"] = float(np.std(total_times))
                self.results["mean_fps"] = round(1000.0 / np.mean(total_times), 2)
            except: pass

        # Fast-LIO2 logs
        lio_time_path = os.path.join(self.results_dir, "fast_lio_time_log.csv")
        if os.path.exists(lio_time_path):
            try:
                # time_stamp, total time, scan point size, incremental time, search time, 
                # delete size, delete time, tree size st, tree size end, add point size, preprocess time
                data = np.genfromtxt(lio_time_path, delimiter=',', skip_header=1)
                total_times_ms = data[:, 1] * 1000
                self.results["mean_tracking_ms"] = float(np.mean(total_times_ms))
                self.results["max_latency_ms"] = float(np.max(total_times_ms))
                self.results["latency_std_dev"] = float(np.std(total_times_ms))
                self.results["mean_fps"] = round(1000.0 / np.mean(total_times_ms), 2)
                
                # Complexity analogues for LiDAR
                self.results["avg_points_per_scan"] = int(np.mean(data[:, 2]))
                self.results["final_tree_size"] = int(data[-1, 8]) # End tree size
                self.results["mean_ikd_tree_search_ms"] = float(np.mean(data[:, 4]) * 1000)
            except: pass

    def save_results(self):
        output_path = os.path.join(self.results_dir, "metrics.json")
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"   Done -> {output_path}")

def run_all_evaluations(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "ground_truth" in root: continue
        if "CameraTrajectory.txt" in files or "pos_log.txt" in files:
            est_path = os.path.join(root, "CameraTrajectory.txt")
            if "pos_log.txt" in files and "CameraTrajectory.txt" not in files:
                est_path = os.path.join(root, "pos_log.txt")
            
            parent_dir = os.path.dirname(root)
            gt_dir = os.path.join(parent_dir, "ground_truth")
            if not os.path.exists(gt_dir): continue
            gt_txts = [f for f in os.listdir(gt_dir) if f.endswith(".txt") and "metrics" not in f]
            if not gt_txts: continue
            
            print(f"Processing: {root}")
            eval_obj = SLAMEvaluator(est_path, os.path.join(gt_dir, gt_txts[0]))
            eval_obj.run_all_metrics()
            eval_obj.parse_logs_and_complexity()
            eval_obj.save_results()

if __name__ == "__main__":
    run_all_evaluations("./results")
