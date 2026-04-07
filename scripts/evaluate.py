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
        
        self._convert_fast_lio_if_needed()
        self._clean_nans()
        self.format = self._prepare_formats()

    def _generate_identifier(self):
        path_parts = self.est.split(os.sep)
        if len(path_parts) >= 4:
            return "_".join(path_parts[-4:-1])
        return os.path.basename(self.results_dir)

    def _clean_nans(self):
        """Removes any rows containing 'nan' to prevent EVO from freezing."""
        if not os.path.exists(self.est): return
        with open(self.est, 'r') as f:
            lines = f.readlines()
            
        clean_lines = [l for l in lines if "nan" not in l.lower()]
        
        if len(clean_lines) < len(lines):
            dropped = len(lines) - len(clean_lines)
            print(f"   [Warning] Cleaned {dropped} 'NaN' frames from trajectory.")
            with open(self.est, 'w') as f:
                f.writelines(clean_lines)

    def _convert_fast_lio_if_needed(self):
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

        if gt_fmt == "tum" and est_fmt == "tum":
            print(f"   [{self.identifier}] Both GT and Est are in TUM format. No conversion needed.")
            return "tum"
        
        seq_root = os.path.dirname(os.path.dirname(self.gt))
        times_path = os.path.join(seq_root, "times.txt")
        
        if not os.path.exists(times_path):
            print(f"   [Error] Could not find times.txt at {times_path}")
            return gt_fmt

        if gt_fmt == "kitti":
            print(f"   [{self.identifier}] Converting KITTI GT to TUM for time-sync...")
            gt_tum = self.gt.replace(".txt", "_tum.txt")
            if not os.path.exists(gt_tum):
                times = np.loadtxt(times_path)
                gt_poses = np.loadtxt(self.gt)
                
                tum_gt_data = []
                # Use min() to prevent index out of bounds if file lengths differ
                for i in range(min(len(times), len(gt_poses))):
                    pose = gt_poses[i].reshape(3, 4)
                    t = pose[:, 3]
                    q = R.from_matrix(pose[:, :3]).as_quat()
                    tum_gt_data.append([times[i], t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
                np.savetxt(gt_tum, tum_gt_data, fmt='%.6f')
            self.gt = gt_tum

        if est_fmt == "kitti":
            print(f"   [{self.identifier}] Est is KITTI format. Converting to TUM...")
            est_tum = self.est.replace(".txt", "_tum.txt")
            
            times = np.loadtxt(times_path)
            est_poses = np.loadtxt(self.est)
            
            tum_est_data = []
            for i in range(len(est_poses)):
                pose = est_poses[i].reshape(3, 4)
                t = pose[:, 3]
                q = R.from_matrix(pose[:, :3]).as_quat()
                # We use times[i] because the first line of SLAM usually corresponds 
                # to the first (or first processed) timestamp
                tum_est_data.append([times[i], t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
            np.savetxt(est_tum, tum_est_data, fmt='%.6f')
            self.est = est_tum
        
        return "tum"


    def calculate_success_rate(self):
        """Calculates success based on total images in sequence vs tracked poses."""
        def count_lines(file_path):
            if not file_path or not os.path.exists(file_path): return 0
            with open(file_path, 'r') as f:
                return len([l for l in f.readlines() if l.strip() and not l.startswith('#')])

        # 1. Frames Tracked (Actual output from SLAM)
        num_est = count_lines(self.est)
        
        # 2. Frames Possible (Pulled from associations.txt)
        assoc_path = os.path.join(os.path.dirname(self.gt), "..", "associations.txt")
        
        if os.path.exists(assoc_path):
            num_possible = count_lines(assoc_path)
        else:
            # Fallback: if association file is missing, we use the GT length 
            print(f"   [Warning] associations.txt not found at {assoc_path}. Using GT as fallback.")
            num_possible = count_lines(self.gt)

        self.results["frames_tracked"] = num_est
        self.results["frames_total_possible"] = num_possible
        self.results["tracking_success_percent"] = round((num_est / num_possible) * 100, 2) if num_possible > 0 else 0.0


    def compute_rpe_deltas(self):
        '''
        For checking drift
        '''
        for delta, unit in [(1, "f"), (10, "f"), (100, "m")]:
            for pose_rel, key in [("trans_part", "trans"), ("angle_deg", "rot")]:
                cmd = ["evo_rpe", self.format, self.gt, self.est,
                    "-va", "--silent",
                    "--delta", str(delta),
                    "--delta_unit", unit,
                    "--pose_relation", pose_rel]
                
                proc = subprocess.run(cmd, capture_output=True, text=True)
                rmse = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc.stdout, re.IGNORECASE)
                if rmse:
                    k = f"rpe_{key}_rmse_delta{delta}{unit}"
                    self.results[k] = float(rmse.group(1))

    # TODO Broken -> not aligned correctly?
    def compute_per_axis_rmse(self):
        """Load both trajectories, align, and compute RMSE per axis."""
        try:
            est_data = np.loadtxt(self.est)
            gt_data  = np.loadtxt(self.gt)
            
            # Sync by nearest timestamp
            from scipy.spatial import KDTree
            gt_ts = gt_data[:, 0]
            est_ts = est_data[:, 0]
            tree = KDTree(gt_ts.reshape(-1, 1))
            _, idx = tree.query(est_ts.reshape(-1, 1))
            gt_sync = gt_data[idx]

            # x, y, z columns
            diff = est_data[:, 1:4] - gt_sync[:, 1:4]
            for i, axis in enumerate(["x", "y", "z"]):
                self.results[f"rmse_{axis}"] = float(np.sqrt(np.mean(diff[:, i] ** 2)))
            self.results["rmse_3d"] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            
        except Exception as e:
            print(f"compute per axis RMSE failed for some reason: {e}")
    

    def compute_drift_rate(self):
        """ATE normalized by total path length for a fair comparison"""
        est_data = np.loadtxt(self.est)
        positions = est_data[:, 1:4]
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        if "ape_rmse_se3_metric" in self.results and path_length > 0:
            self.results["drift_rate_m_per_m"] = round(self.results["ape_rmse_se3_metric"] / path_length, 6)
            self.results["path_length_m"] = round(path_length, 3)
    

    def compute_tracking_loss(self):
        """Detects gaps in trajectory that indicate tracking loss"""
        est_data = np.loadtxt(self.est)
        timestamps = est_data[:, 0]
        gaps = np.diff(timestamps)
        
        median_dt = np.median(gaps)
        # A gap > 5x median interval means tracking loss
        loss_events = gaps[gaps > median_dt * 5]
        
        self.results["tracking_loss_events"]   = int(len(loss_events))
        self.results["total_tracking_loss_s"]  = round(float(np.sum(loss_events)), 3)
        self.results["longest_loss_event_s"]   = round(float(np.max(loss_events)) if len(loss_events) > 0 else 0, 3)
        recovered = 0
        for i, gap in enumerate(gaps):
            if gap > median_dt * 5:
                # Check if tracking resumed within 10 frames = recovery
                if i + 1 < len(timestamps):
                    recovered += 1
        self.results["relocalization_success_rate"] = round(recovered / max(len(loss_events), 1), 3)


    def run_all_metrics(self):
        """APE (Metric & Aligned), RPE, and Success Rate."""
        self.calculate_success_rate()

        # 1. SE(3) APE - Metric Error
        cmd_se3 = ["evo_ape", self.format, self.gt, self.est, "--align", "-v"]
        proc_se3 = subprocess.run(cmd_se3, capture_output=True, text=True)
        
        if proc_se3.returncode != 0:
            print(f"   [ERROR] SE(3) EVO failed: {proc_se3.stderr} | STDOUT: {proc_se3.stdout}")
            
        rmse_se3 = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc_se3.stdout, re.IGNORECASE)
        if rmse_se3: 
            self.results["ape_rmse_se3_metric"] = float(rmse_se3.group(1))

        # 2. Sim(3) APE - Shape Error 
        cmd_sim3 = ["evo_ape", self.format, self.gt, self.est, "--align", "-s", "-v"]
        proc_sim3 = subprocess.run(cmd_sim3, capture_output=True, text=True)
        
        if proc_sim3.returncode != 0:
            print(f"   [ERROR] Sim(3) EVO failed: {proc_sim3.stderr}")
        
        rmse_sim3 = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc_sim3.stdout, re.IGNORECASE)
        scale_val = re.search(r"scale factor\s+[\W]*\s*([\d.]+)", proc_sim3.stdout, re.IGNORECASE)
        length = re.search(r"path_length\s+[\W]*\s*([\d.]+)", proc_sim3.stdout, re.IGNORECASE)

        if rmse_sim3: 
            self.results["ape_rmse_sim3_aligned"] = float(rmse_sim3.group(1))
            if length and float(length.group(1)) > 0:
                self.results["drift_percent"] = round((float(rmse_sim3.group(1)) / float(length.group(1))) * 100, 3)

        if scale_val:
            self.results["scale_factor"] = float(scale_val.group(1))
            self.results["scale_drift_percent"] = round(abs(1.0 - float(scale_val.group(1))) * 100, 2)

        # 3. RPE (Relative Pose Error)
        cmd_rpe = ["evo_rpe", self.format, self.gt, self.est, "--align", "-v"]
        proc_rpe = subprocess.run(cmd_rpe, capture_output=True, text=True)
        
        if proc_rpe.returncode != 0:
            print(f"   [ERROR] RPE EVO failed: {proc_rpe.stderr}")
            
        rpe_match = re.search(r"rmse\s+[\W]*\s*([\d.]+)", proc_rpe.stdout, re.IGNORECASE)
        if rpe_match: 
            self.results["rpe_rmse_metric"] = float(rpe_match.group(1))
        
        # 4. RPE with delta windows
        self.compute_rpe_deltas()

        # 5. RPE by axis
        self.compute_per_axis_rmse()

        # 6. Drift rate for ATE comparisons
        self.compute_drift_rate()

        # 7. Check how many times an algorithm lost tracking and for how long
        self.compute_tracking_loss()


    def parse_logs_and_complexity(self):
        # Look for Execution Mean logs (ORB-SLAM3/DropD style)
        exec_path = os.path.join(self.results_dir, "ExecMean.txt")
        if os.path.exists(exec_path):
            with open(exec_path, 'r') as f:
                content = f.read()
                track = re.search(r"Total Tracking:\s+([\d.]+)", content)
                mapping = re.search(r"Total Local Mapping:\s+([\d.]+)", content)
                if track: self.results["mean_tracking_ms"] = float(track.group(1))
                if mapping: self.results["mean_mapping_ms"] = float(mapping.group(1))
                kf = re.search(r"KFs in map:\s+(\d+)", content)
                mp = re.search(r"MPs in map:\s+(\d+)", content)
                if kf: self.results["keyframes_in_map"] = int(kf.group(1))
                if mp: self.results["points_in_map"] = int(mp.group(1))

        # Look for Time Stats logs
        orb_stats_path = os.path.join(self.results_dir, "TrackingTimeStats.txt")
        if os.path.exists(orb_stats_path):
            try:
                data = np.genfromtxt(orb_stats_path, delimiter=',', comments='#')
                if data.ndim == 1: data = data.reshape(1, -1)
                total_times = data[:, -1]
                self.results["mean_fps"] = round(1000.0 / np.mean(total_times), 2)
            except: pass


    def save_results(self):
        output_path = os.path.join(self.results_dir, "metrics.json")
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"   Done -> {self.identifier}")


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
    # NOTE: This script was for testing purposes and some metrics may be missing/inaccurate (e.g. depth inference time is not considered but should be)
    run_all_evaluations("./rob-530-slam-comparison/results")
