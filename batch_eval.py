import os
import subprocess
import pandas as pd
import re
from tqdm import tqdm

# === Folder name → full task name mapping ===
folder_to_task = {
    "shadow_hand": "Isaac-Repose-Cube-Shadow-Direct-v0",
    "humanoid": "Isaac-Humanoid-v0",
    "g1_rough": "Isaac-Velocity-Rough-G1-v0",
    "ant": "Isaac-Ant-v0",
    "digit_loco_manip": "Isaac-Tracking-LocoManip-Digit-v0",
    "allegro_cube": "Isaac-Repose-Cube-Allegro-v0",
    "franka_open_drawer": "Isaac-Open-Drawer-Franka-v0",
    "franka_lift": "Isaac-Lift-Cube-Franka-v0",
    "reach_ur10": "Isaac-Reach-UR10-v0",
    "franka_reach": "Isaac-Reach-Franka-v0",
}

# === Root log directory ===
log_root = "logs/rsl_rl"
task_dirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]

results = []

for folder_name in task_dirs:
    if folder_name not in folder_to_task:
        print(f"[WARNING] Folder '{folder_name}' not in known task mapping. Skipping.")
        continue

    task_name = folder_to_task[folder_name]
    task_path = os.path.join(log_root, folder_name)
    run_dirs = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]

    for run in tqdm(run_dirs, desc=f"Evaluating {task_name}"):
        run_path = os.path.join(task_path, run)

        # Expected run name format: 2025-08-05_14-00-33_True_1
        match = re.match(r"(.+)_([Tt]rue|[Ff]alse)_(\d+)", run)
        if not match:
            print(f"[WARNING] Skipping unrecognized run name format: {run}")
            continue

        date_time, multihead_str, seed = match.groups()
        multihead = multihead_str.lower() == "true"
        run_name = run

        # Build the play command
        cmd = [
            "./isaaclab.sh",
            "-p", "scripts/reinforcement_learning/rsl_rl/play.py",
            f"--task={task_name}",
            "--num_envs", "4092",
            "--headless",
            "--load_run", run_name,
        ]
        if multihead:
            cmd.append("--use_critic_multi")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            stdout = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {run_name}: {e}")
            continue
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Run {run_name} took too long.")
            continue

        # Extract reward from stdout
        reward_match = re.search(r"Final average episode reward: ([\d\.\-eE]+)", stdout)
        reward = float(reward_match.group(1)) if reward_match else None
        if reward is None:
            print(f"[WARNING] Reward not found for run: {run_name}")

        results.append({
            "task": task_name,
            "run_name": run_name,
            "seed": int(seed),
            "multihead": multihead,
            "reward": reward
        })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("eval_results.csv", index=False)
print("✅ Saved results to eval_results.csv")
