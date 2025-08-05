import os
import subprocess
import pandas as pd
import re
from tqdm import tqdm

# Root log dir
log_root = "logs/rsl_rl"
task_dirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]

results = []

for task in task_dirs:
    task_path = os.path.join(log_root, task)
    run_dirs = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]

    for run in tqdm(run_dirs, desc=f"Evaluating {task}"):
        run_path = os.path.join(task_path, run)

        # Example run name: 2025-08-05_14-00-33_True_1
        match = re.match(r"(.+)_([Tt]rue|[Ff]alse)_(\d+)", run)
        if not match:
            print(f"[WARNING] Skipping unrecognized run name format: {run}")
            continue

        date_time, multihead_str, seed = match.groups()
        multihead = multihead_str.lower() == "true"
        run_name = run

        # Construct play command
        cmd = [
            "./isaaclab.sh",
            "-p", "scripts/reinforcement_learning/rsl_rl/play.py",
            f"--task=Isaac-{task.capitalize()}-v0",
            "--num_envs", "4092",
            "--headless",
            "--load_run", run_name,
        ]
        if multihead:
            cmd.append("--use_critic_multi")

        # Run subprocess and capture output
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            stdout = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {run_name}: {e}")
            continue
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Run {run_name} took too long.")
            continue

        # Extract final reward from stdout
        reward_match = re.search(r"Final average episode reward: ([\d\.\-eE]+)", stdout)
        if reward_match:
            reward = float(reward_match.group(1))
        else:
            print(f"[WARNING] Reward not found in output of {run_name}")
            reward = None

        # Append to result table
        results.append({
            "task": task,
            "run_name": run_name,
            "seed": int(seed),
            "multihead": multihead,
            "reward": reward
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("eval_results.csv", index=False)
print("✅ Saved results to eval_results.csv")
