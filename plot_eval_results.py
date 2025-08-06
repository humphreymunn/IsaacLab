import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
INPUT_CSV = "eval_results.csv"
OUTPUT_DIR = "plots"

# === Load CSV ===
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"'{INPUT_CSV}' not found. Please run the evaluation script first.")

df = pd.read_csv(INPUT_CSV)

# === Create output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Map multihead values to new labels ===
df["method"] = df["multihead"].map({True: "PCGrad", False: "Baseline"})

# === Compute stats: mean, std, count ===
grouped = (
    df.groupby(["task", "method"])["reward"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "reward_mean", "std": "reward_std", "count": "num_runs"})
)

# Drop rows with NaN mean (just in case)
grouped = grouped.dropna(subset=["reward_mean"])

# === Generate separate plots per task ===
for task in grouped["task"].unique():
    task_data = grouped[grouped["task"] == task]

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(
        data=task_data,
        x="method",
        y="reward_mean",
        hue="method",
        errorbar=None,  # For seaborn >= 0.12
        palette="Set2"
    )

    # Add std dev bars and reward labels
    for i, bar in enumerate(ax.patches):
        if i >= len(task_data):
            continue  # Skip bars with no matching data
        row = task_data.iloc[i]
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(
            x,
            0.02 * ax.get_ylim()[1],
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )
        row = task_data.iloc[i]
        if row["num_runs"] > 1 and pd.notnull(row["reward_std"]):
            ax.errorbar(
                x=x,
                y=height,
                yerr=row["reward_std"],
                fmt='none',
                ecolor='black',
                capsize=4,
                linewidth=1.5
            )

    plt.title(f"Baseline vs PCGrad Performance: {task}")
    plt.xlabel("Method")
    plt.ylabel("Average Reward")
    plt.legend(title="Method", loc="upper left")
    plt.tight_layout()

    # Save each plot with safe filename
    plot_filename = f"{task.replace('/', '_')}_comparison.png"
    save_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()
