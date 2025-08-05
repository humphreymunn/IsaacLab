import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
INPUT_CSV = "eval_results.csv"
OUTPUT_DIR = "plots"
OUTPUT_PLOT = "multihead_comparison_all_tasks.png"

# === Load CSV ===
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"'{INPUT_CSV}' not found. Please run the evaluation script first.")

df = pd.read_csv(INPUT_CSV)

# === Create output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Compute stats: mean, std, count ===
grouped = (
    df.groupby(["task", "multihead"])["reward"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "reward_mean", "std": "reward_std", "count": "num_runs"})
)

# Drop rows with NaN mean (just in case)
grouped = grouped.dropna(subset=["reward_mean"])

# === Plot ===
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=grouped,
    x="task",
    y="reward_mean",
    hue="multihead",
    errorbar=None,  # For seaborn >= 0.12
    palette="Set2"
)

# === Add std dev bars and reward labels ===
bar_index = 0
for container in ax.containers:
    for bar in container.get_children():  # Safely access bar patches
        if not hasattr(bar, "get_height"):
            continue  # Skip non-bar elements (e.g., tuples)

        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2

        # Add average label near bottom
        ax.text(
            x,
            0.02 * ax.get_ylim()[1],
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

        # Add error bar if std is valid and multiple runs
        if bar_index < len(grouped):
            row = grouped.iloc[bar_index]
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
        bar_index += 1

# === Final formatting ===
plt.title("Multihead vs Non-Multihead Performance per Task")
plt.xlabel("Task")
plt.ylabel("Average Reward")
plt.xticks(rotation=45)
plt.legend(title="Multihead")
plt.tight_layout()

# Save and show
save_path = os.path.join(OUTPUT_DIR, OUTPUT_PLOT)
plt.savefig(save_path)
print(f"[Saved] {save_path}")
plt.show()
