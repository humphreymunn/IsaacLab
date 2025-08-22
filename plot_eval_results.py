import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "eval_results_aug15_with_task_clea2.csv"
OUTPUT_DIR = "plots"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"'{INPUT_CSV}' not found. Please run the evaluation script first.")

df = pd.read_csv(INPUT_CSV)

# Expect a column named 'method' with values 'baseline' or 'pcgrad'
if "method" not in df.columns:
    raise ValueError("CSV must contain a 'method' column with values 'baseline' or 'pcgrad'.")

# Normalize method column
df["method"] = df["method"].astype(str).str.strip().str.lower()
valid_methods = {"baseline", "pcgrad"}
unknown = set(df["method"].unique()) - valid_methods
if unknown:
    print(f"Warning: Unknown methods found and dropped: {unknown}")
    df = df[df["method"].isin(valid_methods)]

# Optional: pretty display names
display_map = {"baseline": "Baseline", "pcgrad": "PCGrad"}
df["method_display"] = df["method"].map(display_map)

# Order
method_order = ["baseline", "pcgrad"]
display_order = [display_map[m] for m in method_order]

# Compute stats (averages with std/count)
grouped = (
    df.groupby(["task", "method_display"])["reward"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .rename(columns={"mean": "reward_mean", "std": "reward_std", "count": "num_runs"})
      .dropna(subset=["reward_mean"])
)

# Compute stats (max)
grouped_max = (
    df.groupby(["task", "method_display"])["reward"]
      .agg(["max"])
      .reset_index()
      .rename(columns={"max": "reward_max"})
      .dropna(subset=["reward_max"])
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Average plots (original behavior)
for task in grouped["task"].unique():
    task_data = grouped[grouped["task"] == task]

    plt.figure(figsize=(5, 4))
    ax = sns.barplot(
        data=task_data,
        x="method_display",
        y="reward_mean",
        hue="method_display",
        order=display_order,
        hue_order=display_order,
        errorbar=None,
        palette="Set2"
    )

    ymax = ax.get_ylim()[1]
    for bar, (_, row) in zip(ax.patches, task_data.iterrows()):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(
            x,
            max(0, height) + 0.01 * ymax,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )
        if row["num_runs"] > 1 and pd.notnull(row["reward_std"]):
            ax.errorbar(
                x=x,
                y=height,
                yerr=row["reward_std"],
                fmt='none',
                ecolor='black',
                capsize=4,
                linewidth=1.2
            )

    plt.title(f"Baseline vs PCGrad: {task}")
    plt.xlabel("Method")
    plt.ylabel("Average Reward")
    plt.legend(title="Method", loc="upper left")
    plt.tight_layout()

    plot_filename = f"{task.replace('/', '_')}_baseline_pcgrad.png"  # keep original name for averages
    save_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.close()

# Max plots (additional)
for task in grouped_max["task"].unique():
    task_data = grouped_max[grouped_max["task"] == task]

    plt.figure(figsize=(5, 4))
    ax = sns.barplot(
        data=task_data,
        x="method_display",
        y="reward_max",
        hue="method_display",
        order=display_order,
        hue_order=display_order,
        errorbar=None,
        palette="Set2"
    )

    ymax = ax.get_ylim()[1]
    for bar, (_, row) in zip(ax.patches, task_data.iterrows()):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(
            x,
            max(0, height) + 0.01 * ymax,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    plt.title(f"Baseline vs PCGrad (Max): {task}")
    plt.xlabel("Method")
    plt.ylabel("Max Reward")
    plt.legend(title="Method", loc="upper left")
    plt.tight_layout()

    plot_filename = f"{task.replace('/', '_')}_baseline_pcgrad_max.png"
    save_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.close()
