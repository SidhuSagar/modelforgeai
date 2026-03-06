# core/visualizer.py
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_classification_report(report: dict, model_name: str, save_dir="outputs/plots"):
    """
    Creates visual plots for classification metrics and accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract overall accuracy if available
    accuracy = report.get("accuracy", None)

    # Extract per-class metrics
    labels = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    precisions = [report[k].get("precision", 0) for k in labels]
    recalls = [report[k].get("recall", 0) for k in labels]
    f1s = [report[k].get("f1-score", 0) for k in labels]

    # --------------------------
    # Plot grouped bar chart
    # --------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_width = 0.25
    x = np.arange(len(labels))

    ax.bar(x, precisions, bar_width, label="Precision", alpha=0.8)
    ax.bar(x + bar_width, recalls, bar_width, label="Recall", alpha=0.8)
    ax.bar(x + 2 * bar_width, f1s, bar_width, label="F1-score", alpha=0.8)

    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("Scores", fontsize=12)
    ax.set_title(f"Model Performance – {model_name}", fontsize=14, weight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Annotate bars with values
    for i, v in enumerate(precisions):
        ax.text(i - 0.1, v + 0.02, f"{v:.2f}", fontsize=9, rotation=0)
    for i, v in enumerate(recalls):
        ax.text(i + 0.15, v + 0.02, f"{v:.2f}", fontsize=9, rotation=0)
    for i, v in enumerate(f1s):
        ax.text(i + 0.42, v + 0.02, f"{v:.2f}", fontsize=9, rotation=0)

    # Add overall accuracy text
    if accuracy is not None:
        plt.figtext(
            0.9, 0.92, f"Accuracy: {accuracy:.3f}",
            fontsize=11, color="green", weight="bold", ha="right"
        )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_report.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def visualize_training_summary(result_json: dict):
    """
    Generate and save performance visualizations from a training result.
    """
    if isinstance(result_json, str):
        try:
            data = json.loads(result_json)
        except json.JSONDecodeError:
            print("⚠️ Could not parse result JSON for visualization.")
            return None
    else:
        data = result_json

    report = data.get("report", {})
    model_name = data.get("model_name", "model")

    if not report:
        print("⚠️ No classification report found, skipping visualization.")
        return None

    path = plot_classification_report(report, model_name)
    print(f"📊 Classification report plot saved at: {path}")
    return path
