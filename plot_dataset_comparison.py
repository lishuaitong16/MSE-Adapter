import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_comparison(dataset_name, metrics, before_scores, after_scores, ax):
    """
    Plots a grouped bar chart for a single dataset comparison.
    """
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    # Plot the two groups of bars
    rects1 = ax.bar(x - width/2, before_scores, width, label='Before CLS Token', color='#4C72B0')
    rects2 = ax.bar(x + width/2, after_scores, width, label='After CLS Token', color='#DD8452')

    # Add text labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores (%)', fontsize=12)
    ax.set_title(f'{dataset_name} Dataset - ChatGLM3-6B', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)

    # Dynamically adjust Y-axis limits for better visual comparison
    min_val = min(min(before_scores), min(after_scores))
    max_val = max(max(before_scores), max(after_scores))
    ax.set_ylim(min_val - 2, max_val + 3)

    # Attach text labels above each bar
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=10)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=10)

# ================= Data Preparation =================
metrics_labels = ['Accuracy (%)', 'Weighted F1 (%)']

# MELD data (Accuracy, Weighted F1)
meld_before = [60.77, 60.21]
meld_after = [64.64, 61.82]

# CHERMA data (Accuracy, Weighted F1)
cherma_before = [73.03, 72.85]
cherma_after = [68.87, 68.80]

# ================= Plotting =================
# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot both datasets
plot_dataset_comparison('MELD', metrics_labels, meld_before, meld_after, ax1)
plot_dataset_comparison('CHERMA', metrics_labels, cherma_before, cherma_after, ax2)

# Adjust layout and display
plt.tight_layout()
plt.savefig('chatglm3_6b_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()