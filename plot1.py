import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_comparison(dataset_name, metrics, before_scores, after_scores, ax):
    """
    Plots a grouped bar chart for a single dataset comparison.
    """
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    # Plot the two groups of bars
    rects1 = ax.bar(x - width/2, before_scores, width, label='Before (Original)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, after_scores, width, label='After (New)', color='#DD8452')

    # Add text labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title(f'{dataset_name} Dataset - ChatGLM3-6B', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)

    # Dynamically adjust Y-axis limits to fit the annotations properly
    min_val = min(min(before_scores), min(after_scores))
    max_val = max(max(before_scores), max(after_scores))
    ax.set_ylim(min_val - 5, max_val + 8)

    # Attach text labels above each bar
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=10)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=10)

# ================= Data Preparation =================

# SIMS data
sims_metrics = ['Acc-2', 'F1', 'Acc2_weak', 'MAE', 'Corr']
sims_before = [79.40, 78.93, 70.81, 30.54, 70.17]
sims_after  = [80.46, 80.46, 72.05, 31.64, 70.74]

# MOSEI data
mosei_metrics = ['Acc-2', 'F1', 'Acc-7', 'MAE', 'Corr']
mosei_before = [86.91, 86.67, 54.69, 51.84, 78.53]
mosei_after  = [83.82, 84.26, 55.61, 49.63, 80.98]

# ================= Plotting =================

# Create a figure with 1 row and 2 columns, slightly wider to accommodate 5 bars per chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot both datasets
plot_dataset_comparison('SIMS', sims_metrics, sims_before, sims_after, ax1)
plot_dataset_comparison('MOSEI', mosei_metrics, mosei_before, mosei_after, ax2)

# Adjust layout and display
plt.tight_layout()
plt.savefig('chatglm3_6b_sims_mosei_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()