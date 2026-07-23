import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a clean, professional style
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# DATA — from your completed 5-experiment matrix
# ============================================================
runs = ['Run 1\n70B, T=0.2', 'Run 2\n70B, T=0.0', 'Run 3\n70B, T=0.5',
        'Run 4\n8B, T=0.2', 'Run 5\n8B, T=0.0']

accuracy = [40, 20, 20, 60, 60]
sharpe = [0.834, -2.507, -1.481, 2.642, 2.485]
total_return = [2.85, -5.27, -0.84, 10.77, 10.55]
max_drawdown = [-4.39, -7.74, -2.72, -4.08, -4.96]

# Colour scheme: 70B = coral/red tones, 8B = teal/blue tones
colors_accuracy = ['#e94560', '#e94560', '#e94560', '#00b4d8', '#00b4d8']
colors_sharpe = ['#e94560', '#e94560', '#e94560', '#00b4d8', '#00b4d8']

# ============================================================
# FIGURE 4.1 — Classification Accuracy by Experimental Condition
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(runs, accuracy, color=colors_accuracy, edgecolor='white', linewidth=0.8, width=0.6)

# Add value labels on bars
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Naive all-BUY baseline
ax.axhline(y=60, color='#666666', linestyle='--', linewidth=1.2, label='Naive All-BUY Baseline (60%)')

ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Figure 4.1: Classification Accuracy Across Experimental Conditions')
ax.set_ylim(0, 80)
ax.legend(loc='upper left', framealpha=0.9)

# Add model scale labels
ax.text(1, 75, '70B Model', ha='center', fontsize=10, color='#e94560', fontstyle='italic')
ax.text(3.5, 75, '8B Model', ha='center', fontsize=10, color='#00b4d8', fontstyle='italic')

plt.tight_layout()
plt.savefig('figures/fig_4_1_accuracy.png')
plt.close()
print('✅ Saved figures/fig_4_1_accuracy.png')

# ============================================================
# FIGURE 4.2 — Sharpe Ratio Comparison with Benchmark
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

bars = ax.bar(runs, sharpe, color=colors_sharpe, edgecolor='white', linewidth=0.8, width=0.6)

# Value labels
for bar, val in zip(bars, sharpe):
    y_pos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.25
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:.3f}', ha='center', va=va, fontweight='bold', fontsize=10)

# Benchmark line
ax.axhline(y=1.57, color='#52b788', linestyle='--', linewidth=1.5, label='Buy-and-Hold Benchmark (Sharpe ≈ 1.57)')
ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.5)

ax.set_ylabel('Annualised Sharpe Ratio')
ax.set_title('Figure 4.2: Risk-Adjusted Performance (Sharpe Ratio) vs Buy-and-Hold Benchmark')
ax.set_ylim(-3.5, 3.5)
ax.legend(loc='lower left', framealpha=0.9)

# Add model scale labels
ax.text(1, 3.2, '70B Model', ha='center', fontsize=10, color='#e94560', fontstyle='italic')
ax.text(3.5, 3.2, '8B Model', ha='center', fontsize=10, color='#00b4d8', fontstyle='italic')

# Shade negative region
ax.axhspan(-3.5, 0, alpha=0.05, color='red')
ax.text(4.7, -0.3, 'Wealth-destroying\nregion', fontsize=8, color='#999999', ha='center', fontstyle='italic')

plt.tight_layout()
plt.savefig('figures/fig_4_2_sharpe.png')
plt.close()
print('Saved figures/fig_4_2_sharpe.png')

# ============================================================
# FIGURE 4.3 — Temperature Sensitivity (Dual-Axis: Accuracy + Sharpe)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: 70B temperature response
temps_70b = [0.0, 0.2, 0.5]
acc_70b = [20, 40, 20]
sharpe_70b = [-2.507, 0.834, -1.481]

ax1.plot(temps_70b, acc_70b, 'o-', color='#e94560', linewidth=2, markersize=8, label='Accuracy (%)')
ax1_twin = ax1.twinx()
ax1_twin.plot(temps_70b, sharpe_70b, 's--', color='#ff8a80', linewidth=2, markersize=8, label='Sharpe Ratio')

ax1.set_xlabel('Temperature')
ax1.set_ylabel('Accuracy (%)', color='#e94560')
ax1_twin.set_ylabel('Sharpe Ratio', color='#ff8a80')
ax1.set_title('(a) 70B Model: Temperature Sensitivity')
ax1.set_xticks(temps_70b)
ax1.set_ylim(0, 80)
ax1_twin.set_ylim(-3.5, 3.5)
ax1_twin.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.5)

# Add annotations for failure modes
ax1.annotate('Directional\nInversion', xy=(0.0, 20), xytext=(0.08, 50),
            fontsize=8, fontstyle='italic', color='#666666',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))
ax1.annotate('Decision\nParalysis', xy=(0.5, 20), xytext=(0.38, 50),
            fontsize=8, fontstyle='italic', color='#666666',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=9)

# Panel B: 8B temperature response
temps_8b = [0.0, 0.2]
acc_8b = [60, 60]
sharpe_8b = [2.485, 2.642]

ax2.plot(temps_8b, acc_8b, 'o-', color='#00b4d8', linewidth=2, markersize=8, label='Accuracy (%)')
ax2_twin = ax2.twinx()
ax2_twin.plot(temps_8b, sharpe_8b, 's--', color='#90e0ef', linewidth=2, markersize=8, label='Sharpe Ratio')

ax2.set_xlabel('Temperature')
ax2.set_ylabel('Accuracy (%)', color='#00b4d8')
ax2_twin.set_ylabel('Sharpe Ratio', color='#90e0ef')
ax2.set_title('(b) 8B Model: Temperature Stability')
ax2.set_xticks(temps_8b)
ax2.set_xlim(-0.05, 0.3)
ax2.set_ylim(0, 80)
ax2_twin.set_ylim(-3.5, 3.5)
ax2_twin.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.5)

# Stability annotation
ax2.annotate('Sharpe range:\nonly 0.16 points', xy=(0.1, 60), xytext=(0.15, 35),
            fontsize=9, fontstyle='italic', color='#52b788',
            arrowprops=dict(arrowstyle='->', color='#52b788', lw=0.8))

# Combined legend
lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper center', fontsize=9)

plt.suptitle('Figure 4.3: Temperature Sensitivity by Model Scale', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_4_3_temperature.png')
plt.close()
print('Saved figures/fig_4_3_temperature.png')

# ============================================================
# FIGURE 4.4 — Per-Ticker Decision Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))

tickers = ['RR.L', 'TSCO.L', 'AAL.L', 'BA.L', 'IHG.L']
ground_truth = ['BUY', 'HOLD', 'BUY', 'SELL', 'BUY']

decisions = {
    'Run 1\n70B T=0.2': ['SELL', 'HOLD', 'SELL', 'BUY', 'BUY'],
    'Run 2\n70B T=0.0': ['HOLD', 'HOLD', 'SELL', 'BUY', 'HOLD'],
    'Run 3\n70B T=0.5': ['HOLD', 'SELL', 'HOLD', 'HOLD', 'HOLD'],
    'Run 4\n8B T=0.2':  ['BUY', 'HOLD', 'HOLD', 'HOLD', 'BUY'],
    'Run 5\n8B T=0.0':  ['HOLD', 'SELL', 'BUY', 'SELL', 'BUY'],
}

# Encode: correct = 2 (green), wrong direction = 0 (red), partial = 1 (yellow)
def encode(decision, truth, ticker_idx):
    if decision == truth:
        return 2  # exact match
    return 0  # mismatch

run_labels = list(decisions.keys())
matrix = []
for run in run_labels:
    row = []
    for i, (dec, gt) in enumerate(zip(decisions[run], ground_truth)):
        row.append(encode(dec, gt, i))
    matrix.append(row)

matrix = np.array(matrix)

# Custom colormap: red (wrong) -> yellow (partial) -> green (correct)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#e94560', '#f4a261', '#52b788'])

im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

# Add decision text in each cell
for i in range(len(run_labels)):
    for j in range(len(tickers)):
        dec = decisions[run_labels[i]][j]
        color = 'white' if matrix[i, j] != 1 else 'black'
        ax.text(j, i, dec, ha='center', va='center', fontweight='bold', fontsize=10, color=color)

ax.set_xticks(range(len(tickers)))
ax.set_xticklabels([f'{t}\n(GT: {g})' for t, g in zip(tickers, ground_truth)], fontsize=9)
ax.set_yticks(range(len(run_labels)))
ax.set_yticklabels(run_labels, fontsize=9)
ax.set_title('Figure 4.4: Per-Ticker Decision Heatmap (Green = Correct, Red = Incorrect)')

# Add legend manually
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#52b788', label='Exact Match'),
                   Patch(facecolor='#e94560', label='Mismatch')]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))

plt.tight_layout()
plt.savefig('figures/fig_4_4_heatmap.png')
plt.close()
print('Saved figures/fig_4_4_heatmap.png')

print('\n🏆 All figures generated in figures/ directory')