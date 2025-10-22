import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Your data
steps = [0, 10, 20, 30, 40, 50, 60, 70, 80]

data = {
    'NSR': {
        '0.5B': [9.1, 9.3, 10.4, 9.0, 8.9, 9.6, 9.3, 10.1, 10.3],
        '1.5B': [26.2, 26.6, 24.5, 24.5, 24.6, 24.6, 26.8, 25.7, 26.3],
        '3B': [38.4, 36.4, 38.2, 38.0, 36.7, 36.7, 37.7, 36.2, 35.1]
    },
    'W-REINFORCE': {
        '0.5B': [9.1, 8.7, 9.5, 9.6, 10.7, 8.9, 9.1, 9.6, 10.0],
        '1.5B': [26.2, 25.0, 26.9, 25.8, 25.7, 25.3, 25.9, 26.4, 25.8],
        '3B': [38.4, 38.0, 38.4, 37.6, 38.6, 38.7, 37.4, 37.2, 37.5]
    },
    'PSR': {
        '0.5B': [9.1, 9.5, 8.6, 9.7, 9.8, 10.1, 10.4, 9.5, 9.8],
        '1.5B': [23.7, 25.4, 24.9, 27.6, 24.6, 25.8, 26.2, 27.2, 27.6],
        '3B': [35.9, 38.0, 36.3, 36.6, 36.9, 37.6, 37.5, 36.0, 36.9]
    }
}

def smooth_curve(y, window=2):
    """Apply moving average smoothing"""
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().values

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = {'NSR': '#2E86AB', 'W-REINFORCE': '#A23B72', 'PSR': '#F18F01'}
model_sizes = ['0.5B', '1.5B', '3B']

for idx, model in enumerate(model_sizes):
    ax = axes[idx]
    
    # Plot smoothed curves
    for method in ['NSR', 'W-REINFORCE', 'PSR']:
        y_raw = data[method][model]
        y_smooth = smooth_curve(y_raw, window=3)
        
        # Plot smoothed line (bold)
        ax.plot(steps, y_smooth, 
                color=colors[method],
                label=method, 
                linewidth=3,
                alpha=0.9,
                zorder=3)
        
        # Plot raw data as small dots
        ax.scatter(steps, y_raw,
                  color=colors[method],
                  s=20,
                  alpha=0.3,
                  zorder=2)
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Qwen2.5-{model}-Instruct', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_ylim([
        min(min(data[m][model]) for m in ['NSR', 'W-REINFORCE', 'PSR']) - 1,
        max(max(data[m][model]) for m in ['NSR', 'W-REINFORCE', 'PSR']) + 1
    ])

plt.tight_layout()
plt.savefig('training_curves_smoothed.png', dpi=300, bbox_inches='tight')
plt.show()