import matplotlib.pyplot as plt
import numpy as np

models = ['0.5B', '1.5B', '3B']
methods = ['Baseline', 'NSR', 'W-REINFORCE', 'PSR']

# Your data
data = {
    '0.5B': [9.1, 10.3, 10.0, 9.8],
    '1.5B': [26.2, 26.3, 25.8, 27.6],
    '3B': [38.4, 35.1, 37.5, 36.9]
}

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    values = [data[model][i] for model in models]
    bars = ax.bar(x + i*width, values, width, label=method)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Final Accuracy (Step 80) Across Models and Methods', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('final_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()