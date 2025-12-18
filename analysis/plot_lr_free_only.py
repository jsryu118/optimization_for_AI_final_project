"""
Visualization for LR-Free vs Baseline only (ScheduleFree 제외).
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from analyze_lr_free_only import (
    get_best_results_by_task,
    get_all_results,
    categorize_optimizer
)

matplotlib.use('Agg')

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')


def get_category_color(category):
    """Get color for optimizer category."""
    colors = {
        'parameter-free': '#2ecc71',  # Green (LR-Free)
        'baseline': '#3498db'          # Blue
    }
    return colors.get(category, '#95a5a6')


def plot_lr_sensitivity(task, save_dir="final_plots"):
    """Plot LR sensitivity for baseline optimizers."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_results = get_all_results()

    # Separate baseline optimizers by LR
    baseline_data = defaultdict(list)
    for (t, opt_family), results in all_results.items():
        if t == task and categorize_optimizer(opt_family) == 'baseline':
            if len(results) > 1:  # Multiple LRs tested
                sorted_results = sorted(results, key=lambda x: x['lr'])
                baseline_data[opt_family] = sorted_results

    if not baseline_data:
        print(f"No baseline LR sensitivity data for {task}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'sgd': '#3498db', 'adam': '#e74c3c', 'adamw': '#f39c12'}

    for opt_family, results in baseline_data.items():
        lrs = [r['lr'] for r in results]
        accs = [r['test_accuracy'] for r in results]

        ax.plot(lrs, accs, 'o-', color=colors.get(opt_family, '#95a5a6'),
               label=opt_family.upper(), linewidth=2.5, markersize=10)

        # Mark best
        best_idx = np.argmax(accs)
        ax.plot(lrs[best_idx], accs[best_idx], '*', markersize=20,
               color='red', markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlabel('Learning Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'{task.upper()} - Baseline LR Sensitivity',
                 fontweight='bold', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    filename = f"{save_dir}/{task}_lr_sensitivity.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


from collections import defaultdict


def plot_best_comparison(task, save_dir="final_plots"):
    """Compare best LR-Free vs Baseline."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    best_results = get_best_results_by_task()

    if task not in best_results:
        print(f"No results for {task}")
        return

    # Separate by category
    param_free = []
    baseline = []

    for opt_family, result in best_results[task].items():
        category = categorize_optimizer(opt_family)
        if category == 'parameter-free':
            param_free.append((opt_family, result))
        else:
            baseline.append((opt_family, result))

    # Sort by accuracy
    param_free.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)
    baseline.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)

    # Combine
    all_data = param_free + baseline
    opt_names = [x[0] for x in all_data]
    accuracies = [x[1]['test_accuracy'] for x in all_data]
    colors = [get_category_color(categorize_optimizer(x[0])) for x in all_data]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(range(len(opt_names)), accuracies, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Optimizer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{task.upper()} - LR-Free vs Baseline (Best Configurations)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=get_category_color('parameter-free'), edgecolor='black',
              label='Learning Rate Free (LR=1.0, no tuning)'),
        Patch(facecolor=get_category_color('baseline'), edgecolor='black',
              label='Baseline (Best LR from tuning)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    # Add separator line
    if param_free and baseline:
        sep_x = len(param_free) - 0.5
        ax.axvline(sep_x, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    plt.tight_layout()

    filename = f"{save_dir}/{task}_lr_free_vs_baseline.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_summary_all_tasks(save_dir="final_plots"):
    """Summary plot for all tasks."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    best_results = get_best_results_by_task()
    tasks = ['cifar10', 'oxford_pet', 'sst2']
    task_titles = ['CIFAR-10\n(From Scratch)', 'Oxford-Pet\n(Fine-tuning)', 'SST-2\n(Fine-tuning)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('LR-Free vs Baseline: All Tasks Summary',
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, (task, title) in enumerate(zip(tasks, task_titles)):
        ax = axes[idx]

        if task not in best_results:
            continue

        # Get top performers
        sorted_opts = sorted(best_results[task].items(),
                            key=lambda x: x[1]['test_accuracy'],
                            reverse=True)[:8]

        opt_names = []
        accuracies = []
        colors = []

        for opt_family, result in sorted_opts:
            opt_names.append(opt_family)
            accuracies.append(result['test_accuracy'])
            category = categorize_optimizer(opt_family)
            colors.append(get_category_color(category))

        bars = ax.bar(range(len(opt_names)), accuracies, color=colors,
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(opt_names)))
        ax.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=get_category_color('parameter-free'), edgecolor='black',
              label='Learning Rate Free'),
        Patch(facecolor=get_category_color('baseline'), edgecolor='black',
              label='Baseline')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=2, fontsize=11, frameon=True)

    plt.tight_layout()

    filename = f"{save_dir}/summary_lr_free_vs_baseline.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def generate_all_plots():
    """Generate all plots."""
    print("\n" + "="*80)
    print("GENERATING LR-FREE vs BASELINE PLOTS")
    print("="*80 + "\n")

    tasks = ['cifar10', 'oxford_pet', 'sst2']

    for task in tasks:
        print(f"\nTask: {task.upper()}")
        print("-" * 40)
        plot_lr_sensitivity(task)
        plot_best_comparison(task)

    print(f"\nAll Tasks Summary")
    print("-" * 40)
    plot_summary_all_tasks()

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED!")
    print("="*80)


if __name__ == "__main__":
    generate_all_plots()
