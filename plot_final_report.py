"""
Final report visualization: LR-Free vs Baseline detailed comparison.
각 task별로 5개 그래프 생성.
"""
import json
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

matplotlib.use('Agg')

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')


def load_all_results(task):
    """Load all results for a specific task (excluding ScheduleFree)."""
    pattern = f"results/{task}_*.json"
    files = glob.glob(pattern)

    results = defaultdict(list)

    for filepath in files:
        filename = Path(filepath).stem

        # Skip ScheduleFree
        if 'schedulefree' in filename:
            continue

        with open(filepath) as f:
            data = json.load(f)

        # Parse optimizer type
        parts = filename.replace(f'{task}_', '').split('_')

        if any(opt in parts for opt in ['dog', 'ldog', 'tdog', 'prodigy']):
            # LR-Free
            for opt in ['dog', 'ldog', 'tdog', 'prodigy']:
                if opt in parts:
                    opt_type = opt
                    break
        else:
            # Baseline
            opt_type = parts[0]  # sgd, adam, adamw

        # Extract LR
        lr_part = parts[-1]
        lr = float(lr_part[2:]) if lr_part.startswith('lr') else 1.0

        results[opt_type].append({
            'lr': lr,
            'test_accuracy': data['test_accuracy'],
            'training_time': data['training_time'],
            'train_losses': data['train_losses'],
            'val_accuracies': data['val_accuracies']
        })

    return results


def plot_baseline_learning_curves(task, optimizer, save_dir="final_report"):
    """
    Plot learning curves for all LRs of a baseline optimizer.

    Args:
        task: cifar10, oxford_pet, or sst2
        optimizer: sgd, adam, or adamw
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = load_all_results(task)

    if optimizer not in results:
        print(f"No results for {task} - {optimizer}")
        return

    data_list = sorted(results[optimizer], key=lambda x: x['lr'])

    if len(data_list) == 0:
        print(f"No data for {task} - {optimizer}")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{task.upper()} - {optimizer.upper()} Learning Curves (All LRs)',
                 fontsize=15, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))

    for idx, data in enumerate(data_list):
        lr = data['lr']
        epochs = [e['epoch'] for e in data['train_losses']]
        train_losses = [e['loss'] for e in data['train_losses']]
        val_accs = [e['accuracy'] for e in data['val_accuracies']]

        label = f"LR={lr}"

        # Training loss
        ax1.plot(epochs, train_losses, 'o-', color=colors[idx],
                label=label, linewidth=2, markersize=5, alpha=0.8)

        # Validation accuracy
        ax2.plot(epochs, val_accs, 'o-', color=colors[idx],
                label=label, linewidth=2, markersize=5, alpha=0.8)

    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Training Loss', fontweight='bold', fontsize=11)
    ax1.set_title('Training Loss', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')

    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=11)
    ax2.set_title('Validation Accuracy', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')

    plt.tight_layout()

    filename = f"{save_dir}/{task}_1_{optimizer}_all_lr_curves.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_best_vs_lrfree_curves(task, save_dir="final_report"):
    """
    Plot learning curves: Best baseline vs all LR-Free optimizers.

    Args:
        task: cifar10, oxford_pet, or sst2
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = load_all_results(task)

    # Find best baseline for each optimizer
    best_baseline = {}
    for opt in ['sgd', 'adam', 'adamw']:
        if opt in results:
            best = max(results[opt], key=lambda x: x['test_accuracy'])
            best_baseline[opt] = best

    # Get all LR-Free results
    lr_free_opts = ['dog', 'ldog', 'tdog', 'prodigy']
    lr_free_data = {}
    for opt in lr_free_opts:
        if opt in results and len(results[opt]) > 0:
            lr_free_data[opt] = results[opt][0]  # Only one LR for LR-Free

    if not best_baseline or not lr_free_data:
        print(f"Insufficient data for {task}")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{task.upper()} - Best Baseline vs Learning Rate Free',
                 fontsize=15, fontweight='bold')

    # Colors
    baseline_colors = {'sgd': '#3498db', 'adam': '#e74c3c', 'adamw': '#f39c12'}
    lrfree_colors = {'dog': '#2ecc71', 'ldog': '#27ae60', 'tdog': '#16a085', 'prodigy': '#1abc9c'}

    # Plot baseline
    for opt, data in best_baseline.items():
        epochs = [e['epoch'] for e in data['train_losses']]
        train_losses = [e['loss'] for e in data['train_losses']]
        val_accs = [e['accuracy'] for e in data['val_accuracies']]

        label = f"{opt.upper()} (LR={data['lr']})"

        ax1.plot(epochs, train_losses, 'o-', color=baseline_colors[opt],
                label=label, linewidth=2.5, markersize=6, alpha=0.8)
        ax2.plot(epochs, val_accs, 'o-', color=baseline_colors[opt],
                label=label, linewidth=2.5, markersize=6, alpha=0.8)

    # Plot LR-Free
    for opt, data in lr_free_data.items():
        epochs = [e['epoch'] for e in data['train_losses']]
        train_losses = [e['loss'] for e in data['train_losses']]
        val_accs = [e['accuracy'] for e in data['val_accuracies']]

        label = f"{opt.upper()} (LR-Free)"

        ax1.plot(epochs, train_losses, 's--', color=lrfree_colors[opt],
                label=label, linewidth=2.5, markersize=6, alpha=0.8)
        ax2.plot(epochs, val_accs, 's--', color=lrfree_colors[opt],
                label=label, linewidth=2.5, markersize=6, alpha=0.8)

    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Training Loss', fontweight='bold', fontsize=11)
    ax1.set_title('Training Loss', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')

    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=11)
    ax2.set_title('Validation Accuracy', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')

    plt.tight_layout()

    filename = f"{save_dir}/{task}_4_best_vs_lrfree_curves.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_training_time_comparison(task, save_dir="final_report"):
    """
    Plot training time comparison: Best baseline vs LR-Free.

    Args:
        task: cifar10, oxford_pet, or sst2
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = load_all_results(task)

    # Find best baseline
    best_baseline = {}
    for opt in ['sgd', 'adam', 'adamw']:
        if opt in results:
            best = max(results[opt], key=lambda x: x['test_accuracy'])
            best_baseline[opt] = best

    # Get LR-Free
    lr_free_opts = ['dog', 'ldog', 'tdog', 'prodigy']
    lr_free_data = {}
    for opt in lr_free_opts:
        if opt in results and len(results[opt]) > 0:
            lr_free_data[opt] = results[opt][0]

    if not best_baseline or not lr_free_data:
        print(f"Insufficient data for {task}")
        return

    # Prepare data
    all_data = []
    categories = []
    colors = []

    baseline_colors = {'sgd': '#3498db', 'adam': '#e74c3c', 'adamw': '#f39c12'}
    lrfree_color = '#2ecc71'

    # Baseline
    for opt, data in sorted(best_baseline.items()):
        all_data.append({
            'name': f"{opt.upper()}\n(LR={data['lr']})",
            'time': data['training_time'] / 60,
            'acc': data['test_accuracy']
        })
        categories.append('Baseline')
        colors.append(baseline_colors[opt])

    # LR-Free
    for opt, data in sorted(lr_free_data.items()):
        all_data.append({
            'name': opt.upper(),
            'time': data['training_time'] / 60,
            'acc': data['test_accuracy']
        })
        categories.append('LR-Free')
        colors.append(lrfree_color)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [d['name'] for d in all_data]
    times = [d['time'] for d in all_data]

    bars = ax.barh(range(len(names)), times, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, time, data) in enumerate(zip(bars, times, all_data)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {time:.1f}min ({data["acc"]:.2f}%)',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Training Time (minutes)', fontweight='bold', fontsize=12)
    ax.set_title(f'{task.upper()} - Training Time Comparison (Best Configurations)',
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Add separator line
    if best_baseline and lr_free_data:
        sep_y = len(best_baseline) - 0.5
        ax.axhline(sep_y, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Baseline (Tuned LR)'),
        Patch(facecolor=lrfree_color, edgecolor='black', label='Learning Rate Free')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()

    filename = f"{save_dir}/{task}_5_training_time_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_test_accuracy_bar(task, save_dir="final_report"):
    """
    Plot test accuracy bar chart: Best baseline vs LR-Free.

    Args:
        task: cifar10, oxford_pet, or sst2
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = load_all_results(task)

    # Find best baseline
    best_baseline = {}
    for opt in ['sgd', 'adam', 'adamw']:
        if opt in results:
            best = max(results[opt], key=lambda x: x['test_accuracy'])
            best_baseline[opt] = best

    # Get LR-Free
    lr_free_opts = ['dog', 'ldog', 'tdog', 'prodigy']
    lr_free_data = {}
    for opt in lr_free_opts:
        if opt in results and len(results[opt]) > 0:
            lr_free_data[opt] = results[opt][0]

    if not best_baseline and not lr_free_data:
        print(f"Insufficient data for {task}")
        return

    # Prepare data
    all_data = []
    categories = []
    colors = []

    baseline_colors = {'sgd': '#3498db', 'adam': '#e74c3c', 'adamw': '#f39c12'}
    lrfree_color = '#2ecc71'

    # Baseline
    for opt in ['sgd', 'adam', 'adamw']:
        if opt in best_baseline:
            data = best_baseline[opt]
            all_data.append({
                'name': opt.upper(),
                'acc': data['test_accuracy'],
                'lr': data['lr']
            })
            categories.append('Baseline')
            colors.append(baseline_colors[opt])

    # LR-Free
    for opt in ['dog', 'ldog', 'tdog', 'prodigy']:
        if opt in lr_free_data:
            data = lr_free_data[opt]
            all_data.append({
                'name': opt.upper(),
                'acc': data['test_accuracy'],
                'lr': data['lr']
            })
            categories.append('LR-Free')
            colors.append(lrfree_color)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [d['name'] for d in all_data]
    accs = [d['acc'] for d in all_data]

    bars = ax.bar(range(len(names)), accs, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, acc, data) in enumerate(zip(bars, accs, all_data)):
        height = bar.get_height()
        lr_text = f"LR={data['lr']}" if data['lr'] != 1.0 else "LR-Free"
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%\n({lr_text})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'{task.upper()} - Test Accuracy Comparison',
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add separator line
    if best_baseline and lr_free_data:
        sep_x = len(best_baseline) - 0.5
        ax.axvline(sep_x, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Baseline (Tuned LR)'),
        Patch(facecolor=lrfree_color, edgecolor='black', label='Learning Rate Free')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    # Set y-axis range with some padding
    y_min = min(accs) - 2
    y_max = max(accs) + 5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    filename = f"{save_dir}/{task}_2_test_accuracy_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def generate_all_task_plots(task):
    """Generate all 6 plots for a task."""
    print(f"\n{'='*80}")
    print(f"Generating plots for: {task.upper()}")
    print(f"{'='*80}\n")

    # 1-3: Baseline learning curves
    for optimizer in ['sgd', 'adam', 'adamw']:
        print(f"  Plot {optimizer.upper()} all LR curves...")
        plot_baseline_learning_curves(task, optimizer)

    # 2: Test accuracy bar chart
    print(f"  Plot test accuracy comparison...")
    plot_test_accuracy_bar(task)

    # 4: Best vs LR-Free curves
    print(f"  Plot Best vs LR-Free curves...")
    plot_best_vs_lrfree_curves(task)

    # 5: Training time comparison
    print(f"  Plot training time comparison...")
    plot_training_time_comparison(task)

    print(f"\n✓ All plots generated for {task.upper()}")


def main():
    """Generate all plots for all tasks."""
    print("\n" + "="*80)
    print("FINAL REPORT VISUALIZATION")
    print("="*80)

    tasks = ['cifar10', 'oxford_pet', 'sst2']

    for task in tasks:
        generate_all_task_plots(task)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: final_report/")
    print(f"  - Each task has 6 plots:")
    print(f"    1. SGD all LR curves")
    print(f"    2. Test accuracy comparison (bar chart)")
    print(f"    3. Adam all LR curves")
    print(f"    4. AdamW all LR curves")
    print(f"    5. Best baseline vs LR-Free curves")
    print(f"    6. Training time comparison")
    print(f"\nTotal: 18 plots (6 per task × 3 tasks)")
    print("="*80)


if __name__ == "__main__":
    main()
