"""
Analyze scheduler experiments.
Scheduler 방식 (10+20+30+40+50) vs ScheduleFree 방식 (50 with checkpoints) 비교.
"""
import json
import glob
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

matplotlib.use('Agg')

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')


def load_scheduler_results():
    """Load scheduler experiment results (separate training runs)."""
    results = {}

    # Load summary files
    pattern = "scheduler_experiments/*_scheduler_summary.json"
    files = glob.glob(pattern)

    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)

        # Extract optimizer name from filename
        filename = Path(filepath).stem
        optimizer = filename.replace('_scheduler_summary', '')

        results[optimizer] = data

    return results


def load_schedulefree_results():
    """Load ScheduleFree experiment results (single run with checkpoints)."""
    results = {}

    pattern = "scheduler_experiments/cifar10_schedulefree_*_50epochs_results.json"
    files = glob.glob(pattern)

    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)

        # Extract optimizer type
        filename = Path(filepath).stem
        # cifar10_schedulefree_sgd_50epochs_results
        parts = filename.split('_')
        opt_type = parts[2]  # sgd, adam, or adamw

        # Extract checkpoint results
        checkpoint_data = {}
        for cp in data['checkpoint_results']:
            epoch = cp['epoch']
            checkpoint_data[f"{epoch}_epochs"] = {
                'epochs': epoch,
                'test_accuracy': cp['test_accuracy'],
                'training_time': cp['training_time'] if 'training_time' in cp else data['training_time']
            }

        results[f"schedulefree_{opt_type}"] = checkpoint_data

    return results


def plot_comparison(save_dir="scheduler_experiments"):
    """
    Plot scheduler vs schedulefree comparison.
    """
    scheduler_results = load_scheduler_results()
    schedulefree_results = load_schedulefree_results()

    if not scheduler_results or not schedulefree_results:
        print("Missing results!")
        return

    optimizer_types = ['sgd', 'adam', 'adamw']
    epoch_points = [10, 20, 30, 40, 50]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Scheduler vs ScheduleFree Comparison - CIFAR-10',
                 fontsize=16, fontweight='bold')

    colors = {
        'sgd': '#3498db',
        'adam': '#e74c3c',
        'adamw': '#f39c12'
    }

    for idx, opt_type in enumerate(optimizer_types):
        ax = axes[idx]

        # Scheduler results (separate runs)
        if opt_type in scheduler_results:
            data = scheduler_results[opt_type]
            epochs = []
            accs = []

            for epoch in epoch_points:
                key = f"{epoch}_epochs"
                if key in data:
                    epochs.append(epoch)
                    accs.append(data[key]['test_accuracy'])

            ax.plot(epochs, accs, 'o-', color=colors[opt_type],
                   linewidth=2.5, markersize=10,
                   label=f'{opt_type.upper()} + Scheduler\n(별도 학습)',
                   markeredgecolor='black', markeredgewidth=1.5)

        # ScheduleFree results (single run with checkpoints)
        sf_key = f"schedulefree_{opt_type}"
        if sf_key in schedulefree_results:
            data = schedulefree_results[sf_key]
            epochs = []
            accs = []

            for epoch in epoch_points:
                key = f"{epoch}_epochs"
                if key in data:
                    epochs.append(epoch)
                    accs.append(data[key]['test_accuracy'])

            ax.plot(epochs, accs, 's--', color='#9b59b6',
                   linewidth=2.5, markersize=10,
                   label=f'ScheduleFree-{opt_type.upper()}\n(50 epoch 학습)',
                   markeredgecolor='black', markeredgewidth=1.5)

        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'{opt_type.upper()}', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks(epoch_points)

    plt.tight_layout()

    filename = f"{save_dir}/scheduler_vs_schedulefree.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_training_cost(save_dir="scheduler_experiments"):
    """
    Plot cumulative training time comparison.
    """
    scheduler_results = load_scheduler_results()
    schedulefree_results = load_schedulefree_results()

    optimizer_types = ['sgd', 'adam', 'adamw']
    epoch_points = [10, 20, 30, 40, 50]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'sgd': '#3498db',
        'adam': '#e74c3c',
        'adamw': '#f39c12'
    }

    for opt_type in optimizer_types:
        # Scheduler: cumulative time (sum of all separate runs)
        if opt_type in scheduler_results:
            data = scheduler_results[opt_type]
            cumulative_times = []
            total_time = 0

            for epoch in epoch_points:
                key = f"{epoch}_epochs"
                if key in data:
                    total_time += data[key]['training_time'] / 60  # minutes
                    cumulative_times.append(total_time)

            ax.plot(epoch_points[:len(cumulative_times)], cumulative_times,
                   'o-', color=colors[opt_type], linewidth=2.5, markersize=10,
                   label=f'{opt_type.upper()} + Scheduler (누적)',
                   markeredgecolor='black', markeredgewidth=1.5)

        # ScheduleFree: single run time (approximately linear)
        sf_key = f"schedulefree_{opt_type}"
        if sf_key in schedulefree_results:
            # Get final time at epoch 50
            if "50_epochs" in schedulefree_results[sf_key]:
                total_time = schedulefree_results[sf_key]["50_epochs"]["training_time"] / 60

                # Estimate time at each checkpoint (linear assumption)
                times_at_checkpoints = [total_time * (e / 50) for e in epoch_points]

                ax.plot(epoch_points, times_at_checkpoints,
                       's--', color='#9b59b6', linewidth=2.5, markersize=10,
                       label=f'ScheduleFree-{opt_type.upper()} (단일 학습)',
                       markeredgecolor='black', markeredgewidth=1.5, alpha=0.7)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Training Time (minutes)', fontweight='bold', fontsize=12)
    ax.set_title('Training Time Comparison (Cumulative)',
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xticks(epoch_points)

    plt.tight_layout()

    filename = f"{save_dir}/training_time_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def print_summary():
    """Print text summary."""
    scheduler_results = load_scheduler_results()
    schedulefree_results = load_schedulefree_results()

    print("\n" + "="*80)
    print("SCHEDULER vs SCHEDULEFREE COMPARISON - SUMMARY")
    print("="*80 + "\n")

    optimizer_types = ['sgd', 'adam', 'adamw']
    epoch_points = [10, 20, 30, 40, 50]

    for opt_type in optimizer_types:
        print(f"{opt_type.upper()}:")
        print("-" * 80)

        # Scheduler results
        if opt_type in scheduler_results:
            print(f"\n  Scheduler 방식 (별도 학습):")
            data = scheduler_results[opt_type]
            total_time = 0

            for epoch in epoch_points:
                key = f"{epoch}_epochs"
                if key in data:
                    acc = data[key]['test_accuracy']
                    time = data[key]['training_time'] / 60
                    total_time += time
                    print(f"    {epoch:2d} epochs: {acc:6.2f}% (학습시간: {time:5.1f}분)")

            print(f"    총 학습시간: {total_time:.1f}분")

        # ScheduleFree results
        sf_key = f"schedulefree_{opt_type}"
        if sf_key in schedulefree_results:
            print(f"\n  ScheduleFree 방식 (50 epoch 단일 학습):")
            data = schedulefree_results[sf_key]

            for epoch in epoch_points:
                key = f"{epoch}_epochs"
                if key in data:
                    acc = data[key]['test_accuracy']
                    print(f"    {epoch:2d} epochs: {acc:6.2f}%")

            if "50_epochs" in data:
                total_time = data["50_epochs"]["training_time"] / 60
                print(f"    총 학습시간: {total_time:.1f}분")

        # Comparison at epoch 50
        if opt_type in scheduler_results and sf_key in schedulefree_results:
            if "50_epochs" in scheduler_results[opt_type] and "50_epochs" in schedulefree_results[sf_key]:
                sched_acc = scheduler_results[opt_type]["50_epochs"]["test_accuracy"]
                sf_acc = schedulefree_results[sf_key]["50_epochs"]["test_accuracy"]
                diff = sched_acc - sf_acc

                print(f"\n  Epoch 50 비교:")
                print(f"    성능 차이: {diff:+.2f}% (Scheduler - ScheduleFree)")

        print("\n")


def generate_all_plots():
    """Generate all plots and summary."""
    print("\n" + "="*80)
    print("ANALYZING SCHEDULER EXPERIMENTS")
    print("="*80)

    plot_comparison()
    plot_training_cost()
    print_summary()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    generate_all_plots()
