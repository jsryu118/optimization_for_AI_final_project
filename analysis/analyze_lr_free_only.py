"""
LR-Free vs Baseline comparison analysis (ScheduleFree 제외).
"""
import json
import glob
from pathlib import Path
from collections import defaultdict


def parse_result_file(filepath):
    """Parse a single result file."""
    with open(filepath) as f:
        data = json.load(f)

    filename = Path(filepath).stem
    parts = filename.split('_')

    # Extract task
    if filename.startswith('oxford_pet'):
        task = 'oxford_pet'
        parts = parts[2:]
    elif filename.startswith('sst2'):
        task = 'sst2'
        parts = parts[1:]
    elif filename.startswith('cifar10'):
        task = 'cifar10'
        parts = parts[1:]
    else:
        task = parts[0]
        parts = parts[1:]

    # Skip ScheduleFree results
    if 'schedulefree' in filename:
        return None

    # Extract optimizer info
    if any(opt in parts for opt in ['dog', 'ldog', 'tdog', 'prodigy']):
        # Parameter-free
        for opt in ['dog', 'ldog', 'tdog', 'prodigy']:
            if opt in parts:
                optimizer_family = opt
                lr_part = parts[-1]
                lr = float(lr_part[2:]) if lr_part.startswith('lr') else 1.0
                break
    else:
        # Baseline: sgd, adam, adamw
        opt_type = parts[0]
        optimizer_family = opt_type
        lr_part = parts[-1]
        lr = float(lr_part[2:]) if lr_part.startswith('lr') else None

    return {
        'task': task,
        'optimizer_family': optimizer_family,
        'lr': lr,
        'test_accuracy': data['test_accuracy'],
        'training_time': data['training_time'],
        'train_losses': data['train_losses'],
        'val_accuracies': data['val_accuracies']
    }


def get_all_results():
    """Load all non-schedulefree results."""
    files = glob.glob("results/*.json")
    grouped = defaultdict(list)

    for filepath in files:
        result = parse_result_file(filepath)
        if result is None:  # Skip ScheduleFree
            continue

        key = (result['task'], result['optimizer_family'])
        grouped[key].append(result)

    return grouped


def find_best_lr(results_list):
    """Find the result with best test accuracy."""
    return max(results_list, key=lambda x: x['test_accuracy'])


def categorize_optimizer(opt_family):
    """Categorize optimizer into baseline or parameter-free."""
    if opt_family in ['dog', 'ldog', 'tdog', 'prodigy']:
        return 'parameter-free'
    else:
        return 'baseline'


def get_best_results_by_task():
    """Get best result for each optimizer family, grouped by task."""
    all_results = get_all_results()
    best_by_task = defaultdict(dict)

    for (task, opt_family), results in all_results.items():
        best = find_best_lr(results)
        best_by_task[task][opt_family] = best

    return best_by_task


def print_summary():
    """Print summary of best results (excluding ScheduleFree)."""
    best_results = get_best_results_by_task()
    tasks = ['cifar10', 'oxford_pet', 'sst2']

    for task in tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task.upper()}")
        print('='*80)

        if task not in best_results:
            print("No results found.")
            continue

        # Group by category
        by_category = defaultdict(list)
        for opt_family, result in best_results[task].items():
            category = categorize_optimizer(opt_family)
            by_category[category].append((opt_family, result))

        # Print by category
        for category in ['parameter-free', 'baseline']:
            if category not in by_category:
                continue

            print(f"\n{category.upper()}:")
            print("-" * 80)

            sorted_opts = sorted(by_category[category],
                                key=lambda x: x[1]['test_accuracy'],
                                reverse=True)

            for opt_family, result in sorted_opts:
                lr = result['lr']
                acc = result['test_accuracy']
                time = result['training_time'] / 60

                print(f"  {opt_family:15s} | LR={lr:8.5f} | Acc={acc:6.2f}% | Time={time:6.1f}min")

        # Find overall best
        all_opts = [(opt, res) for opt, res in best_results[task].items()]
        best_opt, best_res = max(all_opts, key=lambda x: x[1]['test_accuracy'])

        print(f"\n{'*' * 80}")
        print(f"BEST: {best_opt} | LR={best_res['lr']} | Acc={best_res['test_accuracy']:.2f}%")
        print(f"Category: {categorize_optimizer(best_opt).upper()}")
        print(f"{'*' * 80}")


def export_best_results_json():
    """Export best results to JSON (excluding ScheduleFree)."""
    best_results = get_best_results_by_task()

    output = {}
    for task, optimizers in best_results.items():
        output[task] = {}
        for opt_family, result in optimizers.items():
            output[task][opt_family] = {
                'lr': result['lr'],
                'test_accuracy': result['test_accuracy'],
                'training_time': result['training_time'],
                'category': categorize_optimizer(opt_family)
            }

    with open('best_results_lr_free_only.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n✓ Best results exported to: best_results_lr_free_only.json")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LR-FREE vs BASELINE COMPARISON (ScheduleFree 제외)")
    print("="*80)

    print_summary()
    export_best_results_json()
