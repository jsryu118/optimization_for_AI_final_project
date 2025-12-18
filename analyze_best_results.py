"""
Best result analysis for report generation.
각 optimizer의 최적 learning rate에서의 성능을 추출하고 분석.
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

    # Extract task (handle oxford_pet, sst2, cifar10)
    if filename.startswith('oxford_pet'):
        task = 'oxford_pet'
        parts = parts[2:]  # Remove 'oxford' and 'pet'
    elif filename.startswith('sst2'):
        task = 'sst2'
        parts = parts[1:]  # Remove 'sst2'
    elif filename.startswith('cifar10'):
        task = 'cifar10'
        parts = parts[1:]  # Remove 'cifar10'
    else:
        task = parts[0]
        parts = parts[1:]

    # Extract optimizer info
    if 'schedulefree' in filename:
        # Find schedulefree index in remaining parts
        sf_idx = parts.index('schedulefree')
        opt_type = parts[sf_idx + 1]
        optimizer_family = f"schedulefree_{opt_type}"
        lr_part = parts[-1]
        lr = float(lr_part[2:]) if lr_part.startswith('lr') else None
    elif any(opt in parts for opt in ['dog', 'ldog', 'tdog', 'prodigy']):
        # Parameter-free
        for opt in ['dog', 'ldog', 'tdog', 'prodigy']:
            if opt in parts:
                optimizer_family = opt
                lr_part = parts[-1]
                lr = float(lr_part[2:]) if lr_part.startswith('lr') else 1.0
                break
    else:
        # Baseline: sgd, adam, adamw
        opt_type = parts[0]  # First part after task name
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
    """Load all results grouped by task and optimizer family."""
    files = glob.glob("results/*.json")

    # Group by (task, optimizer_family)
    grouped = defaultdict(list)

    for filepath in files:
        result = parse_result_file(filepath)
        key = (result['task'], result['optimizer_family'])
        grouped[key].append(result)

    return grouped


def find_best_lr(results_list):
    """Find the result with best test accuracy from a list."""
    return max(results_list, key=lambda x: x['test_accuracy'])


def categorize_optimizer(opt_family):
    """Categorize optimizer into one of three groups."""
    if opt_family in ['dog', 'ldog', 'tdog', 'prodigy']:
        return 'parameter-free'
    elif 'schedulefree' in opt_family:
        return 'schedulefree'
    else:
        return 'baseline'


def get_best_results_by_task():
    """
    Get best result for each optimizer family, grouped by task.

    Returns:
        dict: {task: {optimizer_family: best_result}}
    """
    all_results = get_all_results()
    best_by_task = defaultdict(dict)

    for (task, opt_family), results in all_results.items():
        best = find_best_lr(results)
        best_by_task[task][opt_family] = best

    return best_by_task


def get_lr_sensitivity_data(task):
    """
    Get all results for a task, grouped by optimizer family.
    Used for LR sensitivity plots.

    Returns:
        dict: {optimizer_family: [results sorted by lr]}
    """
    all_results = get_all_results()
    lr_data = defaultdict(list)

    for (t, opt_family), results in all_results.items():
        if t == task:
            # Sort by learning rate
            sorted_results = sorted(results, key=lambda x: x['lr'])
            lr_data[opt_family] = sorted_results

    return lr_data


def print_summary():
    """Print a summary of best results."""
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
        for category in ['parameter-free', 'schedulefree', 'baseline']:
            if category not in by_category:
                continue

            print(f"\n{category.upper()}:")
            print("-" * 80)

            # Sort by test accuracy
            sorted_opts = sorted(by_category[category],
                                key=lambda x: x[1]['test_accuracy'],
                                reverse=True)

            for opt_family, result in sorted_opts:
                lr = result['lr']
                acc = result['test_accuracy']
                time = result['training_time'] / 60  # minutes

                print(f"  {opt_family:20s} | LR={lr:8.5f} | Acc={acc:6.2f}% | Time={time:6.1f}min")

        # Find overall best
        all_opts = [(opt, res) for opt, res in best_results[task].items()]
        best_opt, best_res = max(all_opts, key=lambda x: x[1]['test_accuracy'])

        print(f"\n{'*' * 80}")
        print(f"BEST: {best_opt} | LR={best_res['lr']} | Acc={best_res['test_accuracy']:.2f}%")
        print(f"{'*' * 80}")


def export_best_results_json():
    """Export best results to a JSON file for easy access."""
    best_results = get_best_results_by_task()

    # Convert to serializable format
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

    with open('best_results_summary.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n✓ Best results exported to: best_results_summary.json")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BEST RESULTS ANALYSIS")
    print("="*80)

    print_summary()
    export_best_results_json()
