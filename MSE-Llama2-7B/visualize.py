import re
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

LOG_DIR = 'logs'

def parse_log(log_file, dataset=None):
    runs = []
    current = None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        m = re.search(r"'datasetName': '(\w+)'.*'seed': (\d+)", line)
        if m:
            ds, seed = m.group(1), int(m.group(2))
            if dataset and ds != dataset:
                current = None
                continue
            current = {'dataset': ds, 'seed': seed, 'train_losses': [], 'val_metrics': []}
            runs.append(current)
            continue

        if current is None:
            continue

        m = re.search(r'TRAIN-.*>> loss: ([\d.]+)', line)
        if m:
            current['train_losses'].append(float(m.group(1)))
            continue

        m = re.search(r'M: >> (.+)', line)
        if m and i > 0 and 'VAL' in lines[i - 1]:
            metrics = {}
            for kv in re.finditer(r'(\w+): ([\d.eE+\-]+)', m.group(1)):
                try:
                    metrics[kv.group(1)] = float(kv.group(2))
                except ValueError:
                    pass
            current['val_metrics'].append(metrics)

    return runs


def parse_logs(log_dir, dataset=None):
    all_runs = []
    for path in sorted(glob.glob(f'{log_dir}/cmcm_reg-*.log')):
        all_runs.extend(parse_log(path, dataset=dataset))
    return all_runs


def plot_runs(runs, save_path='logs/training_curves.png'):
    if not runs:
        print("No data found in log.")
        return

    by_dataset = defaultdict(list)
    for r in runs:
        by_dataset[r['dataset']].append(r)

    datasets = list(by_dataset.keys())
    n = len(datasets)
    fig = plt.figure(figsize=(7 * n, 8))
    gs = gridspec.GridSpec(2, n, hspace=0.4, wspace=0.35)
    colors = plt.cm.tab10.colors

    for col, ds in enumerate(datasets):
        ds_runs = by_dataset[ds]
        ax_loss = fig.add_subplot(gs[0, col])
        ax_val  = fig.add_subplot(gs[1, col])

        val_keys = []
        for r in ds_runs:
            if r['val_metrics']:
                val_keys = list(r['val_metrics'][0].keys())
                break

        for idx, run in enumerate(ds_runs):
            color = colors[idx % len(colors)]
            label = f"seed {run['seed']}"
            if run['train_losses']:
                ax_loss.plot(run['train_losses'], color=color, label=label)
            if run['val_metrics'] and val_keys:
                key = val_keys[0]
                vals = [m[key] for m in run['val_metrics'] if key in m]
                ax_val.plot(vals, color=color, label=label, marker='o', markersize=3)

        ax_loss.set_title(f'{ds.upper()} — Train Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        val_label = val_keys[0] if val_keys else 'metric'
        ax_val.set_title(f'{ds.upper()} — Val {val_label}')
        ax_val.set_xlabel('Epoch (after warmup)')
        ax_val.set_ylabel(val_label)
        ax_val.legend(fontsize=8)
        ax_val.grid(True, alpha=0.3)

    plt.suptitle('MSE-Adapter Training Curves', fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--save', type=str, default='logs/training_curves_reg.png')
    args = parser.parse_args()

    runs = parse_logs(args.log_dir, dataset=args.dataset)
    print(f"Found {len(runs)} run(s): {[(r['dataset'], r['seed']) for r in runs]}")
    plot_runs(runs, save_path=args.save)
