"""
Analyze early vs late cumulative target probabilities from training dynamics.

For each sample we form a "cumulative target prob" = mean of its true-class
probability over a range of epochs:
    early = mean over epochs [0, early_epochs)
    late  = mean over ALL epochs [0, T)        (the full trajectory)

It then reports:
  1. Correlation between early and late cumulative probs across samples
     (both Spearman rank correlation and Pearson linear correlation).
  2. Jaccard overlap between the *hardest* subsets (lowest cumulative prob,
     i.e. the samples a coreset method would keep) selected by early vs late
     scores, swept over pruning ratios 0.1 - 0.9.

Results are printed and written to <save-path>/early_late_analysis.csv, with
optional scatter / Jaccard-vs-ratio plots saved alongside.

Usage:
    python analyze_target_probs.py \
        --td-path /dir/to/td \
        --task-name imagenet \
        --label-path ./labels.npy \
        --early-epochs 30 \
        --save-path ./results
"""
import os
import csv
import pickle
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt


def load_target_probs(td_path, task_name, label_path):
    """Reconstruct target_probs of shape [num_epochs, num_samples] from the
    pickled training-dynamics files. Mirrors generate_importance_score_imagenet_dual.py."""
    total_result = {}
    for i, _ in enumerate(os.listdir(td_path)):
        fp = os.path.join(td_path, f'td-{task_name}-epoch-{i}.pickle')
        with open(fp, 'rb') as f:
            td_data = pickle.load(f)

        grouped_data = defaultdict(lambda: {'idx': [], 'output': []})
        for entry in td_data['training_dynamics']:
            epoch = entry['epoch']
            grouped_data[epoch]['idx'].append(entry['idx'])
            grouped_data[epoch]['output'].append(entry['output'])

        for epoch, tensors in grouped_data.items():
            total_result[epoch] = {
                'idx': torch.cat(tensors['idx']),
                'output': torch.cat(tensors['output']),
            }

    idxs, outputs = [], []
    for epoch in total_result.keys():
        idxs.append(total_result[epoch]['idx'])
        outputs.append(total_result[epoch]['output'])
    idxs = torch.stack(idxs, dim=0)
    outputs = torch.stack(outputs, dim=0)

    # scatter each epoch's outputs back into sample-index order
    probs_rearranged = []
    for i in range(idxs.shape[0]):
        probs_re = torch.zeros_like(torch.tensor(outputs[i]))
        probs_re = probs_re.index_add(0, idxs[i].type(torch.int64), torch.tensor(outputs[i]))
        probs_rearranged.append(probs_re)
    rearranged = torch.stack(probs_rearranged)
    rearranged = F.softmax(rearranged, dim=-1)

    labels = np.load(label_path)
    labels_t = torch.from_numpy(labels).long().to(rearranged.device)
    labels_expanded = labels_t.view(1, -1, 1).expand(rearranged.size(0), -1, 1)
    target_probs = torch.gather(rearranged, dim=2, index=labels_expanded).squeeze(-1)
    return target_probs  # [T, N]


def hardest_set(scores, ratio):
    """Indices of the lowest-scoring (hardest) `ratio` fraction of samples."""
    n = int(ratio * len(scores))
    return set(np.argsort(scores)[:n].tolist())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser(description='Early vs late cumulative target prob analysis.')
    parser.add_argument('--td-path', type=str, required=True, help='Dir of saved training-dynamics pickles.')
    parser.add_argument('--task-name', type=str, required=True, help='Task name used in the td filenames.')
    parser.add_argument('--label-path', type=str, required=True, help='Path to ground-truth labels (.npy).')
    parser.add_argument('--early-epochs', type=int, default=30, help='Number of early epochs for the early window.')
    parser.add_argument('--save-path', type=str, default='./results', help='Dir to write the CSV/plots.')
    parser.add_argument('--no-plots', action='store_true', help='Skip saving plots.')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    target_probs = load_target_probs(args.td_path, args.task_name, args.label_path)
    T, N = target_probs.shape
    print(f'Loaded target_probs: {T} epochs x {N} samples')

    early_epochs = min(args.early_epochs, T)
    early = target_probs[:early_epochs].mean(dim=0).cpu().numpy()  # mean over first 30
    late = target_probs.mean(dim=0).cpu().numpy()                  # mean over full trajectory

    # --- correlations ---
    sp_corr, sp_p = spearmanr(early, late)
    pe_corr, pe_p = pearsonr(early, late)
    print(f'Early window: epochs [0, {early_epochs}) | Late window: full [0, {T})')
    print(f'Spearman correlation: {sp_corr:.4f} (p={sp_p:.2e})')
    print(f'Pearson  correlation: {pe_corr:.4f} (p={pe_p:.2e})')

    # --- Jaccard sweep over pruning ratios (hardest samples) ---
    ratios = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 .. 0.9
    jaccards = []
    for r in ratios:
        a = hardest_set(early, r)
        b = hardest_set(late, r)
        j = jaccard(a, b)
        jaccards.append(j)
        print(f'ratio={r:.1f}  Jaccard(hardest)={j:.4f}')

    # --- write CSV ---
    csv_path = os.path.join(args.save_path, 'early_late_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_name', args.task_name])
        writer.writerow(['early_epochs', early_epochs])
        writer.writerow(['total_epochs', T])
        writer.writerow(['num_samples', N])
        writer.writerow(['spearman', f'{sp_corr:.6f}'])
        writer.writerow(['pearson', f'{pe_corr:.6f}'])
        writer.writerow([])
        writer.writerow(['ratio', 'jaccard_hardest'])
        for r, j in zip(ratios, jaccards):
            writer.writerow([r, f'{j:.6f}'])
    print(f'Wrote {csv_path}')

    # --- plots ---
    if not args.no_plots:
        # scatter: early vs late cumulative prob
        plt.figure(figsize=(5, 5))
        plt.scatter(early, late, s=3, alpha=0.3)
        plt.xlabel(f'Early cumulative target prob (epochs 0-{early_epochs})')
        plt.ylabel('Late cumulative target prob (full)')
        plt.title(f'Spearman={sp_corr:.3f}, Pearson={pe_corr:.3f}')
        plt.tight_layout()
        scatter_path = os.path.join(args.save_path, 'early_vs_late_scatter.png')
        plt.savefig(scatter_path, dpi=150)
        plt.close()

        # Jaccard vs ratio
        plt.figure(figsize=(5, 4))
        plt.plot(ratios, jaccards, marker='o')
        plt.xlabel('Pruning ratio (fraction kept as hardest)')
        plt.ylabel('Jaccard overlap (early vs late)')
        plt.title('Subset agreement: early vs late selection')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        jac_path = os.path.join(args.save_path, 'jaccard_vs_ratio.png')
        plt.savefig(jac_path, dpi=150)
        plt.close()
        print(f'Wrote {scatter_path} and {jac_path}')


if __name__ == '__main__':
    main()
