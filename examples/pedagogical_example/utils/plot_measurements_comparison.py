#!/usr/bin/env python3
"""
Plot comparison of PINN vs DAPINN metrics from summary CSV.

Usage:
  python3 examples/pedagogical_example/plot_comparison.py --csv results/pedagogical_summary.csv --outdir results/plots
"""
import argparse
import os
import csv
import collections
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Summary CSV produced by collect_results.py')
    parser.add_argument('--outdir', default='results/plots', help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = read_csv(args.csv)
    # group by model type
    by_model = collections.defaultdict(list)
    for r in rows:
        try:
            ss = int(r.get('sample_size', -1))
            mae = float(r.get('mae', 'nan'))
        except Exception:
            continue
        by_model[r['model']].append((ss, mae))

    plt.figure()
    for model, vals in by_model.items():
        vals = sorted(vals, key=lambda x: x[0])
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        plt.plot(xs, ys, marker='o', label=model)

    plt.xlabel('Number of Measurements')
    plt.ylabel('MAE on Test Trajectory')
    plt.title('PINN vs DAPINN — MAE vs Measurements')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(args.outdir, 'mae_vs_measurements.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
