#!/usr/bin/env python3
"""
Scan result folders from pedagogical_example runs, compute numeric metrics (MAE/MSE)
and write a summary CSV for later plotting.

Usage:
  python3 examples/pedagogical_example/collect_results.py --workdir . --pattern "results/pedagogical_sample*_*" --out results/pedagogical_summary.csv
"""
import argparse
import glob
import os
import re
import csv
import torch
import numpy as np

from examples.pedagogical_example.configs.default import get_config
from examples.pedagogical_example.models import Pedagogical, Corrector
from examples.pedagogical_example.utils import generate_reaction_ode_dataset


def compute_metrics_for_run(workdir, run_subdir):
    run_path = os.path.join(workdir, run_subdir)
    # Determine model type from directory name
    name = os.path.basename(run_subdir)
    model_type = 'DAPINN' if 'DAPINN' in name.upper() else 'PINN'

    # Load config
    config = get_config()
    config.use_corrector = (model_type == 'DAPINN')
    config.workdir = workdir

    # Prepare test data
    params = config.system_pedagogical.system_params
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    t, u, f, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(config.device)
    u_true = sol.y[0]

    # Load finetuned model
    finetune_path = os.path.join(run_path, config.saving.finetune_path, "final_model.pt")
    if not os.path.exists(finetune_path):
        finetune_path = os.path.join(run_path, config.saving.finetune_path, "best_finetuned_model.pt")

    model = Pedagogical(config).to(config.device)
    try:
        model.load_finetuned_model(finetune_path)
    except Exception as e:
        print(f"[Warn] Could not load finetuned model for {run_subdir}: {e}")
        return None

    model.eval()
    with torch.no_grad():
        u_pred = model(t_test).cpu().numpy().ravel()

    mae = float(np.mean(np.abs(u_pred - u_true)))
    mse = float(np.mean((u_pred - u_true)**2))

    row = {
        'run': run_subdir,
        'model': model_type,
        'mae': mae,
        'mse': mse,
    }

    # If DAPINN, try to evaluate corrector discovery performance
    if model_type == 'DAPINN':
        corr = Corrector(config).to(config.device)
        corr_path = os.path.join(run_path, config.saving.corrector_path, "final_corrector.pt")
        if not os.path.exists(corr_path):
            corr_path = os.path.join(run_path, config.saving.corrector_path, "best_corrector.pt")
        try:
            corr.load_corrector_model(corr_path)
            corr.eval()
            # compute s_pred vs s_true (using u_true for reference)
            t_t = t_test.clone().detach()
            t_t.requires_grad = True
            u_model = model(t_t)
            du_model = torch.autograd.grad(u_model, t_t, torch.ones_like(u_model), create_graph=False)[0]
            inputs = torch.cat([u_model, du_model], dim=1)
            s_pred = corr(inputs).cpu().detach().numpy().ravel()

            lam = params['lambda']
            s_true = lam * u_true * (1 - u_true)
            mae_s = float(np.mean(np.abs(s_pred - s_true)))
            mse_s = float(np.mean((s_pred - s_true)**2))
            row.update({'mae_s': mae_s, 'mse_s': mse_s})
        except Exception as e:
            print(f"[Warn] Corrector not available for {run_subdir}: {e}")

    # try to parse sample size from name
    m = re.search(r'sample(\d+)', run_subdir)
    if m:
        row['sample_size'] = int(m.group(1))
    else:
        row['sample_size'] = -1

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='.', help='Workspace root (same as training workdir)')
    parser.add_argument('--pattern', default='results/pedagogical_sample*_*', help='Glob pattern for run subdirs (relative to workdir)')
    parser.add_argument('--out', default='results/pedagogical_summary.csv', help='Output CSV path')
    args = parser.parse_args()

    matches = sorted(glob.glob(os.path.join(args.workdir, args.pattern)))
    # convert to relative run_subdir strings
    run_subdirs = [os.path.relpath(p, args.workdir) for p in matches]

    rows = []
    for sub in run_subdirs:
        print(f"Processing {sub} ...")
        r = compute_metrics_for_run(args.workdir, sub)
        if r is not None:
            rows.append(r)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # Write CSV
    if rows:
        keys = ['run', 'sample_size', 'model', 'mae', 'mse', 'mae_s', 'mse_s']
        # Some rows may lack mae_s/mse_s -> fill with empty
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                out = {k: r.get(k, '') for k in keys}
                writer.writerow(out)

        print(f"Wrote summary CSV to {args.out}")
    else:
        print("No runs processed — check your pattern and paths.")


if __name__ == '__main__':
    main()
