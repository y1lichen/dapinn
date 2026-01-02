#!/bin/bash

# ============================================================
# Memory Diffusion with History Term
# ∂_t u = D ∂_xx u + ∫_0^t K(t-s) u(x,s) ds
#
# This script mirrors the DHO experiment workflow:
#   1. Finetuning (PINN + Corrector)
#   2. Evaluation
# ============================================================

set -e  # exit immediately if a command fails

# ------------------------------------------------------------
# Pretraining
# ------------------------------------------------------------
echo "Running: python3 -m examples.viscoelasticity.main --mode train --is_pretrained=True"
python3 -m examples.viscoelasticity.main --mode train --is_pretrained=True

# ------------------------------------------------------------
# Finetuning (with pointwise corrector)
# ------------------------------------------------------------
echo "Running: python3 -m examples.viscoelasticity.main --mode train --is_pretrained=False"
python3 -m examples.viscoelasticity.main --mode train --is_pretrained=False

if [ $? -ne 0 ]; then
    echo "Command failed: python3 -m examples.viscoelasticity.main --mode train --is_pretrained=False"
    exit 1
fi

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
echo "Running: python3 -m examples.viscoelasticity.main --mode eval"
python3 -m examples.viscoelasticity.main --mode eval

if [ $? -ne 0 ]; then
    echo "Command failed: python3 -m examples.viscoelasticity.main --mode eval"
    exit 1
fi

echo "All memory diffusion experiments executed successfully."
