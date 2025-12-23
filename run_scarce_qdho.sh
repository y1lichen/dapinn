#!/bin/bash
set -e

# Define the list of sample sizes
sample_sizes=(1 3 5 10 15 100 10000)

# Loop through each sample size
for s in "${sample_sizes[@]}"; do
    echo "============================"
    echo "Running for sample_size=$s"
    echo "============================"

    echo "Running: python -m examples.qdho.main --mode train --is_pretrained=False --finetune_sample_size=$s"
    python -m examples.qdho.main --mode train --is_pretrained=False --finetune_sample_size="$s"
    if [ $? -ne 0 ]; then
        echo "Command failed: train for sample_size=$s"
        exit 1
    fi

    echo "Running: python -m examples.qdho.main --mode eval --is_pretrained=False --finetune_sample_size=$s"
    python -m examples.qdho.main --mode eval --is_pretrained=False --finetune_sample_size="$s"
    if [ $? -ne 0 ]; then
        echo "Command failed: eval for sample_size=$s"
        exit 1
    fi

    # Create output directory
    outdir="scarce/qdho/$s"
    mkdir -p "$outdir"

    # Copy default.py
    cp -f examples/qdho/configs/default.py "$outdir/default.py"

    # Copy results/qdho folder recursively
    cp -r results/qdho "$outdir/qdho"

    echo "Finished processing sample_size=$s"
done

echo "All sample sizes processed successfully."
