#!/bin/bash

# Run training with pretrained=False
echo "Running: python -m examples.qdho.main --mode train --is_pretrained=False"
python -m examples.qdho.main --mode train --is_pretrained=False
if [ $? -ne 0 ]; then
    echo "Command failed: python -m examples.qdho.main --mode train --is_pretrained=False"
    exit 1
fi

# Run evaluation
echo "Running: python -m examples.qdho.main --mode eval"
python -m examples.qdho.main --mode eval
if [ $? -ne 0 ]; then
    echo "Command failed: python -m examples.qdho.main --mode eval"
    exit 1
fi

echo "All commands executed successfully."
