"""Entry point: train and compare linear regression variants, tracked in MLflow.

Run from the project root:
    python train.py

The implementation lives in the src/ package:
    src/config.py       constants and paths
    src/data.py         loading, hashing and the train/test split
    src/features.py     preprocessing pipeline
    src/experiments.py  the model variants
    src/tracking.py     MLflow logging and champion registration
    src/pipeline.py     end-to-end orchestration
"""

from src.pipeline import main

if __name__ == "__main__":
    main()
