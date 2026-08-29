# Repository Directives

## Architecture & Code Standards
- Keep training logic strictly modular (`dataset.py`, `model.py`, `train.py`, `config.yaml`).
- Use `argparse` or YAML configuration files for hyperparameters so they can be overridden easily from notebook cells.
- Maintain a clean `requirements.txt` containing only the packages needed inside the Kaggle environment.

## Execution & Artifact Boundaries
- The local environment has no Python/GPU; write code targeting remote execution on Kaggle.
- Never commit large datasets, `.pth`/`.pt` weights, or output artifacts to Git.