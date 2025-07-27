# ChemTransformer

This project contains utilities for preprocessing chemical reaction data and training a transformer-based model.

## Directory structure

- `config/` - YAML configuration files
- `data/` - dataset files (add `USPTO_480K` here)
- `models/` - model implementations
- `preprocessing/` - data preparation scripts
- `train/` - training utilities
- `utils/` - helper functions and dataset classes
- `scripts/` - shell scripts for common tasks

## Example usage

```bash
# preprocess data
sh scripts/preprocess.sh

# train model
sh scripts/train_g2s.sh
```
