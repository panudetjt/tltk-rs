# Segment Oracle Environment Setup

## Background

TLTK 1.10's `nlp.segment` function uses an EDU segmentation model (`sent_segment.pklz`) containing a scikit-learn `RandomForestClassifier` pickled with scikit-learn 1.0.2.

On modern environments (Python 3.12+ with scikit-learn >= 1.2), unpickling this estimator fails because the underlying tree `node` struct dtype changed in scikit-learn 1.2 (`node array from the pickle has an incompatible dtype`).

To unlock the segment oracle without retraining or modifying the upstream model, a period-correct environment is used:
- **Python**: 3.9.x (managed via `mise`)
- **scikit-learn**: 1.1.1
- **numpy**: < 1.24 (e.g. 1.23.5)
- **scipy**: < 1.13 (e.g. 1.12.0, required for gensim 4.3.2)
- **pandas**: 1.5.3
- **gensim**: 4.3.2
- **nltk**, **sklearn-crfsuite**, **requests**
Default venv path: `~/.local/share/tltk-rs/venv-segment-oracle` (persistent, survives /tmp wipes)

## Usage

### Check existing environment (dry run)
```bash
bash scripts/setup_segment_oracle.sh --check
```
This checks Python 3.9, the package versions, and runs a test probe on `segment('แมวกินปลาทะเล')` without modifying the environment.

### Reproduce environment from scratch
```bash
# 1. Ensure Python 3.9 is installed (via mise)
mise install python@3.9

# 2. Run the setup script to create $HOME/.local/share/tltk-rs/venv-segment-oracle with pinned packages
bash scripts/setup_segment_oracle.sh
```

To force recreate an existing environment:
```bash
bash scripts/setup_segment_oracle.sh --force
```

### Running with the oracle
```bash
PYTHONPATH=~/work/my-project/tltk-ref/tltk-110 $HOME/.local/share/tltk-rs/venv-segment-oracle/bin/python your_script.py
```
