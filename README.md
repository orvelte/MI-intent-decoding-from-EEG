# EEG Motor Imagery Intent Classification

EEG-based motor imagery intent classification using both classical machine learning approaches (CSP+LDA, Riemannian geometry) and deep learning (EEGNet).

## Structure

- `data/` - Raw and processed EEG data (gitignored)
- `src/` - Main source code
  - `data.py` - Data loaders, epoching, and train/test splits
  - `preprocess.py` - Filtering and referencing
  - `features.py` - CSP and Riemannian geometry utilities
  - `models_classic.py` - Classical ML models (LDA, SVM)
  - `models_dl.py` - Deep learning models (EEGNet)
  - `train_classic.py` - CLI for training classical models
  - `train_dl.py` - CLI for training deep learning models
  - `eval.py` - Evaluation metrics, confusion matrices, per-subject statistics
- `notebooks/` - Exploratory data analysis and figure generation
- `configs/` - YAML configuration files for bands, windows, and model parameters
- `reports/` - Generated reports and results

## Setup

1. Create conda environment: `conda env create -f environment.yml`
2. Activate environment: `conda activate eeg-mi-intent`
3. Place your EEG data in the `data/` directory

## Usage

### Classical ML Training
```bash
python src/train_classic.py --config configs/csp_lda.yaml
```

### Deep Learning Training
```bash
python src/train_dl.py --config configs/eegnet.yaml
```

### Evaluation
```bash
python src/eval.py --model_path models/best_model.pkl
```
