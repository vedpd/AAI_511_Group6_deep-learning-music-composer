# Composer Classification using Deep Learning

**University of San Diego – MS in Applied Artificial Intelligence**  
**AAI 511 Deep Learning Final Project — Group 6**

## Project Overview

This project implements a complete deep learning pipeline to classify the composer of classical piano pieces from MIDI scores. The system explores three progressively more powerful architectures:

- **LSTM (Long Short-Term Memory)**: Processes sequential note-level features (baseline)
- **CNN (Convolutional Neural Network)**: Analyzes piano roll representations (baseline)
- **CRNN (CNN + LSTM Hybrid)**: Combines spatial and temporal modeling for superior performance (improved)

### Target Composers

- **Bach** (~1,024 files) — Dense counterpoint, fugues, 4-voice textures
- **Beethoven** (~213 files) — Dramatic dynamics, structural innovation
- **Chopin** (~136 files) — Lyrical melody over arpeggiated accompaniment
- **Mozart** (~257 files) — Balanced classical form, Alberti bass patterns

---

## Dataset

**Source**: [MIDI Classic Music (Kaggle)](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)

The dataset contains MIDI files from various classical composers. This project filters and uses only the four composers listed above.

**Data Location**: `Data/midiclassics/{Composer}/`

**Dataset Statistics**:
- Total composers: 4
- MIDI files per composer: ~500-1000 (varies)
- File format: `.mid` (MIDI)

---

## Repository Structure

```
composer-classification/
├── README.md                          # This file
├── Readme_1.md                        # Detailed requirements specification
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── Data/
│   └── midiclassics/                  # Kaggle MIDI dataset
│       ├── Bach/
│       ├── Beethoven/
│       ├── Chopin/
│       └── Mozart/
│
├── data/                              # Generated data (created at runtime)
│   ├── raw/                           # Symlink to Data/midiclassics/
│   ├── interim/                       # Cleaned and split metadata
│   ├── processed/                     # Processed datasets
│   └── features/                      # Extracted features (LSTM/CNN)
│
├── models/                            # Model artifacts
│   ├── checkpoints/                   # Training checkpoints
│   └── saved_models/                  # Final trained models
│
├── reports/                           # Submitted reports & generated outputs
│   ├── Composer_identification_Project_AAI_511_Group6.pdf   # Final project report (PDF)
│   ├── Composer_Identification_Project_Report_AAI_511_Group6.docx  # Final report (Word)
│   ├── figures/                       # Plots and visualizations
│   └── tables/                        # Comparison tables
│
├── notebooks/                         # Jupyter notebooks
│   ├── Composer_identification_Project_AAI_511_Group6.ipynb  # ★ FINAL SUBMISSION NOTEBOOK
│   ├── Composer_Classification_Colab_Improved.ipynb  # Improved CRNN (Google Colab)
│   ├── Composer_Classification_Final.ipynb  # End-to-end pipeline (LSTM + CNN)
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Data cleaning and splitting
│   ├── 03_Feature_Extraction.ipynb    # Feature extraction for LSTM/CNN
│   ├── 04_LSTM_Model.ipynb            # LSTM model training
│   ├── 05_CNN_Model.ipynb             # CNN model training
│   └── 06_Hyperparameter_Tuning.ipynb # Hyperparameter optimization
│
└── src/                               # Source code
    ├── __init__.py
    ├── main.py                        # Entry point
    ├── run.py                         # Alternative entry point
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py                  # Configuration constants
    │   ├── helpers.py                 # Utility functions
    │   └── setup.py                   # Project initialization
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── midi_loader.py             # MIDI file loading
    │   ├── clean_dataset.py           # Data cleaning
    │   ├── augmentation.py            # Data augmentation
    │   └── split_dataset.py           # Train/val/test splitting
    │
    ├── features/
    │   ├── __init__.py
    │   ├── note_features.py           # Note-level features (LSTM)
    │   ├── piano_roll.py              # Piano roll features (CNN)
    │   ├── chord_features.py          # Chord analysis
    │   └── tempo_features.py          # Tempo and timing features
    │
    ├── datasets/
    │   ├── __init__.py
    │   ├── lstm_dataset.py            # LSTM dataset preparation
    │   └── cnn_dataset.py             # CNN dataset preparation
    │
    ├── models/
    │   ├── __init__.py
    │   ├── lstm_model.py              # LSTM architectures
    │   └── cnn_model.py               # CNN architectures
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train_lstm.py              # LSTM training loop
    │   └── train_cnn.py               # CNN training loop
    │
    └── evaluation/
        ├── __init__.py
        ├── metrics.py                 # Evaluation metrics
        └── visualize.py               # Visualization utilities
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd composer-classification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Download from [Kaggle MIDI Classic Music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)
   - Extract to `Data/midiclassics/`
   - Ensure the structure is: `Data/midiclassics/{Bach,Beethoven,Chopin,Mozart}/`

---

## Quick Start

### Option 1: Final Submission Notebook (Recommended)

The **primary deliverable** is the CRNN-based notebook that achieves 77.2% test accuracy:

```bash
jupyter notebook notebooks/Composer_identification_Project_AAI_511_Group6.ipynb
```

This notebook contains the complete end-to-end pipeline:
1. Data collection and preprocessing
2. Piano-roll feature extraction with sliding windows
3. CRNN model building (CNN + LSTM hybrid)
4. Model training with class weights
5. Hyperparameter tuning with KerasTuner Hyperband
6. Final evaluation and per-class analysis

> **Submitted reports**: See `reports/Composer_identification_Project_AAI_511_Group6.pdf` (PDF)
> and `reports/Composer_Identification_Project_Report_AAI_511_Group6.docx` (Word) for the
> full written project report.

### Option 2: Improved CRNN on Google Colab (Improvisation on Final Model)

An **improvisation on the final submitted model** that pushes performance further with
additional design enhancements — 3 Conv blocks (vs 2), Bidirectional LSTM, 2-channel
input (velocity + binary), Mixup augmentation, label smoothing, and Optuna-based
hyperparameter tuning. See the full [Architecture Deep Dive](#architecture-deep-dive-why-crnn-works)
and [Design Improvements table](#design-improvements-in-the-improved-crnn) for details.

```
notebooks/Composer_Classification_Colab_Improved.ipynb
```

1. Upload `notebooks/Composer_Classification_Colab_Improved.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially — the notebook auto-downloads the dataset from Kaggle

### Option 3: Baseline LSTM + CNN Notebook (Local)

Execute the separate LSTM and CNN pipeline locally:

```bash
jupyter notebook notebooks/Composer_Classification_Final.ipynb
```

### Option 4: Run Individual Notebooks

Execute notebooks sequentially:

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Preprocessing.ipynb
jupyter notebook notebooks/03_Feature_Extraction.ipynb
jupyter notebook notebooks/04_LSTM_Model.ipynb
jupyter notebook notebooks/05_CNN_Model.ipynb
jupyter notebook notebooks/06_Hyperparameter_Tuning.ipynb
```

### Option 5: Run from Python

```python
from src.utils.config import ensure_directories
from src.preprocessing import load_midi_dataset
from src.datasets import prepare_lstm_dataset, prepare_cnn_dataset
from src.training import train_lstm_model, train_cnn_model

# Initialize directories
ensure_directories()

# Load and preprocess data
metadata, stats = load_midi_dataset()
# ... continue with training
```

---

## Project Workflow

```
Load Dataset
    ↓
Data Exploration (EDA)
    ↓
Data Preprocessing & Cleaning
    ↓
Feature Extraction
    ├─ Note-level features (LSTM)
    └─ Piano roll features (CNN)
    ↓
Data Augmentation
    ↓
Dataset Splitting (80/10/10)
    ↓
┌─────────────────────┬──────────────────────┐
│                     │                      │
LSTM Training      CNN Training
│                     │                      │
└─────────────────────┴──────────────────────┘
    ↓
Hyperparameter Tuning (Optional)
    ↓
Model Evaluation
    ├─ Accuracy, Precision, Recall, F1
    ├─ Confusion Matrix
    └─ ROC Curves
    ↓
Model Comparison
    ↓
Final Report
```

---

## Model Architectures

### Baseline: LSTM Model

```
Input (500 time steps, 3 features: pitch, velocity, duration)
    ↓
LSTM (256 units) → Dropout (0.3)
    ↓
LSTM (256 units)
    ↓
Dense (512, ReLU) → Dropout (0.3)
    ↓
Dense (4, Softmax) → Composer class
```

**Strength**: Captures sequential note patterns (melody, rhythm)  
**Weakness**: No concept of harmony — can't see what notes sound *simultaneously*

### Baseline: CNN Model

```
Input (128 pitches × 1000 time frames × 1 channel)
    ↓
Conv2D (64, 3×3) → MaxPool (2×2)
    ↓
Conv2D (128, 3×3) → MaxPool (2×2)
    ↓
Conv2D (256, 3×3) → Flatten
    ↓
Dense (512, ReLU) → Dropout (0.3)
    ↓
Dense (4, Softmax) → Composer class
```

**Strength**: Detects spatial patterns in the piano roll (chords, texture density)  
**Weakness**: No temporal memory — treats the piece as a static image

### Improved: CRNN (CNN → LSTM Hybrid)

```
Input (88 pitches × 60 time frames × 2 channels: velocity + binary)
    ↓
┌─── CNN Front-end (spatial feature extraction) ───┐
│  Conv2D (64,  3×3) → BatchNorm → MaxPool → Drop  │
│  Conv2D (128, 3×3) → BatchNorm → MaxPool → Drop  │
│  Conv2D (256, 3×3) → BatchNorm → MaxPool → Drop  │
└──────────────────────────────────────────────────-┘
    ↓
Permute + Reshape → (time_steps, pitch_features)
    ↓
┌─── LSTM Back-end (temporal modeling) ────────────┐
│  Bidirectional LSTM (128 units, return_sequences) │
│  Dropout (0.3)                                    │
│  Bidirectional LSTM (64 units)                    │
└──────────────────────────────────────────────────-┘
    ↓
Dense (128, ReLU) → Dropout (0.3)
    ↓
Dense (4, Softmax) → Composer class
```

**Input**: 2-channel piano roll windows (velocity + binary)  
**Optimizer**: Adam with cosine decay  
**Loss**: Categorical Crossentropy with label smoothing  
**Regularisation**: Dropout + Mixup augmentation  
**HP Tuning**: Optuna (Bayesian optimization with MedianPruner)

---

## Architecture Deep Dive: Why CRNN Works

Music has **two types of patterns** that matter for composer identification:

1. **Vertical patterns** — what notes sound *together* (chords, intervals, texture)
2. **Horizontal patterns** — how notes *change over time* (melody, phrasing, dynamics)

An LSTM sees only the sequence. A CNN sees only the image. **Neither alone captures both.** The CRNN solves this by chaining them.

### How the CRNN Processes a Bach Fugue

Imagine a Bach fugue where 4 voices enter one at a time:

```
Time →   t1  t2  t3  t4  t5  t6  t7  t8  t9  t10  t11  t12
        ──────────────────────────────────────────────────────
C5   │    .   .   .   .   .   .   1   1   1    .    .    1    ← Voice 3 enters
A4   │    .   .   .   1   1   1   .   .   1    1    .    .    ← Voice 2 enters
E4   │    1   1   1   .   .   1   1   1   .    .    .    .    ← Voice 1 (starts alone)
C4   │    .   .   .   .   .   .   .   .   .    .    1    1    ← Voice 4 enters
```

**Step 1 — CNN layers ask: "What's happening locally?"**

The Conv2D slides a 3×3 filter across the piano roll like a magnifying glass:

```
Patch at (E4, t1):              Patch at (A4, t4):
┌───────────┐                   ┌───────────┐
│  .   .   .  │  ← empty        │  1   1   1  │  ← two voices
│  1   1   1  │     above       │  .   1   1  │     active
│  .   1   1  │  ← one voice    │  .   .   .  │     together
└───────────┘                   └───────────┘
 "single melodic line"           "two parallel voices"
```

The CNN learns filters that activate for different textures:
- **Filter A** fires for "single voice moving stepwise" → typical Chopin melody
- **Filter B** fires for "two voices in close intervals" → typical Bach counterpoint
- **Filter C** fires for "dense chord cluster" → typical Beethoven sforzando

**Step 2 — The Reshape (the critical step):**

Converts the CNN's 3D feature maps into a **time sequence** that the LSTM can read.
Each time step becomes a rich summary: "what harmonic/textural patterns exist *at this moment*."

**Step 3 — LSTM asks: "How does the texture evolve?"**

```
t1–t3:   "Single voice"         → Memory: [solo ✓]
t4–t6:   "Two-voice counterpoint" → Memory: [solo ✓, counterpoint ✓, voices_increasing ✓]
t7–t9:   "Three voices, denser"  → Memory: [voices_increasing ✓, dense ✓]
t10–t12: "Four-voice fugue"     → Memory: [progressive voice entry = FUGUE PATTERN]
```

This temporal pattern — *voices entering one at a time into dense counterpoint* — is
**extremely Bach-specific**. Mozart rarely writes fugues. Chopin almost never. The LSTM
captures this because the CNN has already summarised *what's happening*, and the LSTM
only needs to learn *how it changes*.

### What Each Component Catches Per Composer

| Composer | CNN Detects (spatial) | LSTM Detects (temporal) |
|---|---|---|
| **Bach** | High note density, multiple independent voices | Voices entering sequentially (fugue structure) |
| **Beethoven** | Sudden clusters of high-velocity notes | Dramatic tension-release arcs |
| **Chopin** | Wide pitch spread, sparse regular spacing | Rubato-like irregular temporal flow |
| **Mozart** | Regular low-register pattern + single melody | Predictable, periodic phrase structure |

### Why Separate Models Fail

- **LSTM alone** sees `(E4, vel=80, dur=0.3), (D4, vel=75, dur=0.3), ...` — it knows E4 comes
  before D4 but has **no idea that A4 was playing simultaneously**. Harmony is invisible.
- **CNN alone** sees the entire piano roll as a flat image — it can spot chord shapes but
  **has no temporal memory**. It can't distinguish "voices entering left-to-right" from
  "voices entering right-to-left."
- **CRNN** chains them: CNN says *"here's what's happening at each moment,"* LSTM says
  *"here's how those moments connect into a story."*

> **A fugue isn't defined by its individual notes (LSTM input) or its visual shape
> (CNN input) — it's defined by how its *texture evolves over time*.
> Only CNN → LSTM captures that.**

---

## Feature Extraction

### LSTM Features (Note-Level)

- **Pitch**: MIDI note number (0-127)
- **Duration**: Note length in seconds
- **Velocity**: Note loudness (0-127)
- **Interval**: Pitch difference between consecutive notes

All sequences are padded/truncated to `SEQUENCE_LENGTH=500`.

### CNN Features (Piano Roll)

- **Velocity Piano Roll**: Note velocity at each time step
- **Binary Piano Roll**: Active notes (1) vs inactive (0)
- **Onset Piano Roll**: Note onsets only
- **Combined Piano Roll**: Velocity + onset information
- **Multi-Channel Piano Roll**: Per-instrument representation

All piano rolls are resized to `(128, MAX_TIME_STEPS)` where `MAX_TIME_STEPS=1000`.

### Optional Features

- **Chord Statistics**: Detected chords and progressions
- **Tempo Features**: Average tempo, tempo changes, variance
- **Note Density**: Notes per second over time
- **Velocity Statistics**: Mean, std, min, max velocity

---

## Configuration

Edit `src/utils/config.py` to modify:

- **Data paths**: `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, etc.
- **Model hyperparameters**: `LSTM_HIDDEN_UNITS`, `CNN_FILTERS`, learning rates, etc.
- **Training parameters**: `BATCH_SIZE`, `EPOCHS`, `EARLY_STOPPING_PATIENCE`
- **Feature parameters**: `SEQUENCE_LENGTH`, `MAX_TIME_STEPS`, `PITCH_RANGE`

---

## Evaluation Metrics

### Per-Model Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score (Weighted)**: Harmonic mean of precision and recall (class-weighted)
- **F1 Score (Macro)**: Unweighted average of per-class F1 scores
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Visualizations

- **Confusion Matrix**: Per-class classification results
- **ROC Curves**: One-vs-rest for each composer
- **Training History**: Accuracy and loss over epochs
- **Per-Class Performance**: Precision, recall, F1 per composer
- **Model Comparison Table**: Side-by-side metrics

---

## Results & Performance Comparison

| Model | Notebook | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| LSTM (Baseline) | `Composer_Classification_Final.ipynb` | 41.2% | 39.4% | — |
| CNN (Baseline) | `Composer_Classification_Final.ipynb` | 38.2% | ~26% | — |
| LSTM (Optuna+Aug) | `Composer_Classification_Final.ipynb` | 46.5% | 43.4% | — |
| CRNN (Baseline) | `Composer_identification_Project_AAI_511_Group6.ipynb` | 75.3% | 63.3% | 75.7% |
| CRNN (Hyperband) | `Composer_identification_Project_AAI_511_Group6.ipynb` | 77.2% | 65.9% | 77.6% |
| **Improved CRNN (Optuna)** | **`Composer_Classification_Colab_Improved.ipynb`** | **TBD** | **TBD** | **TBD** |

### Design Improvements in the Improved CRNN

| Feature | Baseline CRNN | Improved CRNN | Rationale |
|---|---|---|---|
| Conv blocks | 2 | **3** | Deeper feature hierarchy |
| LSTM type | Unidirectional | **Bidirectional** | Sees past + future context |
| Window length | 4 s (40 frames) | **6 s** (60 frames) | More musical context |
| Windows/piece | 8 | **16** | ~2× more training data |
| Input channels | 1 (binary) | **2** (velocity + binary) | Retains dynamics information |
| Loss | Standard CE | **Label-smoothed CE** | Reduces over-confidence |
| Regularisation | Dropout only | Dropout + **Mixup** | Smoother decision boundaries |
| HP tuning | Hyperband (KerasTuner) | **Optuna** (Bayesian) | More sample-efficient search |

### Saved Models

- `models/saved_models/lstm_composer_classifier_best.h5`: Best LSTM model
- `models/saved_models/cnn_composer_classifier_best.h5`: Best CNN model
- `best_improved_crnn.keras`: Improved CRNN baseline (Colab)
- `best_optuna_crnn.keras`: Improved CRNN after Optuna tuning (Colab)

Evaluation results and visualizations are saved in `reports/`:

- `reports/figures/`: Plots (confusion matrices, ROC curves, training history)
- `reports/tables/`: Comparison tables (CSV format)

---

## Reproducibility

This project is fully reproducible:

- **Fixed Random Seeds**: `RANDOM_SEED=42` in `config.py`
- **Deterministic Operations**: TensorFlow and NumPy seeded
- **Version Control**: All dependencies pinned in `requirements.txt`
- **Data Versioning**: Dataset hash can be verified
- **Modular Code**: Each step can be independently verified

To reproduce results:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Composer_Classification_Final.ipynb
# Run all cells
```

---

## Code Quality

The codebase follows best practices:

- **PEP 8**: Compliant with Python style guidelines
- **Type Hints**: Function signatures include type annotations
- **Docstrings**: All functions and classes documented
- **Modular Design**: Separation of concerns (preprocessing, features, models, training, evaluation)
- **Error Handling**: Graceful handling of missing files and invalid data
- **Logging**: Informative print statements for debugging

---

## Dependencies

See `requirements.txt` for the complete list. Key packages:

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **PrettyMIDI**: MIDI file processing
- **music21**: Music analysis (optional)
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pretty_midi'`

**Solution**: Install missing dependency:
```bash
pip install pretty_midi
```

### Issue: `FileNotFoundError: Data/midiclassics/ not found`

**Solution**: Download the dataset from Kaggle and extract to the correct location:
```bash
# Ensure structure: Data/midiclassics/{Bach,Beethoven,Chopin,Mozart}/
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in `config.py`:
```python
LSTM_BATCH_SIZE = 16  # Default: 32
CNN_BATCH_SIZE = 16   # Default: 32
```

### Issue: Notebook kernel crashes

**Solution**: Restart the kernel and run cells sequentially. Ensure sufficient RAM (8GB+ recommended).

---

## Future Improvements

- **Transformer Models**: Implement attention-based sequence models (Music Transformer)
- **Longer Analysis Windows**: 10–15 second windows for better phrase-level modeling
- **Ensemble Methods**: Combine multiple CRNN models with different window sizes
- **Multi-Scale Features**: Note density, polyphony degree, and tempo as additional input channels
- **Explainability**: Grad-CAM visualization to see which piano-roll regions drive classification
- **More Composers**: Extend to Liszt, Debussy, Schubert, etc.
- **Web Deployment**: REST API and web interface for real-time MIDI classification

---

## References

- Kaggle MIDI Classic Music Dataset: https://www.kaggle.com/datasets/blanderbuss/midi-classic-music
- PrettyMIDI Documentation: https://github.com/craffel/pretty-midi
- TensorFlow/Keras: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/

---

## License

This project is part of the MS in Applied Artificial Intelligence program at the University of San Diego.

Licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for the full terms.

```
Copyright 2026 Ved Prakash Dwivedi, Jagdish Mane, Tamayi Mlanda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

## Contributors

1. **Ved Prakash Dwivedi**
2. **Jagdish Mane**
3. **Tamayi Mlanda**

## Faculty Advisor

- **Prof. Azka Azka**  
  University of San Diego – Applied Artificial Intelligence Program

---

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Last Updated**: August 2026  
**Project Status**: Complete
