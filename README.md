# Composer Classification using Deep Learning

**University of San Diego – MS in Applied Artificial Intelligence**  
**Deep Learning Final Project**

## Project Overview

This project implements a complete deep learning pipeline to classify the composer of classical music pieces using MIDI files. The system compares two neural network architectures:

- **LSTM (Long Short-Term Memory)**: Processes sequential note-level features
- **CNN (Convolutional Neural Network)**: Analyzes piano roll representations

### Target Composers

- Bach
- Beethoven
- Chopin
- Mozart

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
├── reports/                           # Generated outputs
│   ├── figures/                       # Plots and visualizations
│   └── tables/                        # Comparison tables
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Data cleaning and splitting
│   ├── 03_Feature_Extraction.ipynb    # Feature extraction for LSTM/CNN
│   ├── 04_LSTM_Model.ipynb            # LSTM model training
│   ├── 05_CNN_Model.ipynb             # CNN model training
│   ├── 06_Hyperparameter_Tuning.ipynb # Hyperparameter optimization
│   └── Composer_Classification_Final.ipynb  # Complete end-to-end pipeline
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

### Option 1: Run the Final Notebook (Recommended)

Execute the complete pipeline in a single notebook:

```bash
jupyter notebook notebooks/Composer_Classification_Final.ipynb
```

This notebook:
1. Loads and explores the MIDI dataset
2. Cleans and preprocesses the data
3. Extracts features for both LSTM and CNN
4. Trains both models
5. Evaluates and compares results

### Option 2: Run Individual Notebooks

Execute notebooks sequentially:

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Preprocessing.ipynb
jupyter notebook notebooks/03_Feature_Extraction.ipynb
jupyter notebook notebooks/04_LSTM_Model.ipynb
jupyter notebook notebooks/05_CNN_Model.ipynb
jupyter notebook notebooks/06_Hyperparameter_Tuning.ipynb
```

### Option 3: Run from Python

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

### LSTM Model

```
Input (sequence_length, num_features)
    ↓
Embedding (if needed)
    ↓
LSTM (256 units)
    ↓
Dropout (0.3)
    ↓
LSTM (256 units)
    ↓
Dense (512 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (4 units, Softmax)
    ↓
Output (composer class)
```

**Input**: Padded sequences of note features (pitch, duration, velocity)  
**Optimizer**: Adam (lr=0.001)  
**Loss**: Categorical Crossentropy  
**Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### CNN Model

```
Input (128, time_steps, 1)
    ↓
Conv2D (64 filters, 3x3 kernel)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3 kernel)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (256 filters, 3x3 kernel)
    ↓
Flatten
    ↓
Dense (512 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (4 units, Softmax)
    ↓
Output (composer class)
```

**Input**: Piano roll matrices (128 pitches × time steps)  
**Optimizer**: Adam (lr=0.001)  
**Loss**: Categorical Crossentropy  
**Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

## Results

The trained models are saved in `models/saved_models/`:

- `lstm_composer_classifier_best.h5`: Best LSTM model (validation)
- `lstm_composer_classifier_final.h5`: Final LSTM model (after training)
- `cnn_composer_classifier_best.h5`: Best CNN model (validation)
- `cnn_composer_classifier_final.h5`: Final CNN model (after training)

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

- **Transformer Models**: Implement attention-based sequence models
- **Bidirectional LSTM**: Capture context from both directions
- **Ensemble Methods**: Combine LSTM and CNN predictions
- **Data Augmentation**: Advanced augmentation techniques (time stretching, pitch shifting)
- **Explainability**: Grad-CAM and attention visualization
- **Web Deployment**: REST API and web interface
- **Real-Time Classification**: Classify music from live MIDI input

---

## References

- Kaggle MIDI Classic Music Dataset: https://www.kaggle.com/datasets/blanderbuss/midi-classic-music
- PrettyMIDI Documentation: https://github.com/craffel/pretty-midi
- TensorFlow/Keras: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/

---

## License

This project is part of the MS in Applied Artificial Intelligence program at the University of San Diego.

---
## MIT License

Copyright (c) 2024 Smart Parking IoT Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Last Updated**: August 2024  
**Project Status**: Complete
