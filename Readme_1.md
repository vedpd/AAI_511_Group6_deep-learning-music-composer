# Composer Classification using Deep Learning

### University of San Diego – MS in Applied Artificial Intelligence

### Deep Learning Final Project

---

# Project Objective

Develop a complete deep learning pipeline to classify the composer of a classical music piece using MIDI files.

The project must classify **only the following four composers**:

* Bach
* Beethoven
* Chopin
* Mozart

The project must compare two deep learning architectures:

1. Long Short-Term Memory (LSTM)
2. Convolutional Neural Network (CNN)

The final repository should be reproducible, modular, well documented, and suitable for graduate-level coursework.

---

# Dataset

Dataset Source:

https://www.kaggle.com/datasets/blanderbuss/midi-classic-music

Only retain MIDI files belonging to:

* Bach
* Beethoven
* Chopin
* Mozart

Ignore every other composer contained in the dataset.

Expected folder structure:

```text
data/
    raw/
        Bach/
        Beethoven/
        Chopin/
        Mozart/
```

---

# Overall Project Workflow

```
Load Dataset
      ↓
Data Exploration
      ↓
Preprocessing
      ↓
Feature Extraction
      ↓
Data Augmentation
      ↓
Dataset Split
      ↓
LSTM Training
      ↓
CNN Training
      ↓
Hyperparameter Tuning
      ↓
Evaluation
      ↓
Comparison
      ↓
Final Report
```

---

# Required Repository Structure

```
composer-classification/

README.md
requirements.txt
.gitignore

data/
    raw/
    interim/
    processed/
    features/

models/
    checkpoints/
    saved_models/

notebooks/
    01_EDA.ipynb
    02_Preprocessing.ipynb
    03_Feature_Extraction.ipynb
    04_LSTM_Model.ipynb
    05_CNN_Model.ipynb
    06_Hyperparameter_Tuning.ipynb
    Composer_Classification_Final.ipynb

reports/
    figures/
    tables/
    final_report.pdf

src/

    preprocessing/

        midi_loader.py
        clean_dataset.py
        augmentation.py
        split_dataset.py

    features/

        note_features.py
        piano_roll.py
        chord_features.py
        tempo_features.py

    datasets/

        lstm_dataset.py
        cnn_dataset.py

    models/

        lstm_model.py
        cnn_model.py

    training/

        train_lstm.py
        train_cnn.py

    evaluation/

        metrics.py
        visualize.py

    utils/

        config.py
        helpers.py
```

---

# Required Python Libraries

Preferred libraries:

* TensorFlow/Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* PrettyMIDI
* music21
* Scikit-learn
* tqdm

Avoid unnecessary dependencies.

---

# Stage 1 — Data Collection

Tasks

* Load MIDI dataset.
* Keep only four composers.
* Verify labels.
* Generate metadata table.

Output

```
metadata.csv
```

containing

* filename
* composer
* duration
* number_of_notes
* tempo

---

# Stage 2 — Exploratory Data Analysis

Perform EDA before training.

Generate visualizations for

* Number of pieces per composer
* Distribution of note counts
* Distribution of tempo
* Distribution of durations
* Pitch histogram
* Velocity histogram
* Example piano roll
* Example note sequence

Save figures inside

```
reports/figures/
```

---

# Stage 3 — Data Preprocessing

Implement

* Load MIDI files
* Remove empty files
* Remove corrupted files
* Normalize note timing
* Standardize sequence lengths
* Pad short sequences
* Truncate long sequences

Save processed dataset.

---

# Stage 4 — Data Augmentation

Implement at least the following augmentations:

* Pitch shifting
* Tempo scaling
* Velocity variation
* Time shifting (optional)

Allow augmentation to be enabled or disabled.

---

# Stage 5 — Feature Extraction

Extract the following features where applicable:

## LSTM Features

* Note sequence
* Pitch
* Duration
* Velocity

Generate padded numerical sequences suitable for sequence modeling.

---

## CNN Features

Convert each MIDI file into a Piano Roll matrix.

Suggested shape

```
128 x fixed_time_steps
```

Resize all piano rolls to identical dimensions.

---

## Optional Additional Features

* Chord statistics
* Average tempo
* Note density
* Pitch histogram

---

# Stage 6 — Dataset Splitting

Split dataset into

```
Training
Validation
Testing
```

Recommended

```
80%
10%
10%
```

Ensure class balance using stratified splitting.

---

# Stage 7 — LSTM Model

Implement an LSTM classifier.

Suggested architecture

Embedding

↓

LSTM

↓

Dropout

↓

LSTM

↓

Dense

↓

Softmax

Use

* Adam optimizer
* EarlyStopping
* ModelCheckpoint

Save best model.

---

# Stage 8 — CNN Model

Implement CNN using piano roll representation.

Suggested architecture

Conv2D

↓

MaxPooling

↓

Conv2D

↓

Flatten

↓

Dense

↓

Softmax

Train independently from LSTM.

Save best model.

---

# Stage 9 — Hyperparameter Tuning

Perform experiments varying

* Learning rate
* Batch size
* Epochs
* Hidden units
* Dropout
* Sequence length

Document results.

---

# Stage 10 — Evaluation

Generate

Classification Report

Confusion Matrix

Accuracy

Precision

Recall

F1-score

ROC Curves (if applicable)

Training Accuracy Plot

Validation Accuracy Plot

Training Loss Plot

Validation Loss Plot

Store every figure under

```
reports/figures/
```

---

# Stage 11 — Model Comparison

Create a comparison table

| Metric        | LSTM | CNN |
| ------------- | ---- | --- |
| Accuracy      |      |     |
| Precision     |      |     |
| Recall        |      |     |
| F1 Score      |      |     |
| Parameters    |      |     |
| Training Time |      |     |

Discuss

* strengths
* weaknesses
* computational cost
* observations

---

# Stage 12 — Final Notebook

Create a single polished notebook

```
Composer_Classification_Final.ipynb
```

The notebook must execute sequentially without errors.

Sections

1. Introduction
2. Dataset
3. EDA
4. Preprocessing
5. Feature Extraction
6. LSTM
7. CNN
8. Hyperparameter Tuning
9. Evaluation
10. Model Comparison
11. Conclusion

Avoid debugging code or unused cells.

---

# Report Requirements

Prepare an APA 7 formatted report including

* Abstract
* Introduction
* Literature Review
* Dataset Description
* Methodology
* Data Preprocessing
* Feature Extraction
* LSTM Architecture
* CNN Architecture
* Experimental Setup
* Results
* Discussion
* Limitations
* Future Work
* Conclusion
* References

---

# Code Quality Requirements

All code should

* Follow PEP8
* Include docstrings
* Use type hints where practical
* Be modular
* Avoid duplicate logic
* Separate preprocessing from model training
* Be reproducible using fixed random seeds

---

# Deliverables

The final repository must produce

* Final Project Notebook
* Trained LSTM model
* Trained CNN model
* Evaluation metrics
* Confusion matrices
* Accuracy and loss plots
* Comparison table
* Final APA 7 report

---

# Success Criteria Checklist

* Dataset filtered to four composers.
* Exploratory data analysis completed.
* MIDI preprocessing implemented.
* Data augmentation implemented.
* Feature extraction completed.
* LSTM model trained and evaluated.
* CNN model trained and evaluated.
* Hyperparameter tuning performed.
* Metrics calculated (Accuracy, Precision, Recall, F1).
* Confusion matrices generated.
* Models compared.
* Final notebook runs end-to-end.
* APA 7 report completed.
* Repository is reproducible from scratch.

---

# Future Improvements

Potential future extensions include

* Transformer-based sequence models
* Bidirectional LSTM
* Attention mechanisms
* Vision Transformers on piano-roll images
* Multi-modal models combining sequential and image representations
* Composer similarity visualization using embeddings
* Grad-CAM or attention-based explainability
* Deployment as a simple web application
