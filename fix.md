# MASTER PROMPT — Fix and Finalize AAI-511 Deep Learning Composer Classification Project

You are working on my GitHub repository:

`https://github.com/vedpd/AAI_511_Group6_deep-learning-music-composer`

I need you to perform a **complete audit, correction, validation, cleanup, and documentation pass** on the existing repository in a single workflow.

Do NOT create a new project from scratch unless absolutely necessary. Preserve good existing work, improve the architecture where required, remove broken/redundant components, and ensure the final repository is genuinely runnable and aligned with the course requirements below.

---

# 1. COURSE PROJECT REQUIREMENT — THIS IS THE SOURCE OF TRUTH

The project requirement is:

## Introduction

Music is a form of art that is ubiquitous and has a rich history. Different composers have created music with their unique styles and compositions. However, identifying the composer of a particular piece of music can be a challenging task, especially for novice musicians or listeners.

The project aims to use deep learning techniques to identify the composer of a given piece of music accurately.

## Objective

Develop a deep learning model that predicts the composer of a given musical score accurately.

The project MUST use TWO deep learning techniques:

1. Long Short-Term Memory (LSTM)
2. Convolutional Neural Network (CNN)

## Dataset

Use the Kaggle dataset:

https://www.kaggle.com/datasets/blanderbuss/midi-classic-music

Only use these four composers:

1. Bach
2. Beethoven
3. Chopin
4. Mozart

The task is therefore:

**4-class composer classification from MIDI musical scores.**

IMPORTANT:

This is NOT a music-generation project.

Do NOT redesign the project into a generative music composer.

The correct problem is:

MIDI musical score
→ preprocessing
→ feature extraction
→ deep learning
→ composer classification
→ Bach / Beethoven / Chopin / Mozart

---

# 2. REQUIRED METHODOLOGY

The final project must demonstrably implement all of these:

### 1. Data Collection

Use the specified Kaggle MIDI Classic Music dataset.

Select only:

* Bach
* Beethoven
* Chopin
* Mozart

Document the dataset source and exact filtering process.

Do not silently use another dataset.

---

### 2. Data Pre-processing

Process the MIDI files into a format suitable for deep learning.

The preprocessing pipeline should include appropriate MIDI processing such as:

* parsing MIDI files
* extracting notes
* handling timing/duration
* handling velocity where appropriate
* handling tempo where appropriate
* sequence construction
* normalization/scaling where appropriate
* padding/truncation where appropriate
* label encoding

Data augmentation should be implemented where appropriate.

Possible augmentation includes:

* transposition/pitch shifting
* tempo variation
* other musically valid transformations

However, avoid introducing unrealistic transformations.

---

### 3. Feature Extraction

Extract meaningful musical features from MIDI files.

At minimum investigate/use features such as:

* pitch/note information
* note duration
* note timing/inter-onset interval
* velocity
* tempo
* chord/harmony information where feasible

Do not include features merely because they are available.

Explain why each selected feature is useful for composer classification.

---

### 4. Model Building

Build TWO separate deep learning approaches:

## Model A — LSTM

Use an LSTM-based sequence model appropriate for sequential musical information.

The model should learn patterns across sequences of musical events.

## Model B — CNN

Use a CNN-based architecture appropriate for the selected representation.

If the input is transformed into a matrix/tensor representation, clearly explain what each dimension represents.

Both models must perform the same final 4-class classification task:

```text
Bach
Beethoven
Chopin
Mozart
```

---

### 5. Model Training

Train both models using the processed data.

The training pipeline must include:

* train/validation/test split
* appropriate loss function
* optimizer
* metrics
* callbacks where appropriate
* early stopping where appropriate
* model checkpointing
* reproducibility/random seeds

---

### 6. Model Evaluation

Evaluate both LSTM and CNN using at minimum:

* Accuracy
* Precision
* Recall

Also include:

* Macro F1
* Weighted F1
* per-class precision
* per-class recall
* per-class F1
* confusion matrix

Because the dataset is not perfectly balanced, do NOT rely only on accuracy or weighted metrics.

---

### 7. Model Optimization

Perform meaningful hyperparameter optimization.

Potential parameters include:

* learning rate
* batch size
* sequence length
* number of LSTM units
* number of LSTM layers
* dropout
* CNN filter sizes
* number of CNN filters
* dense layer size
* optimizer parameters

Do not perform optimization merely for appearance.

Use validation data for hyperparameter selection.

The test set must remain isolated until final evaluation.

---

# 3. CRITICAL DATA LEAKAGE REQUIREMENT

This is extremely important.

The data split must happen BEFORE augmentation.

Correct:

```text
Original MIDI files
        |
        v
Composer-level / composition-level split
        |
        +----------------+
        |                |
      TRAIN            VAL/TEST
        |
        v
Augmentation
        |
        v
Feature extraction
        |
        v
Model training
```

Do NOT do:

```text
Original MIDI
     |
     v
Augmentation
     |
     v
Random train/test split
```

because augmented versions of the same musical composition could appear in both training and test sets.

This would cause data leakage and artificially inflate performance.

If possible, split at the composition/file level and preserve composer stratification.

Clearly document the split methodology.

---

# 4. IMPORTANT DATASET REQUIREMENT

The repository currently does not contain the original MIDI dataset.

Do NOT blindly commit the full Kaggle dataset to GitHub.

Instead:

1. Document the official Kaggle dataset source.
2. Explain how to download it.
3. Provide exact instructions for placing the files.
4. Provide a script or utility to filter the four required composers.
5. Clearly document the expected directory structure.
6. If licensing permits, optionally provide a very small sample dataset for testing.

Create a clear dataset setup section.

Expected conceptual structure:

```text
data/
├── raw/
│   ├── Bach/
│   ├── Beethoven/
│   ├── Chopin/
│   └── Mozart/
├── processed/
├── features/
└── splits/
```

Do not require the full raw dataset to be committed.

---

# 5. REPOSITORY CLEANUP

Audit the entire repository and remove unnecessary/generated files.

In particular, remove all committed:

```text
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
```

Create/fix a root-level `.gitignore`.

At minimum include:

```gitignore
__pycache__/
*.py[cod]

.ipynb_checkpoints/

.venv/
venv/
env/

.env

.vscode/

data/raw/
data/processed/
data/features/

models/checkpoints/
models/saved_models/

reports/figures/
reports/tables/

*.h5
*.keras
*.pth
*.pt
*.ckpt

*.log
```

Adjust this if the project needs to track particular files.

Do not ignore source code, notebooks, configuration, README files, or required project artifacts.

---

# 6. FIX README DUPLICATION

The repository currently contains both:

```text
README.md
Readme.md
```

Keep only:

```text
README.md
```

Ensure Git handles the case-only filename change correctly.

The README must become the primary entry point for the project.

---

# 7. REWRITE THE README

Create a professional master's-level README.

It should include:

# Deep Learning-Based Composer Classification from MIDI

## 1. Project Overview

Explain the problem and why composer identification is useful.

## 2. Objective

Explicitly state:

> Develop deep learning models using LSTM and CNN architectures to classify MIDI musical scores into Bach, Beethoven, Chopin, and Mozart.

## 3. Dataset

Include:

* Kaggle source
* dataset description
* selected composers
* number of files actually used after filtering
* class distribution

Do not invent numbers.

Calculate them from the actual dataset if available.

## 4. Methodology

Show:

```text
Kaggle MIDI Dataset
        ↓
Select 4 composers
        ↓
Data validation
        ↓
Composition-level Train/Validation/Test Split
        ↓
Training-only Data Augmentation
        ↓
MIDI Preprocessing
        ↓
Feature Extraction
        ↓
       ┌───────────────┐
       │               │
      LSTM            CNN
       │               │
       └───────┬───────┘
               ↓
       Composer Prediction
               ↓
Bach / Beethoven / Chopin / Mozart
```

## 5. Features

Document the actual features used.

## 6. LSTM Architecture

Explain the architecture.

## 7. CNN Architecture

Explain the architecture.

## 8. Training

Explain:

* split
* loss
* optimizer
* epochs
* batch size
* callbacks
* hyperparameter tuning

## 9. Results

Include actual experimental results.

Do NOT fabricate results.

## 10. Model Comparison

Compare LSTM and CNN.

## 11. Error Analysis

Discuss which composers are confused and why.

## 12. How to Run

Provide exact setup commands.

## 13. Repository Structure

Show the complete structure.

## 14. Team Members

Preserve existing team information if already present.

## 15. References

Include dataset and relevant technical references.

---

# 8. FIX SETUP GUIDE

The existing setup documentation contains incorrect/broken commands.

Correct Windows virtual environment activation.

Use:

```powershell
.\venv\Scripts\Activate.ps1
```

or, where appropriate:

```cmd
venv\Scripts\activate
```

Do not use malformed commands such as:

```text
venv\S cripts\a ctivate
```

Also correct references to the project directory.

Do not use an obsolete directory name such as:

```text
AAI_511_Music_Composer
```

Use the actual repository/project name consistently.

---

# 9. ENVIRONMENT REPRODUCIBILITY

Audit:

```text
requirements.txt
environment.yml
```

Make them consistent.

Prefer pinned or sufficiently constrained versions based on the versions actually used during development.

Do not randomly upgrade TensorFlow/Keras just because a newer version exists.

The goal is:

> A fresh environment should reproduce the project.

Document the Python version and major dependency versions.

If both Conda and pip are retained, explain which one is the recommended setup.

---

# 10. AUDIT THE EXISTING SOURCE CODE

Do not assume the existing code is correct.

Inspect every module under:

```text
src/
```

including:

```text
preprocessing/
features/
datasets/
models/
training/
evaluation/
utils/
```

Check:

* imports
* paths
* data types
* tensor dimensions
* sequence construction
* labels
* model output dimensions
* loss function
* metrics
* train/validation/test handling
* callbacks
* checkpoint paths
* reproducibility
* error handling

Fix all actual issues you find.

Do not introduce unnecessary abstraction merely for the sake of complexity.

---

# 11. AUDIT THE LSTM

The LSTM must genuinely process sequential musical information.

Verify:

* input shape
* sequence length
* feature dimension
* masking/padding handling
* LSTM layers
* dropout
* dense classification layer
* softmax output

Final output must be:

```text
4 classes
```

Ensure the architecture is documented accurately.

Do not describe it as a music-generation model.

---

# 12. AUDIT THE CNN

The CNN must receive a representation appropriate for convolution.

Clearly establish:

```text
MIDI
→ feature representation
→ tensor/image-like/matrix representation
→ CNN
→ 4-class classification
```

Ensure that the CNN is not simply receiving an arbitrary reshaped vector just to make Conv layers work.

Document exactly what the CNN input dimensions mean.

---

# 13. REMOVE QUESTIONABLE CONFIGURATION

Audit `config.py`.

Do not create directories merely because a module is imported.

Avoid side effects such as:

```python
from config import ...
```

automatically creating project directories.

Use explicit initialization functions where appropriate.

Also audit configuration values such as:

```text
SAMPLE_RATE = 44100
```

because the project is based on MIDI/symbolic music rather than raw audio.

Remove or justify parameters that are not actually used.

Do not leave audio-specific configuration in a symbolic MIDI pipeline without a clear reason.

---

# 14. NOTE FEATURE ENGINEERING

Audit whether the existing features are actually being used.

Do not document features that the code does not use.

If the assignment says:

> notes, chords, tempo

then make sure the implementation extracts meaningful equivalents where technically feasible.

For every major feature, document:

```text
Feature
Purpose
Representation
Used by LSTM?
Used by CNN?
```

---

# 15. NOTEBOOK STRUCTURE

Audit the notebooks.

The project currently has notebooks corresponding approximately to:

```text
01_EDA
02_Preprocessing
03_Feature_Extraction
04_LSTM_Model
05_CNN_Model
06_Hyperparameter_Tuning
Composer_Classification_Final
```

Preserve this modular approach if useful, but make the notebooks clean and executable.

The final notebook should be the main academic demonstration.

It should clearly show:

1. Problem statement
2. Dataset
3. EDA
4. Preprocessing
5. Feature extraction
6. Train/validation/test split
7. Augmentation
8. LSTM
9. CNN
10. Hyperparameter tuning
11. Evaluation
12. Model comparison
13. Conclusions

Avoid requiring the final notebook to unnecessarily retrain every experiment.

Where appropriate, load saved models/results.

---

# 16. FINAL NOTEBOOK REQUIREMENT

The final notebook should be presentation-quality.

It should contain:

### Dataset

Show:

* number of MIDI files
* number of samples per composer
* class distribution

### EDA

Show useful visualizations such as:

* class distribution
* note distribution
* pitch distribution
* duration distribution
* tempo distribution

Only include visualizations that are meaningful.

### Preprocessing

Clearly demonstrate the pipeline.

### LSTM

Show:

* architecture
* training
* validation curves
* final metrics
* confusion matrix

### CNN

Show:

* architecture
* training
* validation curves
* final metrics
* confusion matrix

### Hyperparameter Tuning

Show:

* parameters searched
* best parameters
* validation performance

### Final Comparison

Create a table:

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
| ----- | -------- | --------- | ------ | -------- | ----------- |
| LSTM  | actual   | actual    | actual | actual   | actual      |
| CNN   | actual   | actual    | actual | actual   | actual      |

Do not fabricate any values.

---

# 17. MODEL EVALUATION

Use appropriate sklearn metrics.

At minimum:

```python
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
classification_report
```

For multi-class classification report:

* macro average
* weighted average

must be shown.

Use:

```python
average="macro"
```

for macro metrics.

Do not report only weighted metrics.

---

# 18. REPRODUCIBILITY

Set random seeds wherever practical.

For example:

```python
random.seed(...)
numpy.random.seed(...)
tensorflow.random.set_seed(...)
```

Document that deterministic behavior may still vary depending on hardware/backend.

Save:

* preprocessing configuration
* label mapping
* model architecture
* best model weights
* training history
* final evaluation metrics

Do not commit enormous model files unless appropriate.

---

# 19. RESULTS MUST NEVER BE FABRICATED

This is extremely important.

If the repository currently does not contain executed results, you must:

1. Run the pipeline if the dataset/environment is available.
2. Generate actual results.
3. Save them.
4. Update the README and final notebook.

If the dataset cannot be downloaded automatically because Kaggle authentication is required:

* clearly document that limitation
* create the pipeline so it runs after the user places the dataset
* do NOT invent accuracy/precision/recall/F1 numbers
* clearly mark unavailable results as pending rather than making them up

---

# 20. TEST THE ENTIRE PIPELINE

After making changes, perform an end-to-end validation.

At minimum verify:

```text
Environment
    ↓
Dataset loading
    ↓
Composer filtering
    ↓
Data validation
    ↓
Train/Val/Test split
    ↓
Augmentation
    ↓
Feature extraction
    ↓
Dataset creation
    ↓
LSTM training
    ↓
CNN training
    ↓
Evaluation
    ↓
Metrics
    ↓
Plots
    ↓
Final results
```

Fix errors instead of merely documenting them.

Where the full dataset/training is computationally expensive, perform a smoke test using a small subset first.

Then explain how to run the full experiment.

---

# 21. ADD AUTOMATED VALIDATION / SMOKE TESTS

Create lightweight tests or validation scripts where useful.

At minimum test:

* dataset loader
* label mapping
* feature extraction
* sequence dimensions
* CNN input dimensions
* LSTM input dimensions
* model output dimensions
* four-class labels
* no overlap between train/validation/test files

A particularly important test:

```text
Train files ∩ Validation files = empty
Train files ∩ Test files = empty
Validation files ∩ Test files = empty
```

Also verify that augmented samples never leak into validation/test.

---

# 22. DATA QUALITY VALIDATION

Add a dataset validation step that detects:

* corrupt MIDI files
* files with no notes
* unsupported MIDI structures
* missing composer labels
* duplicate files where detectable
* unexpected composer labels

Generate a summary such as:

```text
Total files discovered:
Files successfully parsed:
Corrupt/unreadable files:
Bach:
Beethoven:
Chopin:
Mozart:
Excluded composers:
```

Do not hard-code these numbers.

Calculate them.

---

# 23. REPOSITORY STRUCTURE

Aim for a clean structure similar to:

```text
AAI_511_Group6_deep-learning-music-composer/
│
├── README.md
├── SETUP_GUIDE.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Extraction.ipynb
│   ├── 04_LSTM_Model.ipynb
│   ├── 05_CNN_Model.ipynb
│   ├── 06_Hyperparameter_Tuning.ipynb
│   └── 07_Final_Results.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── tests/
│
├── scripts/
│
└── data/
    └── .gitkeep
```

Do not create empty directories simply to make the repository look sophisticated.

Use only directories that serve a purpose.

---

# 24. PROJECT SUMMARY

Rewrite `PROJECT_SUMMARY.md` so that it reflects the ACTUAL state of the project.

It should include:

* problem
* objective
* dataset
* four composers
* preprocessing
* features
* LSTM
* CNN
* training
* optimization
* evaluation
* results
* limitations
* future improvements

Remove exaggerated statements such as:

> "All components have been successfully generated and are ready for use"

unless you have actually validated the complete pipeline.

Distinguish clearly between:

```text
Implemented
Tested
Experimentally validated
```

---

# 25. PROJECT NAMING

The current repository name contains:

```text
deep-learning-music-composer
```

Do not rename the GitHub repository unless necessary because it already exists.

However, throughout the documentation, make the actual task explicit:

**Deep Learning-Based Composer Classification from MIDI**

Do not describe the system as generating original music.

Use language such as:

> "composer classification"

> "composer identification"

> "4-class MIDI composer classification"

rather than:

> "music generation"

> "music composition"

unless discussing the broader motivation.

---

# 26. ACADEMIC ALIGNMENT MATRIX

Create a file:

```text
REQUIREMENTS_TRACEABILITY.md
```

with a table:

| Course Requirement    | Implementation                  | File/Notebook | Status   |
| --------------------- | ------------------------------- | ------------- | -------- |
| Data Collection       | Kaggle MIDI dataset             | ...           | Complete |
| Four composers        | Bach, Beethoven, Chopin, Mozart | ...           | Complete |
| Data preprocessing    | ...                             | ...           | Complete |
| Data augmentation     | ...                             | ...           | Complete |
| Feature extraction    | ...                             | ...           | Complete |
| LSTM                  | ...                             | ...           | Complete |
| CNN                   | ...                             | ...           | Complete |
| Model training        | ...                             | ...           | Complete |
| Accuracy              | ...                             | ...           | Complete |
| Precision             | ...                             | ...           | Complete |
| Recall                | ...                             | ...           | Complete |
| Hyperparameter tuning | ...                             | ...           | Complete |

Only mark something "Complete" if it is actually implemented and validated.

---

# 27. FINAL ACADEMIC REPORT SUPPORT

The repository should contain enough material to support the final Project Report.

Make sure the final outputs support these report sections:

1. Introduction
2. Problem Statement
3. Objective
4. Dataset
5. Exploratory Data Analysis
6. Data Preprocessing
7. Feature Engineering
8. Data Augmentation
9. LSTM Methodology
10. CNN Methodology
11. Training
12. Hyperparameter Optimization
13. Evaluation Metrics
14. Results
15. Comparison of LSTM vs CNN
16. Error Analysis
17. Limitations
18. Conclusion
19. Future Work
20. References

Do not write a report full of unsupported claims.

Use actual experimental evidence.

---

# 28. IMPORTANT SOFTWARE ENGINEERING RULES

While modifying the repository:

* Do not delete working functionality without reason.
* Do not rewrite everything just for stylistic reasons.
* Do not introduce unnecessary libraries.
* Do not introduce unnecessary complexity.
* Do not hard-code local Windows paths.
* Use relative/project-root paths.
* Make code work on Windows and Linux where practical.
* Keep imports clean.
* Avoid circular imports.
* Avoid hidden side effects.
* Use functions/classes where they improve maintainability.
* Add docstrings to important functions.
* Add comments only where they clarify non-obvious logic.
* Do not commit secrets.
* Do not commit datasets unnecessarily.
* Do not fabricate metrics.
* Do not fabricate dataset counts.
* Do not fabricate experiment results.

---

# 29. SPECIFIC ISSUES ALREADY IDENTIFIED IN THE CURRENT REPOSITORY

Make sure you explicitly inspect and resolve these known issues:

### Issue 1

`__pycache__` directories are committed.

FIX.

### Issue 2

Both:

```text
README.md
Readme.md
```

exist.

FIX.

### Issue 3

`.gitignore` is missing or inconsistent with the repository state.

FIX.

### Issue 4

Windows activation command in setup documentation is malformed.

FIX.

### Issue 5

Setup documentation references the wrong project directory name.

FIX.

### Issue 6

Dataset is described but not actually present in the repository.

FIX documentation and dataset setup.

### Issue 7

Dependency versions are not sufficiently reproducible.

FIX.

### Issue 8

README is too minimal.

REWRITE.

### Issue 9

Project summary overstates project completeness.

REWRITE based on actual validation.

### Issue 10

Potential augmentation/data leakage needs to be audited.

FIX if present and document the correct pipeline.

### Issue 11

Evaluation should include macro F1 and per-class metrics, not only weighted metrics.

FIX.

### Issue 12

`config.py` may create directories during import.

FIX.

### Issue 13

Questionable audio-specific configuration such as sample rate should be removed or justified for MIDI.

FIX.

### Issue 14

LSTM/CNN implementation needs to be verified against the actual assignment objective.

FIX.

### Issue 15

Notebook workflow needs a clean final notebook.

FIX.

---

# 30. FINAL VALIDATION CHECKLIST

Before declaring the work complete, verify every item below.

## Dataset

* [ ] Kaggle dataset is the documented source
* [ ] Only Bach is included
* [ ] Only Beethoven is included
* [ ] Only Chopin is included
* [ ] Only Mozart is included
* [ ] Other composers are excluded
* [ ] Dataset counts are calculated, not hard-coded
* [ ] Corrupt files are handled
* [ ] Dataset setup is documented

## Preprocessing

* [ ] MIDI parsing works
* [ ] Notes are extracted
* [ ] Timing/duration handled
* [ ] Tempo handled where used
* [ ] Features are documented
* [ ] Sequence construction works
* [ ] Padding/truncation works
* [ ] Data augmentation works
* [ ] Augmentation occurs only after splitting
* [ ] No train/test leakage

## LSTM

* [ ] Correct input shape
* [ ] Correct sequence representation
* [ ] LSTM implemented
* [ ] Four-class softmax output
* [ ] Training works
* [ ] Validation works
* [ ] Evaluation works

## CNN

* [ ] Appropriate CNN representation
* [ ] Correct tensor shape
* [ ] CNN implemented
* [ ] Four-class softmax output
* [ ] Training works
* [ ] Validation works
* [ ] Evaluation works

## Evaluation

* [ ] Accuracy
* [ ] Precision
* [ ] Recall
* [ ] Macro F1
* [ ] Weighted F1
* [ ] Per-class metrics
* [ ] Confusion matrix
* [ ] Training curves
* [ ] Validation curves
* [ ] LSTM vs CNN comparison

## Optimization

* [ ] Hyperparameters defined
* [ ] Validation set used for tuning
* [ ] Test set remains untouched
* [ ] Best configuration recorded

## Repository

* [ ] README.md
* [ ] No duplicate Readme.md
* [ ] .gitignore
* [ ] No **pycache**
* [ ] No unnecessary generated files
* [ ] No secrets
* [ ] No unnecessary datasets
* [ ] Setup instructions work
* [ ] Environment reproducible
* [ ] Project structure clean

## Academic

* [ ] Assignment requirements mapped
* [ ] Methodology documented
* [ ] Results are real
* [ ] Limitations documented
* [ ] Conclusion supported by results
* [ ] References included
* [ ] Requirements traceability included

---

# 31. HOW YOU SHOULD EXECUTE THIS TASK

Do not simply give me a list of recommendations.

Actually inspect the repository and implement the required changes.

Follow this sequence:

### STEP 1

Audit the entire repository.

### STEP 2

Create a concise internal gap analysis against the course requirements.

### STEP 3

Fix repository hygiene.

### STEP 4

Fix preprocessing/data leakage issues.

### STEP 5

Fix feature extraction.

### STEP 6

Fix and validate LSTM.

### STEP 7

Fix and validate CNN.

### STEP 8

Fix training and evaluation.

### STEP 9

Fix hyperparameter optimization.

### STEP 10

Clean and reorganize notebooks.

### STEP 11

Rewrite README and documentation.

### STEP 12

Add requirements traceability.

### STEP 13

Run tests/smoke tests.

### STEP 14

Run the full pipeline if the dataset/environment permits.

### STEP 15

Review the entire repository one final time.

---

# 32. FINAL RESPONSE I WANT FROM YOU

After making the changes, do NOT just say "done."

Give me a final implementation report containing:

## A. What you changed

List every important modification.

## B. Files created

List them.

## C. Files modified

List them.

## D. Files deleted

List them.

## E. ML changes

Explain changes to:

* preprocessing
* augmentation
* feature extraction
* LSTM
* CNN
* training
* optimization
* evaluation

## F. Validation

Tell me exactly what you successfully executed.

For example:

```text
Dataset loading: PASS
Preprocessing: PASS
LSTM smoke test: PASS
CNN smoke test: PASS
Evaluation: PASS
Leakage check: PASS
```

## G. Results

Provide actual metrics if successfully generated.

Never invent metrics.

## H. Remaining limitations

Explicitly state anything that could not be completed because of:

* Kaggle authentication
* missing dataset
* computational limitations
* dependency limitations
* GPU limitations
* anything else

## I. Final repository structure

Show the final tree.

## J. Submission readiness

Give me:

```text
Course Requirement
Status
Evidence
```

and explicitly state whether the repository is ready for the AAI-511 final submission.

---

# MOST IMPORTANT INSTRUCTION

The goal is NOT to make the repository look impressive.

The goal is to make it:

**correct + reproducible + academically aligned + experimentally validated + clean + submission-ready.**

The project is a **4-class composer classification problem**, not a music-generation problem.

Do not fabricate data, metrics, results, or validation.

Where something cannot be executed, clearly identify it rather than pretending it works.

Preserve good existing code where possible, but fix anything that violates the requirements above.
