# Setup Guide for Composer Classification Project

## Prerequisites

- Python 3.9-3.10 (recommended for TensorFlow 2.13+ compatibility)
- Conda or Miniconda (recommended) OR pip
- Git (optional)

## Installation

### Option 1: Using Conda (Recommended)

1. **Clone or download the project repository**

2. **Navigate to the project directory**
   ```bash
   cd AAI_511_Group6_deep-learning-music-composer
   ```

3. **Create conda environment from environment.yml**
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the conda environment**
   ```bash
   conda activate composer_classification
   ```

5. **Verify installation**
   ```bash
   python --version  # Should show Python 3.10.x
   pip list          # Should show all required packages
   ```

### Option 2: Using Virtual Environment (pip)

1. **Clone or download the project repository**

2. **Navigate to the project directory**
   ```bash
   cd AAI_511_Group6_deep-learning-music-composer
   ```

3. **Create a virtual environment with Python 3.10**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Upgrade pip**
   ```bash
   pip install --upgrade pip
   ```

5. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify installation**
   ```bash
   python --version  # Should show Python 3.9.x or 3.10.x
   pip list          # Should show all required packages
   ```

## Dataset Setup

The project expects MIDI files to be organized in the following structure:

```
data/
    raw/
        Bach/
        Beethoven/
        Chopin/
        Mozart/
```

If you have the dataset in a different location, update the `RAW_DATA_DIR` path in `src/utils/config.py`.

## Running the Project

### Option 1: Using Jupyter Notebooks

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open and run the notebooks in order:**
   - `01_EDA.ipynb` - Exploratory Data Analysis
   - `02_Preprocessing.ipynb` - Data cleaning and splitting
   - `03_Feature_Extraction.ipynb` - Feature extraction for LSTM and CNN
   - `04_LSTM_Model.ipynb` - LSTM model training
   - `05_CNN_Model.ipynb` - CNN model training
   - `06_Hyperparameter_Tuning.ipynb` - Hyperparameter optimization (optional)
   - `Composer_Classification_Final.ipynb` - Complete end-to-end pipeline

### Option 2: Using Python Scripts

You can also use the modular Python scripts directly:

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))

# Import and use modules
from preprocessing import load_midi_dataset, clean_dataset, split_dataset
from datasets import prepare_lstm_dataset, prepare_cnn_dataset
from models import create_lstm_model, create_cnn_model
from training import train_lstm_model, train_cnn_model
from evaluation import ModelEvaluator, ModelVisualizer
```

## Configuration

All project settings are centralized in `src/utils/config.py`:

- **Data paths**: Dataset directories, model save locations
- **Model parameters**: LSTM/CNN architecture settings
- **Training parameters**: Batch size, epochs, learning rates
- **Feature extraction**: Sequence lengths, piano roll settings

Modify these parameters as needed for your experiments.

## Project Structure

```
AAI_511_Group6_deep-learning-music-composer/
├── data/
│   ├── raw/              # Original MIDI files
│   ├── interim/          # Cleaned metadata and splits
│   ├── processed/        # Processed datasets
│   └── features/         # Extracted features
├── models/
│   ├── checkpoints/      # Training checkpoints
│   └── saved_models/     # Final trained models
├── notebooks/            # Jupyter notebooks
├── reports/
│   ├── figures/          # Generated plots
│   └── tables/           # Result tables
├── src/
│   ├── preprocessing/    # Data loading and cleaning
│   ├── features/         # Feature extraction
│   ├── datasets/         # Dataset preparation
│   ├── models/           # Model architectures
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Configuration and helpers
├── requirements.txt
├── .gitignore
└── README.md
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've activated the virtual environment and installed requirements
2. **MIDI loading errors**: Verify MIDI files are in the correct directory structure
3. **Memory issues**: Reduce batch size or sequence length in config.py
4. **GPU not detected**: TensorFlow will automatically use CPU if GPU is unavailable

### Getting Help

- Check the inline documentation in each module
- Review the Jupyter notebooks for usage examples
- Examine the README.md for detailed project information

## Next Steps

1. Run the EDA notebook to understand your dataset
2. Follow the preprocessing notebook to clean and split data
3. Extract features using the feature extraction notebook
4. Train LSTM and CNN models
5. Evaluate and compare model performance
6. Generate the final report for submission

## Citation

If you use this code for your project, please cite appropriately in your final report.
