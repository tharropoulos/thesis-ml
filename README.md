# ML-Based Analysis of GitHub Copilot Effectiveness

A machine learning framework for analyzing and evaluating GitHub Copilot's code assistance effectiveness across different programming domains and tasks, using various classification algorithms and visualization techniques.

## Features

- Multiple ML model implementations:
  - Random Forest
  - Support Vector Machines (SVM)
  - Gradient Boosting
  - Naive Bayes
  - Neural Networks (LSTM/GRU)
- Advanced text processing with BERT embeddings
- Comprehensive visualization suite for model performance analysis
- MetaCost implementation for cost-sensitive learning
- Sequential pattern analysis for rating trends
- Subject-wise intervention ratio analysis
- Integration with MySQL for exporting the dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thesis-ml.git
cd thesis-ml
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Export and Preprocessing
```python
# Export data from MySQL database
python src/save-to-csv.py
```

### Rating Distribution Analysis
```python
# Generate visualizations
python src/plot.py


# Analyze subject lengths and patterns
python src/average_subject_length.py

# Run sequential pattern analysis
python src/sequential.py
```

### Training Models
```python
# Train all classification models
python src/random-forest.py
```

### Using the MetaCost Framework
```python
from metaCost import MetaCost

# Initialize MetaCost with your classifier and cost matrix
metacost = MetaCost(training_data, classifier, cost_matrix)
model = metacost.fit(label_column, num_classes)
```

## Configuration

The project uses environment variables for database configuration. Create a `.env` file in the project root with the following variables:

```
DB_HOST=your_host
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_NAME=your_database
```

## Model Outputs

The models generate several visualization outputs in the `plots` directory:
- Confusion matrices
- Model accuracy comparisons
- Subject-wise performance analysis
- Rating distribution charts
- Sequential pattern visualizations

Results and model files are saved in the following directories:
- `/joblibs`: Trained model files
- `/export`: CSV exports and analysis results
- `/plots`: Visualization outputs

## Testing

The project uses standard Python testing frameworks. To run tests:

```bash
python -m pytest tests/
```

## Requirements

Key dependencies include:
- TensorFlow/Keras
- scikit-learn
- BERT (Transformers)
- spaCy
- pandas
- matplotlib/seaborn
- MySQL-python

See `requirements.txt` for a complete list of dependencies.

## Acknowledgments

- The MetaCost implementation is based on the paper "MetaCost: A General Method for Making Classifiers Cost-Sensitive"
- Visualization styling uses the Catppuccin color palette
- BERT implementation uses the Hugging Face Transformers library
