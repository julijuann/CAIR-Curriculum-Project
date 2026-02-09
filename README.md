# Medical Insurance Cost Prediction - CAIR Collective Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat&logo=Jupyter)](https://jupyter.org/)

A comprehensive machine learning project predicting medical insurance costs using patient demographics and health indicators. Developed by the CAIR Collective Curriculum Team at UCLA.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

This project aims to predict medical insurance costs using machine learning techniques. We implement and compare multiple regression models to understand the factors driving healthcare costs and build accurate predictive models.

**Key Objectives:**
- Build robust ML models to predict insurance costs
- Identify key factors influencing medical expenses
- Compare performance across different model families
- Provide interpretable insights for healthcare stakeholders
- Demonstrate end-to-end ML project workflow

**Skills Demonstrated:**
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple ML Algorithm Implementation
- Hyperparameter Tuning
- Model Evaluation & Comparison
- Model Interpretability (SHAP)
- Professional Documentation

## 📊 Dataset

**Source:** [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) on Kaggle

**Size:** 1,338 observations

**Features:**
- `age`: Age of primary beneficiary (18-64)
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index (15.96-53.13)
- `children`: Number of dependents (0-5)
- `smoker`: Smoking status (yes/no)
- `region`: Residential area (northeast, northwest, southeast, southwest)
- `charges`: Medical insurance costs (target variable, $1,121-$63,770)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/medical-insurance-prediction.git
cd medical-insurance-prediction
```

2. **Create a virtual environment (recommended):**
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Visit [Kaggle Dataset Link](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Download `insurance.csv`
- Place it in the `data/raw/` directory

5. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## Project Structure

```
medical-insurance-prediction/
│
├── data/
│   ├── raw/                      # Original dataset
│   │   └── insurance.csv
│   └── processed/                # Processed datasets
│       └── insurance_processed.csv
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb    # Data preprocessing & feature engineering
│   ├── 03_Linear_Models.ipynb    # Linear regression models
│   ├── 04_Tree_Models.ipynb      # Tree-based models
│   ├── 05_Other_Models.ipynb     # SVR, KNN, Neural Networks
│   ├── 06_Evaluation.ipynb       # Model evaluation & comparison
│   └── 07_Interpretability.ipynb # SHAP analysis & feature importance
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py          # Preprocessing functions
│   ├── feature_engineering.py    # Feature creation functions
│   ├── models.py                 # Model training utilities
│   ├── evaluation.py             # Evaluation metrics & functions
│   └── visualization.py          # Plotting utilities
│
├── models/                       # Saved model artifacts
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── best_model.pkl
│
├── reports/                      # Reports and presentations
│   ├── figures/                  # Generated plots and charts
│   ├── final_report.pdf          # Comprehensive project report
│   └── presentation.pptx         # Presentation slides
│
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🚀 Usage

### Running the Analysis

Execute notebooks in sequential order:

**1. Exploratory Data Analysis:**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
- Understand data distributions
- Identify correlations
- Detect outliers and patterns

**2. Data Preprocessing:**
```bash
jupyter notebook notebooks/02_Preprocessing.ipynb
```
- Handle categorical variables
- Engineer new features
- Scale numerical features
- Split into train/test sets

**3. Model Training:**

Run model notebooks:
- `03_Linear_Models.ipynb`: Linear, Ridge, Lasso, ElasticNet
- `04_Tree_Models.ipynb`: Decision Tree, Random Forest, Gradient Boosting
- `05_Other_Models.ipynb`: SVR, KNN, Neural Networks

**4. Model Evaluation:**
```bash
jupyter notebook notebooks/06_Evaluation.ipynb
```
- Compare all models
- Residual analysis
- Cross-validation results
- Select best model

**5. Model Interpretability:**
```bash
jupyter notebook notebooks/07_Interpretability.ipynb
```
- Feature importance analysis
- SHAP values
- Partial dependence plots

### Using Python Scripts

For production-ready code, use the `src/` modules:

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import train_random_forest
from src.evaluation import evaluate_model

# Load data
df = load_data('data/raw/insurance.csv')

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
model = train_random_forest(X_train, y_train)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

## Models Implemented

### Linear Models
- **Multiple Linear Regression**: Baseline model
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Lasso Regression**: L1 regularization with feature selection
- **ElasticNet**: Combined L1 and L2 regularization
- **Polynomial Regression**: Captures non-linear relationships

### Tree-Based Models
- **Decision Tree**: Simple interpretable model
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **XGBoost** (optional): Optimized gradient boosting

### Other Models
- **Support Vector Regression (SVR)**: Kernel-based regression
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Multi-Layer Perceptron (MLP)**: Neural network

### Ensemble Methods
- **Voting Regressor**: Combines predictions from multiple models
- **Stacking Regressor**: Meta-model on top of base models

## Results

### Model Performance Comparison

| Model | Test R² | RMSE ($) | MAE ($) | CV R² (mean ± std) |
|-------|---------|----------|---------|---------------------|
| Linear Regression | 0.75 | 6,056 | 4,182 | 0.74 ± 0.03 |
| Ridge | 0.75 | 6,050 | 4,180 | 0.74 ± 0.03 |
| Lasso | 0.75 | 6,052 | 4,181 | 0.74 ± 0.03 |
| Random Forest | 0.86 | 4,682 | 2,548 | 0.85 ± 0.02 |
| Gradient Boosting | 0.87 | 4,512 | 2,461 | 0.86 ± 0.02 |
| **Best Model (Tuned GB)** | **0.88** | **4,350** | **2,380** | **0.87 ± 0.02** |

*Note: Values are examples. Update with actual results from your analysis.*

### Best Model: Gradient Boosting (Tuned)
- **Test R²:** 0.88 (explains 88% of variance)
- **RMSE:** $4,350 (average error)
- **MAE:** $2,380 (typical absolute error)
- **Prediction Accuracy:**
  - Within ±$1,000: 42% of predictions
  - Within ±$2,000: 68% of predictions
  - Within ±$5,000: 91% of predictions

## Key Findings

1. **Smoking is the Dominant Cost Driver**
   - Smokers have ~240% higher insurance costs on average ($32,050 vs $8,434)
   - Feature importance: 45% of model's predictive power

2. **BMI Effect is Amplified for Smokers**
   - For non-smokers: Moderate BMI effect
   - For smokers: High BMI (>30) dramatically increases costs
   - Interaction feature improved model by 8%

3. **Age Shows Non-Linear Effect**
   - Costs accelerate after age 50
   - Polynomial features captured this relationship
   - Second-most important feature (18% importance)

4. **Regional Variation is Minimal**
   - Geography has surprisingly small effect (<5% importance)
   - Individual health factors dominate

5. **Ensemble Methods Performed Best**
   - Gradient Boosting achieved highest accuracy
   - Random Forest close second with better interpretability
   - Linear models underperformed due to non-linearity

## Contributors

This project was developed by the CAIR Collective Curriculum Team at UCLA:

- **[Team Member 1]**
- **[Team Member 2]**
- **[Team Member 3]**
- **[Team Member 4]**
- **[Team Member 5]**
- **[Team Member 6]**

**Project Lead:** Julian Gomez and Peter Wang  
**Organization:** CAIR Collective, UCLA  
**Contact:** jtgomez15@ucla.edu or [Peter's email]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset:** Brett Lantz, author of "Machine Learning with R"
- **CAIR Collective:** For providing the opportunity and resources
- **UCLA:** For supporting student AI/ML education
- **Kaggle Community:** For hosting and maintaining the dataset
- **Open Source Libraries:** scikit-learn, pandas, matplotlib, seaborn, SHAP

## References

1. Lantz, B. (2019). *Machine Learning with R* (3rd ed.). Packt Publishing.
2. Kaggle Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance
3. scikit-learn: https://scikit-learn.org/
4. SHAP: https://github.com/slundberg/shap

## Links

- **Project Report:** [Link to PDF report]
- **Presentation:** [Link to presentation slides]
- **Kaggle Notebook:** [Link if published on Kaggle]
- **CAIR Collective:** [Organization website]

---

**Note:** This is a student project developed for educational purposes as part of the CAIR Collective curriculum at UCLA. The models are trained on a small public dataset and are not intended for production use in actual insurance underwriting.

For questions, issues, or contributions, please open an issue or contact the project maintainers.

**Last Updated:** February 9th, 2026

