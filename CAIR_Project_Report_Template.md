# Medical Insurance Cost Prediction Using Machine Learning
## A Comprehensive Analysis and Modeling Approach

**CAIR Collective - UCLA**  
**Curriculum Team Project**  
**Date: February 9th, 2026**

---

## Executive Summary Example

> **Note to students**: This should be written LAST. Summarize the entire project in 1 page maximum.

### Problem Statement
[Brief description of the problem: predicting medical insurance costs based on patient demographics and health indicators]

### Key Findings
- [Finding 1: Best performing model and its accuracy]
- [Finding 2: Most important features]
- [Finding 3: Key insights about healthcare costs]

### Recommendations
- [Recommendation 1]
- [Recommendation 2]

### Impact
[Brief statement about how this model could be used in practice]

---

## Table of Contents

1. Introduction
2. Background and Motivation
3. Dataset Description
4. Methodology
5. Exploratory Data Analysis
6. Feature Engineering
7. Model Development
8. Results and Evaluation
9. Model Interpretability
10. Discussion
11. Limitations and Future Work
12. Conclusion
13. References
14. Appendices

---

## 1. Introduction

### 1.1 Problem Context

Medical insurance costs in the United States have been rising steadily over the past decades. Understanding the factors that drive these costs is crucial for insurance companies, policymakers, and individuals. [Expand with statistics and context]

### 1.2 Objective

The primary objective of this project is to develop a machine learning model that can accurately predict medical insurance costs based on individual characteristics. Specifically, we aim to:

1. Identify the key factors that influence medical insurance costs
2. Build and compare multiple predictive models
3. Provide interpretable insights that can inform decision-making
4. Demonstrate the application of ML techniques in healthcare analytics

### 1.3 Significance

[Explain why this problem matters in healthcare, insurance industry, and for individuals]

---

## 2. Background and Motivation

### 2.1 Healthcare Costs in the United States

[Provide context about US healthcare system, insurance, and cost factors]

### 2.2 Related Work

[Cite 3-5 relevant papers or projects on medical cost prediction]

Example references to include:
- Studies on factors affecting healthcare costs
- Previous ML applications in healthcare
- Insurance industry reports

### 2.3 Machine Learning in Healthcare

[Brief overview of how ML is being used in healthcare, focusing on predictive modeling]

---

## 3. Dataset Description

### 3.1 Data Source

- **Dataset Name**: Medical Cost Personal Datasets
- **Source**: Kaggle (https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Original Source**: Book "Machine Learning with R" by Brett Lantz
- **Size**: 1,338 observations
- **Format**: CSV file

### 3.2 Features

The dataset contains the following features:

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| age | Numerical | Age of the primary beneficiary | 18-64 years |
| sex | Categorical | Gender of the beneficiary | male, female |
| bmi | Numerical | Body Mass Index | 15.96-53.13 |
| children | Numerical | Number of dependents | 0-5 |
| smoker | Categorical | Smoking status | yes, no |
| region | Categorical | Residential area in the US | northeast, northwest, southeast, southwest |
| charges | Numerical (Target) | Medical insurance costs | $1,121.87 - $63,770.43 |

### 3.3 Data Quality Assessment

[Discuss]:
- Missing values: [None found]
- Duplicates: [X duplicates identified and removed]
- Outliers: [Identified in charges and BMI]
- Data balance: [Distribution across categorical variables]

---

## 4. Methodology

### 4.1 Overall Approach

Our methodology follows the standard machine learning pipeline:

```
Data Collection → EDA → Feature Engineering → Preprocessing → 
Model Training → Evaluation → Interpretation → Deployment
```

### 4.2 Exploratory Data Analysis

**Techniques Used**:
- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Visualization (histograms, box plots, scatter plots, heatmaps)

### 4.3 Feature Engineering

**Engineered Features**:
1. **BMI Categories**: Clinical classifications (underweight, normal, overweight, obese)
2. **Age Groups**: Young adult (<30), middle-age (30-50), senior (>50)
3. **Polynomial Features**: age², bmi²
4. **Interaction Features**: age × bmi, smoker × bmi
5. **High-Risk Flag**: Binary indicator for smokers with high BMI
6. **Family Size**: Total household size (children + 1)

**Rationale**:
[Explain the healthcare logic behind each engineered feature]

### 4.4 Data Preprocessing

**Steps Performed**:
1. **Encoding Categorical Variables**:
   - Binary encoding for sex (male=1, female=0)
   - Binary encoding for smoker (yes=1, no=0)
   - One-hot encoding for region (drop first to avoid multicollinearity)

2. **Feature Scaling**:
   - StandardScaler applied to all numerical features
   - Comparison with MinMaxScaler and RobustScaler

3. **Train-Test Split**:
   - 80% training, 20% testing
   - Random state = 42 for reproducibility

### 4.5 Model Selection

**Models Implemented**:

**Linear Models**:
- Multiple Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (combined L1/L2)
- Polynomial Regression

**Tree-Based Models**:
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- [Optional: XGBoost, LightGBM]

**Other Models**:
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regressor
- Multi-Layer Perceptron (Neural Network)

**Ensemble Methods**:
- Voting Regressor
- Stacking Regressor

### 4.6 Hyperparameter Tuning

**Approach**:
- 5-fold Cross-Validation
- GridSearchCV for comprehensive search
- RandomizedSearchCV for efficiency on large parameter spaces

**Tuned Hyperparameters** (example for Random Forest):
- n_estimators: [50, 100, 200]
- max_depth: [5, 10, 15, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

### 4.7 Evaluation Metrics

**Primary Metrics**:
1. **R² Score**: Proportion of variance explained
2. **RMSE**: Root Mean Squared Error (in dollars)
3. **MAE**: Mean Absolute Error (interpretable in dollars)
4. **Cross-Validation Score**: Mean and standard deviation across folds

**Additional Metrics**:
- Adjusted R²
- MAPE (Mean Absolute Percentage Error)
- Prediction accuracy within tolerance bands (±$1000, ±$2000, ±$5000)

**Rationale for Metrics**:
[Explain why these metrics are appropriate for this problem]

---

## 5. Exploratory Data Analysis

### 5.1 Univariate Analysis

#### 5.1.1 Target Variable (Charges)

[Insert visualization: Distribution of charges]

**Key Observations**:
- Distribution is right-skewed
- Mean: $[X], Median: $[X]
- Range: $[min] to $[max]
- Potential outliers detected above $[threshold]

#### 5.1.2 Numerical Features

**Age**:
- Distribution: [describe]
- Mean: [X] years, Std: [X] years
- Range: 18-64 years

**BMI**:
- Distribution: [describe]
- Mean: [X], Std: [X]
- Clinical interpretation: [percentage in each BMI category]

**Children**:
- Distribution: [describe]
- Mode: [most common number]
- Most beneficiaries have [0-2] children

#### 5.1.3 Categorical Features

**Sex**:
- Distribution: [X]% male, [X]% female
- Nearly balanced

**Smoker**:
- Distribution: [X]% smokers, [X]% non-smokers
- Important: smoking status is minority class

**Region**:
- Distribution across four regions
- [Percentages for each region]

### 5.2 Bivariate Analysis

#### 5.2.1 Charges vs Numerical Features

[Insert visualizations: Scatter plots]

**Age vs Charges**:
- Correlation: [X]
- Pattern: [describe - positive relationship, non-linear]
- Interpretation: [older individuals tend to have higher costs]

**BMI vs Charges**:
- Correlation: [X]
- Pattern: [describe]
- Different patterns observed for smokers vs non-smokers

**Children vs Charges**:
- Correlation: [X]
- Pattern: [weak or no clear pattern]

#### 5.2.2 Charges vs Categorical Features

[Insert visualizations: Box plots]

**Charges by Smoker Status**:
- Median charges for smokers: $[X]
- Median charges for non-smokers: $[X]
- **Key Finding**: Smokers have dramatically higher costs ([X]% higher)

**Charges by Sex**:
- Males: $[X] (median)
- Females: $[X] (median)
- Difference: [minimal/moderate/significant]

**Charges by Region**:
- Regional variation: [describe]
- Highest: [region], Lowest: [region]
- Possible factors: [healthcare costs, demographics]

### 5.3 Multivariate Analysis

#### 5.3.1 Correlation Matrix

[Insert visualization: Heatmap]

**Key Correlations with Charges**:
1. Smoker: [X] (strongest predictor)
2. Age: [X]
3. BMI: [X]
4. [Other features]

**Multicollinearity Check**:
- [No high correlations between features]
- [VIF analysis if performed]

#### 5.3.2 Feature Interactions

**Smoker × BMI Interaction**:
[Insert visualization showing this critical interaction]
- For non-smokers: BMI has [weak/moderate] effect on charges
- For smokers: High BMI dramatically increases charges
- This motivated the creation of our "high_risk" engineered feature

**Age × Smoker Interaction**:
[Describe pattern]

### 5.4 Outlier Analysis

**Detected Outliers**:
- [X] observations with charges > $[threshold]
- [X] observations with BMI > [threshold]

**Treatment Decision**:
- [Kept outliers as they represent real high-cost cases]
- [Or: Applied Winsorization/capping]
- Rationale: [explain]

### 5.5 Key Insights from EDA

1. **Smoking is the dominant cost driver**: Smokers have [X]% higher costs on average
2. **BMI effect is moderated by smoking status**: Interaction effect observed
3. **Age shows non-linear relationship**: Suggests polynomial features may help
4. **Regional variations exist**: But effect is smaller than individual health factors
5. **Data is generally clean**: No missing values, minimal preprocessing needed

---

## 6. Feature Engineering

### 6.1 Rationale

[Explain why feature engineering is important for this problem]

### 6.2 Created Features

#### 6.2.1 BMI Categories

**Implementation**:
```python
if bmi < 18.5: 'underweight'
elif bmi < 25: 'normal'
elif bmi < 30: 'overweight'
else: 'obese'
```

**Medical Justification**:
[Cite WHO/CDC BMI classifications]

**Impact on Model**:
[Allows model to capture non-linear BMI effects]

#### 6.2.2 Age Groups

**Implementation**:
```python
if age < 30: 'young_adult'
elif age < 50: 'middle_age'
else: 'senior'
```

**Rationale**:
[Healthcare costs tend to increase non-linearly with age brackets]

#### 6.2.3 Polynomial Features

**Created**: age², bmi²

**Rationale**:
- Captures non-linear relationships
- Marginal costs may increase at higher values

**Analysis**:
[Correlation of polynomial features with target]

#### 6.2.4 Interaction Features

**Smoker × BMI**:
- **Hypothesis**: The effect of BMI on costs is amplified for smokers
- **Evidence**: [Show from EDA]
- **Feature**: `smoker_bmi_interaction = smoker * bmi`

**Age × BMI**:
- **Hypothesis**: Older individuals with high BMI face compounded risks
- **Feature**: `age_bmi_interaction = age * bmi`

#### 6.2.5 High-Risk Flag

**Definition**: Binary indicator for individuals who are both smokers AND obese (BMI ≥ 30)

**Medical Justification**:
[Combined risk factors for cardiovascular disease, diabetes, etc.]

**Distribution**:
- [X]% of dataset flagged as high-risk
- Average charges for high-risk: $[X] vs $[Y] for others

#### 6.2.6 Family Size

**Feature**: `family_size = children + 1`

**Rationale**:
- Family coverage may have different cost structure
- Captures household context

### 6.3 Feature Selection

After engineering features, we performed feature selection to avoid overfitting:

**Methods Used**:
1. **Univariate Selection** (SelectKBest with f_regression)
2. **Recursive Feature Elimination** (RFE with Random Forest)
3. **L1 Regularization** (Lasso for embedded selection)

**Selected Features** (Top 10):
1. [Feature name] - Score: [X]
2. [Feature name] - Score: [X]
[... continue]

### 6.4 Impact of Feature Engineering

[Create table showing model performance before and after feature engineering]

| Model | R² (Original) | R² (Engineered) | Improvement |
|-------|---------------|-----------------|-------------|
| Linear Regression | 0.XX | 0.XX | +X% |
| Random Forest | 0.XX | 0.XX | +X% |

**Conclusion**: Feature engineering improved model performance by [X]% on average.

---

## 7. Model Development

### 7.1 Baseline Model

**Simple Mean Baseline**:
- Prediction: Always predict mean charges ($[X])
- R²: 0.00 (by definition)
- RMSE: $[X]
- MAE: $[X]

**Simple Linear Regression** (best single predictor):
- Feature: [smoker status]
- R²: [X]
- RMSE: $[X]

### 7.2 Linear Models

#### 7.2.1 Multiple Linear Regression

**Model**: OLS with all features

**Assumptions Checked**:
1. Linearity: [checked with residual plots]
2. Homoscedasticity: [checked with Breusch-Pagan test]
3. Normality of residuals: [checked with Q-Q plot]
4. No multicollinearity: [checked with VIF]

**Results**:
- Train R²: [X]
- Test R²: [X]
- CV R² (mean ± std): [X] ± [X]
- RMSE: $[X]
- MAE: $[X]

**Significant Predictors** (p < 0.05):
[List with coefficients and interpretation]

#### 7.2.2 Ridge Regression (L2)

**Hyperparameter**:
- Best alpha (via GridSearch): [X]

**Results**:
- Test R²: [X]
- RMSE: $[X]
- MAE: $[X]

**Comparison to OLS**:
[Did regularization improve generalization?]

#### 7.2.3 Lasso Regression (L1)

**Hyperparameter**:
- Best alpha: [X]

**Results**:
- Test R²: [X]
- RMSE: $[X]
- MAE: $[X]

**Feature Selection**:
- Features retained: [X] out of [total]
- Features eliminated: [list]

**Insight**: Lasso automatically performed feature selection, removing [X] features with little predictive power.

#### 7.2.4 ElasticNet

**Hyperparameters**:
- Best alpha: [X]
- Best l1_ratio: [X]

**Results**:
- Test R²: [X]
- RMSE: $[X]

**Analysis**: ElasticNet balanced between Ridge and Lasso characteristics.

#### 7.2.5 Polynomial Regression

**Degree**: [2 or 3]

**Results**:
- Test R²: [X]
- RMSE: $[X]

**Overfitting Check**:
- Train R²: [X]
- Gap between train and test: [X]
- [Conclusion about overfitting]

### 7.3 Tree-Based Models

#### 7.3.1 Decision Tree

**Hyperparameters**:
- max_depth: [X]
- min_samples_split: [X]

**Results**:
- Test R²: [X]
- RMSE: $[X]

**Feature Importance** (Top 5):
1. [Feature]: [X]%
2. [Feature]: [X]%
[continue]

**Tree Visualization**:
[Insert if helpful - but usually too large for full dataset]

**Analysis**:
- Prone to overfitting (train R² much higher than test)
- Captures non-linear relationships well
- Serves as baseline for ensemble methods

#### 7.3.2 Random Forest

**Hyperparameters** (after tuning):
- n_estimators: [X]
- max_depth: [X]
- min_samples_split: [X]
- min_samples_leaf: [X]

**Results**:
- Test R²: [X]
- RMSE: $[X]
- MAE: $[X]
- CV R² (mean ± std): [X] ± [X]

**Feature Importance**:
[Insert bar plot]

**Top Features**:
1. [Feature]: [X]%
2. [Feature]: [X]%
[continue]

**Analysis**:
- Strong performance, reduced overfitting compared to single tree
- Consistent CV scores indicate good generalization
- Feature importances align with EDA insights (smoker most important)

#### 7.3.3 Gradient Boosting

**Hyperparameters** (after tuning):
- n_estimators: [X]
- learning_rate: [X]
- max_depth: [X]

**Results**:
- Test R²: [X]
- RMSE: $[X]
- MAE: $[X]

**Learning Curves**:
[Insert plot showing training progress]

**Analysis**:
[Compare to Random Forest, discuss bias-variance tradeoff]

### 7.4 Other Models

#### 7.4.1 Support Vector Regression (SVR)

**Kernel**: [RBF/linear/poly]
**Hyperparameters**:
- C: [X]
- epsilon: [X]
- gamma: [X] (if RBF)

**Results**:
- Test R²: [X]
- RMSE: $[X]

**Analysis**:
[Performance, computational cost, interpretability]

#### 7.4.2 K-Nearest Neighbors

**Hyperparameter**:
- Best k: [X] (via GridSearch)

**Results**:
- Test R²: [X]
- RMSE: $[X]

**Analysis**:
[Discuss performance, effect of scaling, sensitivity to k]

#### 7.4.3 Neural Network (MLP)

**Architecture**:
- Hidden layers: [X] layers of [Y] neurons each
- Activation: [relu/tanh]
- Optimizer: [adam]

**Results**:
- Test R²: [X]
- RMSE: $[X]

**Training Curves**:
[Insert loss curves]

**Analysis**:
[Did deep learning help? Overfitting concerns?]

### 7.5 Ensemble Methods

#### 7.5.1 Voting Regressor

**Base Models**:
1. Random Forest (weight: [X])
2. Gradient Boosting (weight: [X])
3. Ridge Regression (weight: [X])

**Results**:
- Test R²: [X]
- RMSE: $[X]
- MAE: $[X]

**Analysis**:
[Did ensemble outperform individual models?]

#### 7.5.2 Stacking Regressor

**Base Models**:
1. Random Forest
2. Gradient Boosting
3. Ridge Regression

**Meta-Model**: [Linear Regression/Ridge]

**Results**:
- Test R²: [X]
- RMSE: $[X]
- MAE: $[X]

**Analysis**:
[Performance vs voting, meta-model insights]

---

## 8. Results and Evaluation

### 8.1 Model Comparison

[Insert comprehensive comparison table]

| Model | Train R² | Test R² | CV R² | RMSE ($) | MAE ($) | Training Time (s) |
|-------|----------|---------|-------|----------|---------|-------------------|
| Linear Regression | 0.XX | 0.XX | 0.XX ± 0.XX | X,XXX | X,XXX | X.XX |
| Ridge | 0.XX | 0.XX | 0.XX ± 0.XX | X,XXX | X,XXX | X.XX |
| Lasso | 0.XX | 0.XX | 0.XX ± 0.XX | X,XXX | X,XXX | X.XX |
| Random Forest | 0.XX | 0.XX | 0.XX ± 0.XX | X,XXX | X,XXX | X.XX |
| Gradient Boosting | 0.XX | 0.XX | 0.XX ± 0.XX | X,XXX | X,XXX | X.XX |
| ... | ... | ... | ... | ... | ... | ... |

[Insert bar charts comparing models across different metrics]

### 8.2 Best Model Selection

**Selected Model**: [Model Name]

**Justification**:
1. **Highest Test R²**: [X], explaining [X]% of variance in medical costs
2. **Consistent CV Performance**: Low standard deviation ([X]) indicates stability
3. **Reasonable RMSE**: $[X] average prediction error
4. **Good Generalization**: Small gap between train and test performance
5. **Interpretability**: [If relevant - tree models offer feature importance]
6. **Practical Considerations**: [Training time, deployment ease, etc.]

### 8.3 Detailed Evaluation of Best Model

#### 8.3.1 Performance Metrics

**Regression Metrics**:
- R² Score: [X]
- Adjusted R²: [X]
- RMSE: $[X]
- MAE: $[X]
- MAPE: [X]%

**Interpretation**:
- On average, predictions are within $[MAE] of actual costs
- [X]% of variance in charges is explained by the model

#### 8.3.2 Residual Analysis

[Insert residual plots]

**1. Residuals vs Fitted Values**:
- Pattern: [Random scatter indicates good fit / Pattern indicates issue]
- Heteroscedasticity: [Present/Absent]

**2. Normal Q-Q Plot**:
- Residuals approximately follow normal distribution: [Yes/No]
- Deviations at [tails/middle]: [Interpretation]

**3. Histogram of Residuals**:
- Shape: [Approximately normal / Skewed]
- Mean: [Close to zero]
- Outliers: [X large residuals identified]

#### 8.3.3 Prediction Analysis

[Insert Actual vs Predicted scatter plot]

**Observations**:
- Points cluster around the diagonal line
- Some systematic [under/over]prediction for [high/low] cost patients
- Largest prediction errors occur for charges > $[X]

**Error Distribution by Segment**:

| Segment | N | MAE ($) | MAPE (%) |
|---------|---|---------|----------|
| Charges < $10k | X | X,XXX | X% |
| Charges $10k-$20k | X | X,XXX | X% |
| Charges $20k-$30k | X | X,XXX | X% |
| Charges > $30k | X | X,XXX | X% |

**Insight**: Model performs [better/worse] on [high/low] cost patients.

#### 8.3.4 Business-Relevant Metrics

**Prediction Accuracy within Tolerance Bands**:
- Within ±$1,000: [X]% of predictions
- Within ±$2,000: [X]% of predictions
- Within ±$5,000: [X]% of predictions

**High-Cost Patient Prediction** (charges > $30,000):
- Recall: [X]%
- Precision: [X]%
- F1-Score: [X]

[These metrics matter for identifying patients who need cost management interventions]

### 8.4 Cross-Validation Results

[Insert box plot of CV scores across folds]

**5-Fold CV R² Scores**:
- Fold 1: [X]
- Fold 2: [X]
- Fold 3: [X]
- Fold 4: [X]
- Fold 5: [X]
- Mean: [X]
- Std Dev: [X]

**Interpretation**: Low standard deviation indicates model is stable and not overly sensitive to specific train-test splits.

### 8.5 Learning Curves

[Insert learning curves]

**Analysis**:
- **Training score**: [Converges to X]
- **Validation score**: [Converges to X]
- **Gap**: [Small/Large]
- **Diagnosis**: 
  - [If large gap: Some overfitting present]
  - [If both low: High bias, underfitting]
  - [If both high and close: Good fit]
- **Data sufficiency**: [More data would help: Yes/No]

### 8.6 Validation Curves (Hyperparameter Analysis)

[If performed - insert validation curves for key hyperparameters]

Example for Random Forest `n_estimators`:
- Performance stabilizes at [X] estimators
- Minimal improvement beyond this point
- Computational cost increases linearly

---

## 9. Model Interpretability

### 9.1 Feature Importance

[Insert feature importance bar chart]

**Top 10 Features** (from best model):

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | smoker | 0.XX | Smoking status is the dominant predictor |
| 2 | age | 0.XX | Older individuals have higher costs |
| 3 | bmi | 0.XX | BMI significantly affects costs |
| 4 | smoker_bmi_interaction | 0.XX | Combined effect of smoking and obesity |
| 5 | age_squared | 0.XX | Non-linear age effects |
| ... | ... | ... | ... |

**Key Insights**:
1. **Smoking dominates**: Accounts for [X]% of model's predictive power
2. **Demographics matter**: Age is second-most important
3. **Interactions are valuable**: Engineered interaction features rank high
4. **Regional effects minimal**: Region features have low importance

### 9.2 SHAP Analysis

[If performed - SHAP (SHapley Additive exPlanations) provides detailed interpretability]

#### 9.2.1 SHAP Summary Plot

[Insert SHAP summary plot]

**Interpretation**:
- Red = high feature value, Blue = low feature value
- Points to right increase prediction, left decrease
- Width of spread shows feature impact distribution

**Key Patterns**:
1. **Smoker = Yes (red)**: Dramatically pushes predictions higher
2. **High age (red)**: Increases predicted costs
3. **High BMI (red)**: Increases costs, especially for smokers

#### 9.2.2 SHAP Dependence Plots

[Insert SHAP dependence plots for top features]

**Age Dependence**:
- Non-linear relationship confirmed
- Effect accelerates after age [X]

**BMI Dependence**:
- Effect varies by smoking status (color coding)
- Stronger BMI effect for smokers

#### 9.2.3 Individual Prediction Explanation

[Pick 2-3 example predictions and explain using SHAP]

**Example 1: High-Cost Patient**
- Actual: $[X], Predicted: $[X]
- Base value (average): $[X]
- smoker = yes: +$[X]
- age = [X]: +$[X]
- bmi = [X]: +$[X]
- Other features: ±$[X]

**Example 2: Low-Cost Patient**
- [Similar breakdown]

### 9.3 Partial Dependence Plots

[Insert PDPs for key features]

**Interpretation**:
- Shows marginal effect of feature on predictions
- Averaged over other features
- Reveals non-linearities and thresholds

**Key Findings**:
1. **Age**: Nearly linear increase up to age 50, then steeper
2. **BMI**: Threshold effect around BMI = 30 (obesity)
3. **Smoker**: Categorical step function (+$[X] for smokers)

### 9.4 Feature Interactions

[Analyze 2D PDPs or interaction effects]

**Smoker × BMI Interaction**:
[Insert 2D contour plot]
- For non-smokers: BMI effect is moderate
- For smokers: BMI effect is amplified [X]-fold
- Highest costs: Smokers with BMI > 35

### 9.5 Model Confidence and Uncertainty

[If prediction intervals were calculated]

**Prediction Interval Coverage**:
- 95% intervals covered [X]% of actual values
- Intervals wider for high-uncertainty predictions
- Narrower for typical cases

[Insert plot showing predictions with confidence intervals]

---

## 10. Discussion

### 10.1 Key Findings

**Finding 1: Smoking is the Dominant Cost Driver**
- Smokers have [X]% higher insurance costs on average
- This aligns with extensive medical research on smoking health effects
- Policy implication: Smoking cessation programs could reduce costs

**Finding 2: BMI Effect is Moderated by Smoking**
- For non-smokers, BMI has moderate effect
- For smokers, high BMI dramatically amplifies costs
- Suggests compound health risks from multiple factors

**Finding 3: Age Shows Non-Linear Effects**
- Costs increase with age, but acceleration after 50
- Supports need for age-specific pricing models
- Preventive care investment in middle age could reduce later costs

**Finding 4: Regional Variation is Minimal**
- Geography has surprisingly small effect
- Individual health factors dominate over location
- May reflect standardization in insurance markets

**Finding 5: Ensemble Methods Offer Best Performance**
- [If applicable] Combining multiple models improved predictions
- Random Forest/Gradient Boosting captured complex patterns
- Trade-off: Better accuracy but harder to interpret

### 10.2 Healthcare Implications

#### 10.2.1 For Insurance Companies

1. **Risk Assessment**: Model can improve underwriting accuracy
2. **Pricing**: Better cost prediction enables fair premium calculation
3. **Intervention Targeting**: Identify high-risk individuals for wellness programs
4. **Cost Management**: Predict and prepare for high-cost claims

#### 10.2.2 For Policymakers

1. **Public Health Priorities**: Confirms importance of smoking prevention
2. **Healthcare Access**: Insights on cost drivers inform policy design
3. **Preventive Care**: Evidence for investing in obesity and smoking programs
4. **Regulation**: Informs fair insurance practice guidelines

#### 10.2.3 For Individuals

1. **Cost Awareness**: Understanding personal risk factors
2. **Behavior Change**: Quantified impact of smoking cessation, weight management
3. **Financial Planning**: Better estimate of healthcare expenses
4. **Empowerment**: Data-driven health decisions

### 10.3 Model Comparison Insights

**Why [Best Model] Performed Best**:
- Captures non-linear relationships (vs linear models)
- Handles feature interactions naturally
- Resistant to overfitting (vs single decision tree)
- Good bias-variance tradeoff

**Why Linear Models Fell Short**:
- Strong assumptions (linearity, homoscedasticity)
- Cannot capture complex interactions without engineering
- However: Much more interpretable (explicit coefficients)

**When to Use Which Model**:
- **Interpretation Priority**: Use Ridge/Lasso (linear model with regularization)
- **Accuracy Priority**: Use Gradient Boosting or Stacking
- **Balance**: Use Random Forest
- **Deployment Constraints**: Consider model size, prediction speed

### 10.4 Feature Engineering Impact

Our engineered features improved model performance by [X]%. Key successes:

1. **Smoker × BMI Interaction**: Captured amplification effect
2. **Polynomial Age**: Addressed non-linearity
3. **BMI Categories**: Aligned with clinical standards
4. **High-Risk Flag**: Simplified complex interaction

**Lesson**: Domain knowledge + data exploration → better features → better models

### 10.5 Ethical Considerations

#### 10.5.1 Fairness and Bias

**Sex-Based Pricing Concerns**:
- Model uses sex as feature
- [Analyze]: Does model show bias? Prediction parity across sex?
- Consideration: Many jurisdictions prohibit sex-based insurance pricing
- Recommendation: [Remove sex feature / Monitor for disparate impact]

**Regional Disparities**:
- Model includes region
- Could perpetuate geographic health inequalities
- Need to ensure not proxying for race or socioeconomic status

#### 10.5.2 Privacy

- Dataset is anonymized, no individual identifiers
- Aggregated insights only, not individual-level exposure
- Deployment must comply with HIPAA regulations
- Consider: Differential privacy techniques for added protection

#### 10.5.3 Transparency and Explainability

- **Black Box Concern**: Ensemble models are complex
- **Mitigation**: SHAP analysis provides explanations
- **Stakeholder Communication**: Need non-technical explanations
- **Right to Explanation**: Individuals should understand factors affecting their rates

#### 10.5.4 Potential Harms

**Discrimination Risk**:
- Using health factors (BMI, smoking) could be seen as penalizing illness
- However: These are modifiable risk factors under individual control
- Balance: Incentivizing healthy behavior vs penalizing vulnerable populations

**Access to Care**:
- Accurate cost prediction could make insurance unaffordable for high-risk individuals
- Tension: Actuarial fairness vs universal access
- Policy question beyond model scope

#### 10.5.5 Recommendations

1. **Regular Bias Audits**: Monitor model performance across demographic groups
2. **Human Oversight**: Models assist, not replace, human underwriters
3. **Transparency**: Disclose to customers what factors affect their premiums
4. **Continuous Improvement**: Update model as new data and research emerge
5. **Ethical Review**: Subject model use to institutional review

### 10.6 Comparison to Prior Work

[Compare results to published benchmarks or Kaggle competition results]

**Our R² of [X] compares favorably to**:
- [Study/Kaggle kernel 1]: R² = [X]
- [Study/Kaggle kernel 2]: R² = [X]
- [Study/Kaggle kernel 3]: R² = [X]

**Novel Contributions**:
- [Specific engineered features]
- [Comprehensive ensemble comparison]
- [SHAP interpretability analysis]

---

## 11. Limitations and Future Work

### 11.1 Limitations

#### 11.1.1 Data Limitations

1. **Dataset Size**: 
   - Only 1,338 observations
   - May not fully capture population diversity
   - Limited representation of rare conditions

2. **Feature Completeness**:
   - Missing factors: pre-existing conditions, medications, family history
   - No geographic granularity (state, urban/rural)
   - No temporal data (claims over time)
   - No lifestyle factors (exercise, diet, alcohol)

3. **Data Currency**:
   - Dataset from [year]
   - Healthcare costs and practices have changed
   - COVID-19 impact not reflected

4. **Representativeness**:
   - Sample may not represent full US population
   - Selection bias in data collection unknown
   - Generalization to other countries limited

#### 11.1.2 Model Limitations

1. **Point Estimates Only**:
   - Most models give single prediction
   - No quantification of uncertainty
   - Prediction intervals would be valuable

2. **Static Predictions**:
   - No longitudinal modeling
   - Cannot predict cost trajectory over time
   - Real insurance needs multi-year forecasting

3. **Interpretability Trade-offs**:
   - Best performing models (ensembles) are complex
   - SHAP helps but doesn't fully solve black box problem

4. **Deployment Considerations**:
   - Model requires retraining as data changes
   - Computational cost of ensemble predictions
   - Integration with existing insurance systems

#### 11.1.3 Scope Limitations

1. **Narrow Problem Definition**:
   - Only predicting costs, not health outcomes
   - No causal inference (correlation ≠ causation)
   - No prescription of interventions

2. **Validation**:
   - No external validation dataset
   - No real-world deployment testing
   - No comparison to actual insurance company models

### 11.2 Future Work

#### 11.2.1 Data Enhancement

**Additional Features**:
- Pre-existing conditions (diabetes, heart disease, etc.)
- Medication usage
- Family medical history
- Behavioral factors (exercise frequency, diet quality)
- Mental health indicators
- Occupation and stress factors

**External Data Sources**:
- Merge with public health datasets
- Incorporate economic indicators (inflation, regional healthcare costs)
- Add environmental factors (air quality, access to care)

**Temporal Data**:
- Multi-year claim history
- Cost trajectories and trends
- Seasonal patterns in healthcare utilization

#### 11.2.2 Advanced Modeling Techniques

**1. Deep Learning**:
- Fully-connected neural networks with more layers
- Investigate if added complexity improves predictions
- Compare to tree-based methods

**2. Quantile Regression**:
- Predict full distribution of costs, not just mean
- Identify factors affecting cost variability
- Useful for risk management

**3. Causal Inference**:
- Propensity score matching
- Instrumental variables
- Understand causal impact of interventions (e.g., smoking cessation)

**4. Time Series Models**:
- If longitudinal data available
- ARIMA, Prophet for cost forecasting
- Recurrent neural networks (LSTM) for sequential data

**5. Bayesian Approaches**:
- Uncertainty quantification
- Prior knowledge incorporation
- Probabilistic predictions

**6. AutoML**:
- Automated hyperparameter tuning (Optuna, Hyperopt)
- Neural architecture search
- Automated feature engineering (Featuretools)

#### 11.2.3 Model Deployment

**Production System**:
- RESTful API for predictions
- Web application for user interaction
- Dashboard for monitoring model performance

**Continuous Learning**:
- Online learning with new data
- A/B testing of model versions
- Automated retraining pipelines

**Monitoring**:
- Track prediction accuracy over time
- Detect data drift
- Alert for anomalous predictions

#### 11.2.4 Extended Applications

**1. Cost Category Classification**:
- Classify into cost bands (low/medium/high)
- Multi-class classification problem
- May be useful for tiered insurance products

**2. Claim Prediction**:
- Predict likelihood of filing claim
- Identify factors for preventive outreach
- Binary classification problem

**3. Personalized Recommendations**:
- Suggest interventions to reduce costs
- Simulate impact of behavior changes
- Prescriptive analytics

**4. Fraud Detection**:
- Identify anomalous cost patterns
- Detect potentially fraudulent claims
- Outlier detection techniques

**5. Network Analysis**:
- If provider data available
- Analyze cost efficiency of provider networks
- Recommend optimal care pathways

#### 11.2.5 Research Directions

1. **Fairness-Aware ML**:
   - Techniques to ensure equitable predictions
   - Debiasing algorithms
   - Fair representation learning

2. **Explainable AI**:
   - Develop better interpretability tools
   - Local and global explanations
   - Counterfactual explanations

3. **Federated Learning**:
   - Train on distributed data (multiple insurers)
   - Preserve privacy
   - Build more robust models

4. **Transfer Learning**:
   - Leverage models from similar domains
   - Adapt to new populations or regions
   - Few-shot learning for rare conditions

---

## 12. Conclusion

### 12.1 Summary of Achievements

This project successfully developed and evaluated multiple machine learning models to predict medical insurance costs based on individual health and demographic factors. Our key accomplishments include:

1. **Comprehensive Data Analysis**: Conducted thorough EDA revealing critical insights about cost drivers, particularly the dominant effect of smoking status.

2. **Effective Feature Engineering**: Created domain-informed features (BMI categories, interaction terms, high-risk flags) that improved model performance by [X]%.

3. **Extensive Model Comparison**: Implemented and evaluated 10+ different models, from linear regression to advanced ensembles.

4. **Strong Predictive Performance**: Achieved a test R² of [X] with our best model ([Model Name]), explaining [X]% of variance in insurance costs with an average error of $[MAE].

5. **Interpretable Results**: Leveraged SHAP analysis and feature importance to understand model decisions, ensuring transparency and trustworthiness.

6. **Healthcare Context**: Grounded findings in medical and insurance domain knowledge, making results actionable for stakeholders.

### 12.2 Key Takeaways

**For Data Scientists**:
- Feature engineering is as important as model selection
- Ensemble methods often provide best performance
- Interpretability is crucial in high-stakes domains like healthcare
- Cross-validation and careful evaluation prevent overfitting

**For Healthcare Professionals**:
- Data-driven approaches can augment traditional actuarial methods
- Smoking cessation and obesity management have quantifiable ROI
- Interactions between risk factors matter (not just individual effects)

**For Students**:
- End-to-end ML project requires diverse skills: coding, statistics, domain knowledge, communication
- Documentation and reproducibility are as important as accuracy
- Ethical considerations must be integrated throughout, not added as afterthought

### 12.3 Final Thoughts

Predicting medical insurance costs is both a technical challenge and an opportunity to apply ML for societal benefit. While our model demonstrates strong performance, responsible deployment requires ongoing attention to:

- **Fairness**: Ensuring predictions don't perpetuate health inequities
- **Privacy**: Protecting individual health information
- **Transparency**: Explaining model decisions to affected individuals
- **Accuracy**: Continuously updating as healthcare and populations evolve

This project represents a foundational step toward data-driven healthcare cost management. The insights gained—especially the critical importance of modifiable risk factors like smoking—can inform both business strategy and public health policy.

**Machine learning is a powerful tool, but it must be wielded responsibly, with domain expertise, ethical awareness, and commitment to benefiting people and society.**

---

## 13. References

### Academic Papers

1. [Author et al. (Year)]. "Title of Paper on Healthcare Cost Prediction." *Journal Name*, Volume(Issue), pages. DOI/URL

2. [Author et al. (Year)]. "Machine Learning Applications in Healthcare." *Journal Name*, Volume(Issue), pages.

3. [Author et al. (Year)]. "Factors Affecting Medical Insurance Costs." *Journal Name*, Volume(Issue), pages.

### Books

4. Lantz, B. (2019). *Machine Learning with R* (3rd ed.). Packt Publishing. [Original source of dataset]

5. [Author]. (Year). *Book on Healthcare Analytics*.

### Online Resources

6. Kaggle. (2018). Medical Cost Personal Datasets. Retrieved from https://www.kaggle.com/datasets/mirichoi0218/insurance

7. Scikit-learn Documentation. (2023). Retrieved from https://scikit-learn.org/stable/

8. SHAP Documentation. (2023). Retrieved from https://shap.readthedocs.io/

### Government/Institutional Reports

9. CDC. (Year). "Smoking & Tobacco Use." Retrieved from https://www.cdc.gov/tobacco/

10. WHO. (Year). "BMI Classification." Retrieved from https://www.who.int/

---

## 14. Appendices

### Appendix A: Additional Visualizations

[Include supplementary plots not shown in main text]

### Appendix B: Hyperparameter Tuning Results

**Random Forest GridSearch Results** (Full):
[Table with all combinations tested]

**Gradient Boosting GridSearch Results** (Full):
[Table with all combinations tested]

### Appendix C: Full Model Comparison Table

[Extended table with all models and all metrics]

### Appendix D: Code Repository

**GitHub Repository**: [URL]

**Repository Structure**:
```
/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

**Key Files**:
- `01_EDA.ipynb`: Exploratory data analysis
- `02_Feature_Engineering.ipynb`: Feature creation and selection
- `03_Model_Training.ipynb`: Training all models
- `04_Model_Evaluation.ipynb`: Comprehensive evaluation
- `src/preprocessing.py`: Reusable preprocessing functions
- `src/models.py`: Model classes and utilities

### Appendix E: Instructions for Reproducibility

**Environment Setup**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Running the Analysis**:
```bash
# Run notebooks in order
jupyter notebook
# Execute: 01_EDA.ipynb → 02_Feature_Engineering.ipynb → ...
```

**Data**:
- Download from Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance
- Place in `data/raw/insurance.csv`

### Appendix F: Glossary

**BMI (Body Mass Index)**: Weight in kilograms divided by height in meters squared. Standard metric for assessing body weight relative to height.

**Cross-Validation**: Technique for assessing model generalization by training on multiple data subsets.

**Ensemble Method**: Combining multiple models to improve prediction accuracy.

**Feature Engineering**: Creating new features from existing data to improve model performance.

**L1 Regularization (Lasso)**: Penalty that can shrink coefficients to zero, performing feature selection.

**L2 Regularization (Ridge)**: Penalty that shrinks coefficients but doesn't eliminate them.

**MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals.

**R² (Coefficient of Determination)**: Proportion of variance in target variable explained by the model.

**RMSE (Root Mean Squared Error)**: Square root of average squared prediction errors.

**SHAP (SHapley Additive exPlanations)**: Method for explaining individual predictions based on game theory.

### Appendix G: Team Contributions

[If working in a team, acknowledge individual contributions]

**Team Member 1** (Your Name):
- Led EDA and visualization
- Implemented linear models
- Co-wrote Introduction and Methodology sections

**Team Member 2**:
- Feature engineering
- Implemented tree-based models
- Co-wrote Results section

[Continue for all 6 team members]

### Appendix H: Acknowledgments

We would like to thank:
- **CAIR Collective** at UCLA for providing the opportunity to work on this project
- **Curriculum Team Leads** for guidance and mentorship
- **Brett Lantz** for making the dataset publicly available
- **Kaggle Community** for data hosting and sharing
- **Open Source Community** for the excellent libraries used in this project

---

**END OF REPORT**

*Document prepared by: CAIR Collective Curriculum Team*  
*Affiliation: University of California, Los Angeles*  
*Date: February 9th, 2026*  
*Version: 1.0*
