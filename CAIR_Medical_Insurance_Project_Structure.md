# CAIR Collective: Medical Insurance Cost Prediction Project
## Healthcare AI/ML Portfolio Project

---

## Project Overview

**Goal**
Build an end-to-end machine learning pipeline to predict medical insurance costs using patient demographics and health indicators. The project emphasizes **clean ML workflow, interpretability, and healthcare context**, and culminates in a **portfolio-ready analysis**.

**Dataset**
Medical Cost Personal Dataset (Kaggle)

* **Features**: age, sex, BMI, children, smoker, region
* **Target**: charges (medical insurance costs)
* **Size**: ~1,338 observations

**Team Structure**
6 students working collaboratively, with shared code and rotating analytical ownership.

**Primary Artifacts**

* **Student Template Notebook** (guided, fill-in-the-blank)
* **Instructor Master Notebook** (fully implemented, used for live teaching)
* Final report + presentation

---

## Learning Objectives

By the end of this project, students will be able to:

1. Perform structured EDA on healthcare cost data
2. Build and evaluate multiple regression-based ML models
3. Apply feature engineering, regularization, and ensemble methods
4. Evaluate models using interpretable and healthcare-relevant metrics
5. Explain model behavior using feature importance and interpretability tools
6. Communicate technical findings clearly in a professional report
7. Demonstrate ethical awareness in healthcare AI

---

## Project Structure & Workflow

### Notebook Structure (Aligned With Implementation)

Instead of multiple fragmented notebooks, the project follows **one continuous workflow**:

* **Student Template Notebook**

  * Contains explanations, section headers, and `TODO` blocks
  * Students complete code progressively during instruction
  * Serves as their final submission artifact

* **Instructor Master Notebook**

  * Same structure as the student template
  * Fully implemented code
  * Used to teach live, demonstrate decisions, and debug together

This ensures:

* Everyone stays in sync
* Students understand *why* decisions are made
* No confusion about which notebook is authoritative

---

## Project Phases & Timeline

---

## Phase 1: Data Exploration & Understanding (Week 1)

**Participants**: All students (pair programming encouraged)

### Objectives

Understand the dataset, variables, and healthcare context before modeling.

### Tasks (Student Template Sections)

1. **Data Loading & Inspection**

   * Load dataset
   * Inspect shape, data types, missing values
   * Summary statistics

2. **Exploratory Data Analysis (EDA)**

   * Univariate analysis of features
   * Target variable distribution (charges)
   * Correlation analysis
   * Identify skewness and outliers

3. **Required Visualizations**

   * Histograms & box plots (age, BMI, charges)
   * Count plots (sex, smoker, region)
   * Scatter plots: charges vs age, BMI
   * Box plots: charges by smoker, sex, region
   * Correlation heatmap

4. **Healthcare Context & Ethics**

   * Why each feature may influence cost
   * Smoking and BMI as dominant drivers
   * Ethical considerations of demographic features

### Deliverable

* Completed EDA section in the student notebook
* Written markdown insights embedded directly in the notebook

---

## Phase 2: Preprocessing & Feature Engineering (Week 2)

**Participants**: Split into two working groups (conceptually)

### Objectives

Prepare data correctly and introduce domain-informed features.

### Tasks (Template-Guided)

**Data Preprocessing**

* Train-test split (80/20)
* Categorical encoding:

  * Binary encoding: sex, smoker
  * One-hot encoding: region
* Feature scaling (StandardScaler)
* Discuss alternative scalers conceptually

**Feature Engineering**

* Interaction terms:

  * smoker × BMI
  * age × smoker
* Polynomial features:

  * age², BMI²
* Domain features:

  * BMI categories
  * High-risk indicator (smoker + high BMI)
  * Family size indicator

### Deliverable

* Clean feature matrix used consistently for all models
* Preprocessing steps clearly documented in markdown

---

## Phase 3: Model Development & Training (Weeks 3-4)

**Participants**: All students rotate ownership of model families

### Models Implemented (Core Set)

**Baseline**

* Mean baseline
* Simple linear regression

**Linear Models**

* Multiple Linear Regression
* Ridge
* Lasso
* ElasticNet

**Tree-Based Models**

* Decision Tree
* Random Forest
* Gradient Boosting

**Additional Models (Optional / Stretch)**

* KNN Regressor
* SVR
* Neural Network (MLPRegressor)

### Best Practices Emphasized

* Cross-validation
* Hyperparameter tuning (GridSearchCV)
* Consistent evaluation pipeline
* Avoiding data leakage

### Deliverable

* Trained models evaluated within the same notebook
* Clear comparison tables

---

## Phase 4: Model Evaluation & Comparison (Week 5)

**Participants**: All students collaborate

### Evaluation Metrics

* MAE (interpretable in dollars)
* RMSE
* R²
* Cross-validation mean & variance

### Analysis Performed

* Model comparison table
* Actual vs predicted plots
* Residual analysis
* Error distribution by subgroup
* Performance on high-cost patients

### Feature Importance

* Coefficients (linear models)
* Feature importance (tree models)
* Permutation importance (core)
* SHAP (introduced conceptually or implemented if time permits)

### Deliverable

* Evaluation section completed in notebook
* Clear justification for best-performing model

---

## Phase 5: Advanced Analysis (Week 6 - Optional / Extension)

Students may explore **one or more** of the following:

* SHAP interpretability
* Prediction intervals / uncertainty
* Fairness checks across sex & region
* Feature selection (Lasso, RFE)
* PCA comparison
* Ensemble models (Voting / Stacking)

These are framed as **portfolio extensions**, not core requirements.

---

## Phase 6: Final Report & Presentation (Week 7)

### Written Report

Sections:

1. Executive Summary
2. Introduction & Healthcare Context
3. Data & Methodology
4. Exploratory Analysis
5. Model Results & Evaluation
6. Interpretability & Ethics
7. Limitations & Future Work
8. Conclusion

### Presentation Slides

* Problem framing
* Modeling approach
* Key findings
* Healthcare implications
* Ethical considerations

---

## Success Metrics

### Technical

* R² ≥ 0.75
* RMSE ≤ $6,000
* At least 4 distinct model families
* CV and test performance aligned

### Learning

* Each student contributes code + analysis
* Clear markdown explanations
* Reproducible notebook

### Portfolio

* Clean GitHub repo
* Polished visuals
* Clear narrative connecting ML → healthcare impact

---

## Key Takeaways for Students

Students should be able to explain:

* Why preprocessing choices mattered
* Why smoker & BMI dominate predictions
* Trade-offs between interpretability and performance
* Ethical concerns in healthcare ML
* How to present ML results to non-technical stakeholders

---

## Healthcare-Specific Considerations

### Ethical Considerations:
1. **Fairness**: Ensure models don't discriminate based on sex or region
2. **Privacy**: Discuss HIPAA considerations even with anonymized data
3. **Transparency**: Model decisions should be explainable to stakeholders
4. **Impact**: Consider how predictions might affect insurance accessibility

### Domain Knowledge Integration:
1. **Smoking Impact**: Known to be the strongest cost predictor
2. **BMI Thresholds**: Use clinical definitions (18.5, 25, 30)
3. **Age Effects**: Non-linear relationship with costs
4. **Regional Variations**: Healthcare cost differences across US regions
5. **Family Coverage**: Children count affects insurance structure

### Real-World Deployment Considerations:
1. **Model Updating**: How often should models be retrained?
2. **Regulatory Compliance**: Insurance regulations and model use
3. **Stakeholder Communication**: How to present findings to non-technical audiences
4. **Cost-Benefit Analysis**: Model accuracy vs implementation cost

---

## Resources for Students

### Technical Resources:
- Scikit-learn documentation: https://scikit-learn.org/
- Kaggle Learn (Machine Learning): https://www.kaggle.com/learn
- SHAP documentation: https://shap.readthedocs.io/
- Seaborn gallery: https://seaborn.pydata.org/examples/index.html

### Healthcare ML Resources:
- "Machine Learning in Healthcare" articles
- Healthcare cost analysis papers
- Insurance industry reports
- Healthcare.gov for policy context

### Writing Resources:
- Academic paper templates
- Data science report examples
- Visualization best practices
- Technical writing guides

---

## Bonus Challenges (Optional)

1. **Deploy a simple web app** using Streamlit or Gradio for predictions
2. **Create an interactive dashboard** with Plotly Dash or Tableau
3. **Implement AutoML** comparison with TPOT or Auto-sklearn
4. **Time-series analysis** if treating data as pseudo-longitudinal
5. **Causal inference** techniques to understand true causal relationships
6. **Deep learning** approach with TensorFlow/Keras
7. **Cost category classification** in addition to regression
8. **Synthetic data generation** to augment dataset

---

## Key Takeaways for Portfolio

Students should be able to articulate:
1. How they approached a real-world healthcare ML problem
2. Their specific contributions to the team
3. Technical decisions and trade-offs they made
4. How they evaluated and compared multiple models
5. The healthcare domain insights they discovered
6. Ethical considerations in healthcare AI
7. How they communicated complex findings to different audiences

This project will demonstrate proficiency in:
- End-to-end ML pipeline development
- Healthcare domain application
- Team collaboration on technical projects
- Professional technical communication
- Ethical AI considerations

---

**Good luck to the CAIR Collective team! This project will be an excellent portfolio piece demonstrating ML skills in a meaningful healthcare context.**
