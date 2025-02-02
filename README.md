# Travel Insurance Prediction using Causal Inference and Machine Learning

## Project Overview
This project analyzes the likelihood of individuals purchasing travel insurance using predictive modeling and causal inference techniques. By applying machine learning classifiers and causal inference methods, the study aims to determine key factors influencing travel insurance uptake, particularly the impact of frequent flyer status.

## Features of the Project
- **Predictive Modeling**: Utilizes Random Forest, Logistic Regression, and XGBoost for travel insurance prediction.
- **Feature Engineering**: Includes income-based, age-based, and risk-based variables to enhance model performance.
- **Hyperparameter Tuning**: Implements grid search and cross-validation for optimal model selection.
- **Causal Inference**: Applies S-Learner, T-Learner, X-Learner, and R-Learner methodologies to estimate treatment effects.
- **Model Evaluation**: Assesses model performance using ROC-AUC, accuracy, and confusion matrix.

## Dataset Description
- **Source**: Kaggle dataset containing customer demographics, travel behavior, and financial attributes.
- **Key Features**:
  - **Binary Features**: FrequentFlyer, GraduateOrNot, EverTravelledAbroad.
  - **Numerical Features**: Age, Annual Income, Family Members, Chronic Diseases.
  - **Engineered Features**: Log-transformed income, risk score, and travel risk.
  - **Target Variable**: TravelInsurance (0 = No, 1 = Yes).

## Methodology
1. **Data Preprocessing**:
   - Imputation of missing values.
   - Feature scaling and encoding of categorical variables.
   - Transformation of skewed distributions using power transformation.
2. **Model Development**:
   - Train-Test Split with Stratified Sampling.
   - Model Training using Logistic Regression, Random Forest, and XGBoost.
   - Hyperparameter tuning using Grid Search.
3. **Causal Inference**:
   - Estimation of treatment effects using S-Learner, T-Learner, X-Learner, and R-Learner.
   - DragonNet model for deep-learning-based causal inference.
4. **Evaluation**:
   - ROC-AUC, Log Loss, and Accuracy metrics.
   - Uplift curves to assess segment-based impact.

## Key Results
- **Frequent Flyer Impact**: Positive treatment effect on insurance uptake, but varies across income and age groups.
- **Feature Importance**: Annual income, age, and chronic diseases have significant influence.
- **Best Model**: CatBoost and LGBM models demonstrated the strongest predictive power.
- **Segment-Specific Insights**: Older, high-income individuals are more likely to purchase travel insurance.

## Future Improvements
- **Refining Causal Analysis**: Introduce additional confounders to improve treatment effect accuracy.
- **Expanded Feature Engineering**: Incorporate text-based insights from customer profiles.
- **Real-World Testing**: Apply results to targeted marketing strategies for insurance providers.

## Getting Started
### Dependencies
- **Programming Language**: Python 3.8+
- **Libraries**: pandas, numpy, seaborn, matplotlib, sklearn, xgboost, catboost, lightgbm, causalml

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/travel-insurance-causal-analysis.git
   ```
2. Run the preprocessing and feature engineering scripts.
3. Train the models and evaluate causal effects.

## Contact
For questions or collaboration opportunities, please reach out to:

**Name**: Arturo Medina  
**LinkedIn**: [linkedin.com/in/arturo-medina](https://linkedin.com/in/arturo-medina)
