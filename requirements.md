# DETECT-PD Requirements (EARS Format)

## Data Ingestion Requirements

**WHEN** the system needs to load patient data from Excel CRF  
**THE SYSTEM SHALL** parse demographics, comorbidities, treatment timeline, and biochemical results into a clean DataFrame

**WHEN** the system encounters missing critical fields in patient data  
**THE SYSTEM SHALL** drop rows with missing outcome or key features to ensure complete case analysis

**WHEN** the system validates input data  
**THE SYSTEM SHALL** check date field chronology and flag numerical values outside plausible ranges

## Data Processing Requirements

**WHEN** the system processes patient comorbidity data  
**THE SYSTEM SHALL** compute Charlson Comorbidity Index (CCI) score based on condition indicators with appropriate point weights (1-6 points per condition)

**WHEN** calculating time-based features  
**THE SYSTEM SHALL** derive failure period (eGFR<10 to PD start), waiting period (catheter insertion to PD start), and PD period (PD start to assessment) in days

**WHEN** calculating BMI and body surface area  
**THE SYSTEM SHALL** calculate the BMI in body weight / body height ^2, calculate body surface area with Du Bois formula: BSA = 0.007184 × W0.425 × H0.725

**WHEN** preprocessing continuous variables with high skewness  
**THE SYSTEM SHALL** apply log transformation or normalization to achieve more Gaussian distribution

**WHEN** scaling numeric features  
**THE SYSTEM SHALL** use StandardScaler for normal distributions or MinMaxScaler for extreme ranges/skew

**WHEN** encoding categorical variables  
**THE SYSTEM SHALL** apply one-hot encoding or label encoding as appropriate for PD modality and ESRD cause categories

## Feature Engineering Requirements

**WHEN** performing feature selection  
**THE SYSTEM SHALL** apply LASSO-regularized regression with L1 penalty to identify features with non-zero coefficients

**WHEN** selecting features for binary adequacy classification  
**THE SYSTEM SHALL** define binary targets (e.g., Kt/V < 1.7 vs ≥1.7) and fit LASSO logistic regression

**WHEN** feature selection is complete  
**THE SYSTEM SHALL** extract and log the list of selected features with their coefficients for transparency

## Model Training Requirements

**WHEN** training regression models  
**THE SYSTEM SHALL** implement at least XGBoost and Random Forest algorithms for both Kt/V and PET prediction

**WHEN** training separate models for each target  
**THE SYSTEM SHALL** create dedicated regressors for Kt/V and PET outcomes using preprocessed training data

**WHEN** performing hyperparameter tuning  
**THE SYSTEM SHALL** use cross-validation or hold-out validation from training set with configurable parameters

**WHEN** preventing overfitting  
**THE SYSTEM SHALL** implement early stopping or cross-validated evaluation during model training

## Model Evaluation Requirements

**WHEN** evaluating model performance  
**THE SYSTEM SHALL** compute Mean Absolute Error (MAE), Mean Squared Error (MSE), R² coefficient, and Intraclass Correlation Coefficient (ICC)

**WHEN** assessing prediction calibration  
**THE SYSTEM SHALL** generate calibration curves comparing predicted vs actual values with reference line y=x

**WHEN** comparing multiple trained models  
**THE SYSTEM SHALL** evaluate all models on the same held-out test set and identify best performing model for each metric

**WHEN** evaluating Kt/V predictions against adequacy threshold  
**THE SYSTEM SHALL** assess proportion of patients predicted adequate (≥1.7) vs actually adequate for calibration check

## Pipeline Orchestration Requirements

**WHEN** executing the ML pipeline  
**THE SYSTEM SHALL** implement modular ZenML steps for ingestion, splitting, preprocessing, feature engineering, training, and evaluation

**WHEN** splitting data for training and testing  
**THE SYSTEM SHALL** randomly split cleaned dataset into 80% training and 20% testing with fixed random seed for reproducibility

**WHEN** the test set is created  
**THE SYSTEM SHALL** keep test data completely untouched during training and feature selection, using only for final evaluation

**WHEN** saving preprocessing artifacts  
**THE SYSTEM SHALL** serialize and version fitted transformers (scalers, encoders) for consistent application to new data

## Experiment Tracking Requirements

**WHEN** training models  
**THE SYSTEM SHALL** log all hyperparameters, training metrics, and model performance to MLflow for experiment tracking

**WHEN** preprocessing data  
**THE SYSTEM SHALL** log chosen scaling methods, transformation details, and feature statistics to MLflow for traceability

**WHEN** evaluating models  
**THE SYSTEM SHALL** log final test set metrics (MAE, MSE, R², ICC) and save calibration plots as MLflow artifacts

**WHEN** completing feature selection  
**THE SYSTEM SHALL** log selected feature names and their LASSO coefficients as MLflow parameters

## Configuration Management Requirements

**WHEN** configuring pipeline steps  
**THE SYSTEM SHALL** accept parameters via YAML/JSON configuration for data paths, split ratios, scaling methods, and model hyperparameters

**WHEN** running different experiments  
**THE SYSTEM SHALL** allow modification of configuration without code changes to promote experimentation flexibility

## Model Persistence Requirements

**WHEN** training is complete  
**THE SYSTEM SHALL** save trained model objects in serialized format (.joblib or native format) with clear naming conventions

**WHEN** models are saved  
**THE SYSTEM SHALL** register model artifacts in MLflow model registry with version control for deployment readiness

## Validation and Quality Requirements

**WHEN** validating model predictions  
**THE SYSTEM SHALL** ensure predictions are clinically plausible (e.g., Kt/V values in reasonable range, PET ratios between 0-1)

**WHEN** assessing feature importance  
**THE SYSTEM SHALL** generate and log feature importance rankings to verify clinically sensible decision patterns

**WHEN** the pipeline execution is complete  
**THE SYSTEM SHALL** provide comprehensive evaluation report with all metrics, plots, and selected features for clinical review