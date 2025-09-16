# DETECT-PD System Design Document

## Architecture Overview

The DETECT-PD ML pipeline follows a modular, microservice-oriented architecture built on ZenML orchestration framework with MLflow for experiment tracking. The system is designed for reproducibility, scalability, and clinical deployment readiness.

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│   ZenML Pipeline │───▶│  MLflow Tracker │
│   (Excel CRF)   │    │   Orchestrator   │    │  & Model Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Components                          │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│ Data        │ Data        │ Feature     │ Model       │ Model  │
│ Ingestion   │ Processing  │ Engineering │ Training    │ Eval   │
└─────────────┴─────────────┴─────────────┴─────────────┴────────┘
```

### Core Design Principles

1. **Modularity**: Each pipeline step is independent and testable
2. **Reproducibility**: Fixed seeds, versioned artifacts, comprehensive logging
3. **Configurability**: YAML-driven configuration for all parameters
4. **Clinical Safety**: Data validation and plausible prediction ranges
5. **MLOps Integration**: Complete experiment tracking and model versioning

## Components and Interface Definitions

### 1. Data Ingestion Component

**Purpose**: Load and validate Excel CRF data
**Interface**:
```python
@step
def data_ingestion_step(config: DataIngestionConfig) -> pd.DataFrame:
    """
    Input: DataIngestionConfig (file_path, sheet_name, validation_rules)
    Output: Clean DataFrame with validated patient data
    Raises: DataValidationError, FileNotFoundError
    """
```

**Responsibilities**:
- Parse Excel files using pandas/openpyxl
- Validate date chronology and numerical ranges
- Drop rows with critical missing fields
- Log data quality metrics

### 2. Data Splitting Component

**Purpose**: Split data into training/testing sets
**Interface**:
```python
@step  
def data_splitting_step(
    data: pd.DataFrame, 
    config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input: Clean DataFrame, SplitConfig (test_ratio, random_seed)
    Output: (train_data, test_data) tuple
    """
```

### 3. Data Processing Component

**Purpose**: Feature derivation and preprocessing
**Interface**:
```python
@step
def data_processing_step(
    train_data: pd.DataFrame,
    config: PreprocessingConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Input: Training data, preprocessing configuration
    Output: (processed_data, preprocessing_artifacts)
    """
```

**Key Operations**:
- Charlson Comorbidity Index calculation
- Time-based feature derivation (failure/waiting/PD periods)
- BMI and BSA calculation using Du Bois formula
- Scaling and encoding transformations

### 4. Feature Engineering Component

**Purpose**: LASSO-based feature selection
**Interface**:
```python
@step
def feature_engineering_step(
    processed_data: pd.DataFrame,
    config: FeatureConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Input: Processed training data, feature selection config
    Output: (selected_features_data, selected_feature_names)
    """
```

### 5. Model Training Component

**Purpose**: Train multiple ML models for Kt/V and PET prediction
**Interface**:
```python
@step
def model_training_step(
    train_data: pd.DataFrame,
    config: ModelConfig
) -> Dict[str, Any]:
    """
    Input: Feature-selected training data, model configuration
    Output: Dictionary of trained models and validation scores
    """
```

**Supported Models**:
- XGBoost Regressor
- Random Forest Regressor
- Configurable hyperparameters via config

### 6. Model Evaluation Component

**Purpose**: Comprehensive model performance assessment
**Interface**:
```python
@step
def model_evaluation_step(
    models: Dict[str, Any],
    test_data: pd.DataFrame,
    preprocessing_artifacts: Dict[str, Any]
) -> EvaluationResults:
    """
    Input: Trained models, test data, preprocessing artifacts
    Output: Complete evaluation results with metrics and plots
    """
```

## Data Models and Error Handling Design

### Core Data Models

#### 1. Patient Data Model
```python
@dataclass
class PatientData:
    patient_id: str
    demographics: Demographics
    comorbidities: List[Comorbidity]
    treatment_timeline: TreatmentTimeline
    biochemical_results: BiochemicalResults
    outcomes: Optional[Outcomes]
    
    def validate(self) -> bool:
        """Validate data completeness and plausibility"""
```

#### 2. Configuration Models
```python
@dataclass
class DataIngestionConfig:
    file_path: str
    sheet_name: str = "Sheet1"
    required_columns: List[str]
    date_columns: List[str]
    numeric_validation_rules: Dict[str, Tuple[float, float]]

@dataclass
class ModelConfig:
    model_type: str  # "xgboost" or "random_forest"
    hyperparameters: Dict[str, Any]
    cross_validation_folds: int = 5
    early_stopping_rounds: Optional[int] = None
```

#### 3. Results Models
```python
@dataclass
class EvaluationResults:
    metrics: Dict[str, float]  # MAE, MSE, R2, ICC
    calibration_plots: Dict[str, str]  # Plot file paths
    feature_importance: Dict[str, float]
    model_comparison: pd.DataFrame
```

### Error Handling Strategy

#### 1. Data Quality Errors
```python
class DataValidationError(Exception):
    """Raised when input data fails validation checks"""
    
class MissingCriticalFieldError(DataValidationError):
    """Raised when required fields are missing"""
    
class DateChronologyError(DataValidationError):
    """Raised when dates are not in logical order"""
```

#### 2. Model Training Errors
```python
class ModelTrainingError(Exception):
    """Base class for model training failures"""
    
class ConvergenceError(ModelTrainingError):
    """Raised when model fails to converge"""
    
class FeatureSelectionError(ModelTrainingError):
    """Raised when LASSO selection fails"""
```

#### 3. Error Recovery Mechanisms

- **Data Level**: Automatic imputation for non-critical missing values
- **Model Level**: Fallback to simpler models if complex models fail
- **Pipeline Level**: Checkpoint saving for restart capability
- **Logging**: Comprehensive error logging to MLflow for debugging

### Clinical Safety Validations

```python
def validate_predictions(predictions: np.ndarray, target: str) -> bool:
    """
    Validate predictions are clinically plausible
    Kt/V: Should be between 0.5 and 4.0
    PET: Should be between 0.1 and 1.0
    """
    if target == "ktv":
        return np.all((predictions >= 0.5) & (predictions <= 4.0))
    elif target == "pet":
        return np.all((predictions >= 0.1) & (predictions <= 1.0))
    return False
```

## Testing Strategy and Decision Rationale

### 1. Unit Testing Strategy

#### Component-Level Tests
- **Data Processing Tests**: Validate CCI calculations, time feature derivations, BMI/BSA formulas
- **Feature Engineering Tests**: Test LASSO selection with synthetic data
- **Model Training Tests**: Verify model instantiation and basic fitting
- **Validation Tests**: Test clinical range validations

#### Test Coverage Targets
- Minimum 90% code coverage for core business logic
- 100% coverage for clinical calculation functions
- Edge case testing for all data validation rules

### 2. Integration Testing Strategy

#### Pipeline Integration Tests
```python
def test_full_pipeline_integration():
    """Test complete pipeline execution with sample data"""
    # Use synthetic data matching CRF schema
    # Verify all steps execute successfully
    # Validate output artifacts are created
    # Check MLflow logging completeness
```

#### Data Flow Tests
- Test data consistency between pipeline steps
- Verify preprocessing artifacts are correctly applied
- Validate model predictions on known test cases

### 3. Performance Testing Strategy

#### Model Performance Benchmarks
- Minimum R² threshold: 0.6 for both Kt/V and PET models
- Maximum acceptable MAE: 0.3 for Kt/V, 0.15 for PET
- Training time limits: <30 minutes for full pipeline
- Memory usage limits: <8GB RAM

#### Scalability Testing
- Test pipeline with datasets up to 10,000 patients
- Verify memory efficiency with large feature sets
- Performance regression testing

### 4. Clinical Validation Testing

#### Medical Domain Tests
```python
def test_clinical_plausibility():
    """Validate model behavior aligns with medical knowledge"""
    # High CCI should correlate with lower Kt/V adequacy
    # Longer PD periods should improve outcomes
    # BMI should influence dosing predictions
```

#### Bias and Fairness Testing
- Test model performance across demographic groups
- Validate predictions are not biased by gender/age
- Ensure consistent performance across comorbidity levels

### 5. Decision Rationale

#### Technology Stack Decisions

**ZenML Choice**: 
- **Rationale**: Provides MLOps orchestration with built-in versioning and caching
- **Alternative Considered**: Kubeflow Pipelines (rejected due to complexity)
- **Risk Mitigation**: ZenML's active community and documentation support

**MLflow Integration**:
- **Rationale**: Industry standard for ML experiment tracking and model registry
- **Alternative Considered**: Weights & Biases (rejected due to cost)
- **Risk Mitigation**: Open source with enterprise deployment options

**Model Algorithm Selection**:
- **Primary**: XGBoost and Random Forest
- **Rationale**: Proven performance on tabular medical data, interpretable feature importance
- **Alternative Considered**: Neural networks (rejected due to interpretability requirements)

#### Architecture Decisions

**Modular Pipeline Design**:
- **Rationale**: Enables independent testing, debugging, and component replacement
- **Trade-off**: Slight performance overhead vs. maintainability benefits
- **Risk Mitigation**: Comprehensive integration testing

**Configuration-Driven Approach**:
- **Rationale**: Supports clinical research workflows requiring parameter experimentation
- **Trade-off**: Additional complexity vs. flexibility gains
- **Risk Mitigation**: Strong validation of configuration schemas

#### Clinical Safety Decisions

**Conservative Prediction Ranges**:
- **Rationale**: Medical applications require high confidence in prediction validity
- **Implementation**: Hard validation limits with warning systems
- **Risk Mitigation**: Clinical review of all flagged predictions

**Complete Case Analysis**:
- **Rationale**: Ensures high data quality for medical predictions
- **Trade-off**: Reduced sample size vs. prediction reliability
- **Risk Mitigation**: Comprehensive missing data analysis and reporting