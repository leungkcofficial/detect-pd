# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DETECT-PD (Dialysis Efficiency and Transporter Evaluation Computational Tool in Peritoneal Dialysis) is a machine learning project for predicting dialysis adequacy (Kt/V) and peritoneal membrane transport status (PET) in peritoneal dialysis patients. The project consists of:

1. **ML Pipeline** (`detect_pd/`): ZenML-based pipeline for training ML models using patient data
2. **Frontend** (`frontend/`): Static web application for clinician-facing predictions

## Development Commands

### Python ML Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run the ML training pipeline
cd detect_pd
python -m detect_pd.pipelines.training_pipeline --config configs/pipeline.yaml

# Run tests
cd detect_pd
python -m pytest src/tests/ -v

# Code quality checks
black src/
flake8 src/
mypy src/
```

### Frontend

```bash
# Serve the static application
cd frontend
python -m http.server 8080
# Visit http://localhost:8080/index.html

# Run unit tests
cd frontend
npm run test:unit
# or manually:
node tests/deriveFeatures.test.mjs
node tests/validation.test.mjs
```

## Architecture Overview

### ML Pipeline Architecture

The project uses ZenML for ML pipeline orchestration and MLflow for experiment tracking:

- **Pipeline Entry**: `detect_pd/src/detect_pd/pipelines/training_pipeline.py` - Main pipeline orchestrating all ML steps
- **Configuration**: Centralized YAML config at `detect_pd/configs/pipeline.yaml` with Pydantic models in `detect_pd/src/detect_pd/config/`
- **Steps**: Individual pipeline components in `detect_pd/src/detect_pd/steps/`:
  - `data_ingestion.py` - Load Excel data from CRF
  - `preprocessing.py` - Feature engineering (CCI, BMI, time intervals)
  - `feature_selection.py` - LASSO-based feature selection
  - `model_training.py` - Train multiple ML models (XGBoost, CatBoost, etc.)
  - `evaluation.py` - Model evaluation and calibration analysis
- **Utils**: Clinical calculations in `detect_pd/src/detect_pd/utils/` (CCI, PET classifications)

### Frontend Architecture

The frontend is a vanilla JavaScript SPA with modular utilities:

- **Main App**: `frontend/app.js` - Core application state and UI logic
- **Feature Engineering**: `frontend/utils/derived.js` - Clinical calculations (BMI, CCI, osmolarity)
- **Validation**: `frontend/utils/validation.js` - Input validation and guard bands
- **Inference**: `frontend/utils/inference.js` - ML model prediction calls (mock or API)
- **Configuration**: `frontend/config/frontend_config.json` - All app settings

### Data Flow

1. **Training Data**: Excel CRF (`detect_pd/data/08_a_crf.xlsx`) → Data ingestion → Preprocessing → Feature selection → Model training
2. **Prediction Flow**: Frontend form inputs → Feature engineering → Model inference → Results display

## Key Configuration Files

- `detect_pd/configs/pipeline.yaml` - Complete ML pipeline configuration including:
  - Data ingestion column mappings and Excel sheet structure
  - Feature engineering parameters (CCI calculation, time intervals)
  - Model hyperparameters for multiple algorithms
  - Evaluation metrics and output paths
- `frontend/config/frontend_config.json` - Frontend application settings:
  - Dialysate bag formulations and osmolarity values
  - Clinical guard bands and validation rules
  - API endpoints and inference configuration

## Testing Strategy

### ML Pipeline Tests
- Located in `detect_pd/src/tests/`
- Test individual pipeline steps and utility functions
- Use pytest framework with fixtures for sample data
- Run with: `python -m pytest src/tests/ -v`

### Frontend Tests
- Located in `frontend/tests/`
- Unit tests for feature engineering and validation logic
- ESM modules tested with Node.js
- Run with: `npm run test:unit` or individual node commands

## Clinical Domain Context

This is a medical AI project focusing on peritoneal dialysis adequacy:

- **Kt/V**: Measure of dialysis adequacy (≥1.7 considered adequate)
- **PET**: Peritoneal Equilibration Test classifying membrane transport
- **CCI**: Charlson Comorbidity Index capturing patient complexity
- **Clinical Features**: Demographics, comorbidities, PD prescription, lab values

The models predict clinical outcomes to help identify patients at risk of inadequate dialysis without requiring immediate lab tests.

## Model Artifacts

Trained models and artifacts are stored in `detect_pd/artifacts/`:
- `models/` - Serialized ML models (CatBoost .cbm, XGBoost JSON)
- `evaluation/` - Performance plots and metrics
- `feature_selection/` - SHAP plots and selected feature lists

## Dependencies and Versions

- **Python**: 3.8+ required
- **Key ML Libraries**: scikit-learn, XGBoost, CatBoost, ZenML, MLflow
- **Frontend**: Vanilla JavaScript (ES6+), no build tools required
- **Data**: Excel processing via pandas/openpyxl

## Development Notes

- Use the existing configuration system rather than hardcoding parameters
- Follow the established ZenML step pattern for new pipeline components
- Clinical calculations should use the utility functions in `utils/clinical.py`
- Frontend features should be unit tested in the `tests/` directory
- Maintain compatibility with the Excel CRF structure defined in the pipeline config