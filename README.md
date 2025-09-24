# DETECT-PD: Dialysis Efficiency and Transporter Evaluation Computational Tool

A machine learning system for predicting dialysis adequacy (Kt/V) and peritoneal membrane transport status (PET) in peritoneal dialysis patients using clinical and biochemical data.

## Overview

DETECT-PD leverages AI to predict two critical measures for peritoneal dialysis patient management:

- **Kt/V**: Measure of dialysis adequacy indicating waste removal efficiency (target ≥ 1.7)
- **PET Classification**: Peritoneal membrane transport categorization (Low, Low Average, High Average, High)

The system consists of a machine learning pipeline for model training and a web-based frontend for clinical predictions.

## 🏗️ Architecture

### ML Pipeline (`detect_pd/`)
- **Framework**: ZenML for orchestration, MLflow for experiment tracking
- **Models**: XGBoost, CatBoost, LightGBM, Random Forest, ElasticNet
- **Features**: Demographics, comorbidities, lab values, treatment timeline
- **Outputs**: Trained models, evaluation metrics, feature importance

### Frontend (`frontend/`)
- **Type**: Static web application (vanilla JavaScript)
- **Features**: Clinical data input, feature derivation, ML inference
- **Deployment**: Serves via Python HTTP server or static hosting

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (for frontend testing)

### 1. Setup Environment

```bash
# Clone repository
git clone git@github.com:leungkcofficial/detect-pd.git
cd detect-pd

# Install Python dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

### 2. Run ML Pipeline

```bash
cd detect_pd
python -m detect_pd.pipelines.training_pipeline --config configs/pipeline.yaml
```

### 3. Launch Frontend

```bash
cd frontend
python -m http.server 8080
# Visit http://localhost:8080/index.html
```

## 📁 Project Structure

```
detect_pd/
├── src/detect_pd/
│   ├── config/          # Pydantic configuration models
│   ├── pipelines/       # ZenML pipeline definitions
│   ├── steps/           # Individual pipeline steps
│   └── utils/           # Clinical calculations & utilities
├── configs/             # YAML configuration files
├── data/                # Training data (Excel CRF)
├── artifacts/           # Model outputs & evaluation results
└── tests/               # Unit tests for pipeline components

frontend/
├── config/              # Frontend configuration (units, guards, API)
├── utils/               # Feature engineering & validation modules
├── tests/               # Unit tests for frontend logic
├── styles/              # CSS styling
├── index.html           # Main application interface
└── app.js               # Core application logic
```

## 🧠 Machine Learning Pipeline

### Data Flow
1. **Ingestion**: Load patient data from Excel CRF
2. **Preprocessing**: Feature engineering (CCI, BMI, time intervals, osmolarity)
3. **Feature Selection**: LASSO-based feature importance
4. **Training**: Multi-model ensemble with hyperparameter optimization
5. **Evaluation**: Performance metrics, calibration analysis, SHAP explanations

### Key Features
- **Clinical Features**: Age, BMI, Charlson Comorbidity Index
- **Treatment Timeline**: Failure period, waiting period, PD vintage
- **Lab Values**: PDF/blood biochemistry (urea, creatinine, protein, albumin)
- **Prescription**: PD modality, dialysate composition, dwell times

### Models
- **Kt/V Prediction**: CatBoost, XGBoost, ElasticNet, NGBoost
- **PET Classification**: XGBoost, LightGBM, CatBoost, Random Forest

## 🌐 Frontend Application

### Features
- **Tabbed Interface**: Demographics, comorbidities, regimen, labs
- **Unit Handling**: Automatic conversion between clinical units (mg/dL ↔ mmol/L, g/dL ↔ g/L)
- **Validation**: Required field checks, clinical guard bands
- **Derived Metrics**: Real-time BMI, CCI, osmolarity calculation
- **Predictions**: Kt/V with confidence intervals, PET class probabilities
- **Export**: JSON audit trail with timestamps

### Configuration
Edit `frontend/config/frontend_config.json` to customize:
- Lab unit preferences and conversions
- Clinical guard bands for validation
- Dialysate bag formulations
- API endpoints and timeouts

## 🧪 Testing

### ML Pipeline Tests
```bash
cd detect_pd
python -m pytest src/tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm run test:unit
# or manually:
node tests/deriveFeatures.test.mjs
node tests/validation.test.mjs
```

## 📊 Clinical Context

### Peritoneal Dialysis Adequacy
- **Kt/V ≥ 1.7**: Adequate dialysis dose
- **PET Categories**: 
  - Low: Slow transport, longer dwells
  - High: Fast transport, shorter dwells
  - Guides prescription optimization

### Charlson Comorbidity Index (CCI)
- Standardized comorbidity scoring system
- Accounts for age and disease severity
- Predicts mortality and treatment outcomes

### Laboratory Units
- **Default Units**: Match training data (umol/L, g/L, mg/mmol)
- **Conversions**: Built-in support for regional unit variations
- **Validation**: Clinical guard bands prevent data entry errors

## 🔧 Configuration

### Pipeline Config (`detect_pd/configs/pipeline.yaml`)
- Data ingestion parameters and column mappings
- Feature engineering options (CCI calculation, time intervals)
- Model hyperparameters and search spaces
- Evaluation metrics and output paths

### Frontend Config (`frontend/config/frontend_config.json`)
- Unit preferences and conversion factors
- Clinical validation rules and guard bands
- Dialysate bag database with osmolarity values
- API settings for model inference

## 📈 Model Performance

Current model benchmarks:
- **Kt/V**: R² ≈ 0.451, MAE ≈ 0.241
- **PET**: QWK ≈ 0.325, ECE ≈ 0.39

Evaluation artifacts in `detect_pd/artifacts/evaluation/`:
- Calibration plots
- Discrimination analysis  
- Feature importance (SHAP)
- Cross-validation metrics

## 🚀 Deployment

### Frontend Deployment
```bash
# Static hosting
cd frontend
python -m http.server 8080

# Or copy to web server
cp -r frontend/* /var/www/html/
```

### Model API Deployment
Configure `frontend/config/frontend_config.json`:
```json
{
  "api": {
    "useMock": false,
    "baseUrl": "https://your-api-endpoint.com",
    "predictEndpoint": "/predict"
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Add unit tests for new features
- Update documentation for API changes
- Ensure clinical accuracy for medical calculations

## 📚 Documentation

- `CLAUDE.md`: Development guide for Claude Code
- `PRD.md`: Product Requirements Document
- `task.md`: Detailed task breakdown
- Component READMEs in respective directories

## ⚖️ License

This project is for research and educational purposes. Clinical use requires appropriate validation and regulatory approval.

## 🙏 Acknowledgments

- Clinical domain expertise from peritoneal dialysis specialists
- ZenML and MLflow communities for MLOps frameworks
- Open source machine learning libraries (scikit-learn, XGBoost, CatBoost)

---

**⚠️ Medical Disclaimer**: This software is for research purposes only. Clinical decisions should always be made by qualified healthcare professionals with appropriate validation of predictions.