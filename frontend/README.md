# DETECT-PD Frontend

A stateless clinician-facing calculator that normalizes peritoneal dialysis inputs, derives Charlson/BMI/osmolarity features, and calls packaged ML models (or the bundled mock service) for Kt/V and PET class predictions.

## Quick Start

1. **Serve the static app** from the project root:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   Visit `http://localhost:8080` and open `index.html`.

2. **Remote inference (optional)**: open `config/frontend_config.json` and set `api.useMock` to `false`, then point `api.baseUrl` to the DETECT-PD prediction API (`predictEndpoint` defaults to `/predict`). PET class predictions will continue to use the bundled XGBoost model; only Kt/V falls back to the remote service.

3. **Run unit checks** for feature engineering and validation:
   ```bash
   cd frontend
   node tests/deriveFeatures.test.mjs
   node tests/validation.test.mjs
   # or
   node --experimental-specifier-resolution=node tests/deriveFeatures.test.mjs
   ```
   The sample `package.json` also defines `npm run test:unit` when npm is available.

## Feature Highlights

- **Tabbed intake** for demographics/comorbidities, regimen configuration, and lab inputs; all units normalized to the model contract defined in `PRD_frontend.md`.
- **Derived metric pills** for BMI, Charlson Index (with age adjustment and renal baseline), dwell time, and PDF osmolarity (supports CAPD, NIPD, CCPD flows with volume-weighted means).
- **Validation engine** enforcing required features plus guard-band warnings from `frontend_config.json` before enabling prediction.
- **Inference adapter** loads the packaged PET XGBoost bundle for on-device predictions and can optionally call a remote `/predict` endpoint for Kt/V or future services.
- **Results dashboard** rendering Kt/V value + 95% PI, PET class probabilities/top drivers, and calibration notes.
- **JSON export** capturing request/response pairs with a timestamp and config version for audit trails.

## Configuration

Editable via `config/frontend_config.json`:
- `bags`: dialysate bag dictionary (brand, concentration, osmolarity in mOsm/L).
- `units`: default unit selectors and conversion support for height, weight, and lab inputs (defaults mirror 08_a_crf.xlsx: mmol/L urea, umol/L creatinine, g/L proteins, mg/mmol urine ratio).
- `guards`: clinical guard bands surfaced as inline warnings.
- `api`: base URL, endpoint, timeout, and mock toggle for inference (local PET predictions remain active even when remote calls fail).
- `calibration`: text displayed alongside predictions for clinician context.

Changes to the configuration do not require code edits; reload the page to pick up new values.

## Testing Notes

- `frontend/tests/deriveFeatures.test.mjs` covers BMI parity, Charlson computation, dwell rollover, osmolarity (366 mOsm/L worked example), and unit conversions.
- `frontend/tests/validation.test.mjs` verifies required field detection and guard-band warnings.
- `frontend/tests/petPredictor.test.mjs` asserts PET probabilities match the exported XGBoost model to 1e-6.
- Additional UI flows can be exercised manually by running the static server and entering the example payload from `PRD_frontend.md`.

## Next Steps

- Wire the inference client to the packaged CatBoost/XGBoost service once the backend endpoint is available.
- Add Playwright/Axe-based accessibility checks and e2e regression coverage (per `task_frontend.md` Phase 8).
- Containerize the static bundle plus artifacts for deployment as outlined in `design_frontend.md`.
