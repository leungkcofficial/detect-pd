import assert from 'node:assert/strict';
import { loadXGBoostModel } from '../utils/xgboost.js';

const EXPECTED = [0.04153783, 0.0346017, 0.7884713, 0.1353892];

const features = {
  blood_creatinine: 650,
  blood_albumin: 36,
  dwell_time_minutes: 720,
  urine_protein_creatinine: 1200,
  pdf_osmolarity: 366.6,
  bmi: 24.8,
  age: 57,
  charlson_index: 6,
  pdf_urea: 12.5,
  pdf_creatinine: 900
};

const model = await loadXGBoostModel('../models/pet_class_idx_xgboost.json', { ece: 0.39 });
const prediction = model.predict(features);

assert.equal(prediction.model, 'xgboost_classifier');
assert.equal(prediction.probs.length, 4);
prediction.probs.forEach((value, index) => {
  assert.ok(Math.abs(value - EXPECTED[index]) < 1e-6, `Probability mismatch at class ${index}`);
});
assert.equal(prediction.pred_class, 2, 'Expected class 2 (High Average).');

console.log('petPredictor tests passed');
