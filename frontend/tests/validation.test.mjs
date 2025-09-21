import assert from 'node:assert/strict';
import { validateFeatures } from '../utils/validation.js';

const config = {
  guards: {
    age: { min: 18, max: 110 },
    height_cm: { min: 120, max: 220 },
    weight_kg: { min: 30, max: 200 },
    labs: {
      blood_creatinine: { min: 0.2, max: 20 }
    }
  }
};

const features = {
  age: 57,
  height_cm: 168,
  weight_kg: 70,
  bmi: 24.8,
  charlson_index: 6,
  pdf_osmolarity: 366.6,
  dwell_time_minutes: 720,
  pdf_urea: 10.2,
  pdf_creatinine: null,
  pdf_protein: 0.15,
  urine_protein_creatinine: 0.8,
  blood_creatinine: 21,
  blood_albumin: 3.6,
  blood_protein: 6.5
};

const result = validateFeatures(features, config);

assert.ok(result.missing.includes('pdf_creatinine'), 'Should require pdf_creatinine');
assert.ok(result.warnings.some((warning) => warning.field === 'blood_creatinine'), 'High creatinine should raise warning');
assert.ok(!result.isValid, 'Validation should fail due to missing field and guard warning.');

features.pdf_creatinine = 0.9;
features.blood_creatinine = 7.2;
const result2 = validateFeatures(features, config);
assert.ok(result2.isValid, 'Once fields corrected, validation should succeed.');
assert.equal(result2.missing.length, 0);

console.log('validation tests passed');
