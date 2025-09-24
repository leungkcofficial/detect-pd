import assert from 'node:assert/strict';
import { deriveFeatures } from '../utils/derived.js';

const config = {
  bags: [
    { id: 'bag_a', brand: 'Baxter Dianeal', concentration: '1.5% glucose', osm_mOsmL: 346 },
    { id: 'bag_b', brand: 'Baxter Dianeal', concentration: '2.5% glucose', osm_mOsmL: 396 }
  ],
  units: {
    height: { default: 'cm' },
    weight: { default: 'kg' },
    labs: {
      pdf_urea: { expected: 'mmol/L' },
      pdf_creatinine: { expected: 'umol/L' },
      pdf_protein: { expected: 'g/L' },
      urine_protein_creatinine: { expected: 'mg/mmol' },
      blood_urea: { expected: 'mmol/L' },
      blood_creatinine: { expected: 'umol/L' },
      blood_albumin: { expected: 'g/L' },
      blood_protein: { expected: 'g/L' }
    }
  }
};

const state = {
  patient: { age: 57, heightValue: 168, heightUnit: 'cm', weightValue: 70, weightUnit: 'kg' },
  comorbidities: {
    ht: false,
    mi: false,
    chf: true,
    pvd: false,
    cva: false,
    dementia: false,
    copd: false,
    rheumatic: false,
    mild_liver: false,
    severe_liver: false,
    dm_no_cx: false,
    dm_cx: true,
    hemiplegia: false,
    malignancy: false,
    metastatic: false,
    hiv_aids: false
  },
  regimen: {
    modality: 'NIPD',
    capdBagId: '',
    capdVolumeL: '',
    nipdShifts: [
      { bagId: 'bag_a', volumeL: '2', count: '3' },
      { bagId: 'bag_b', volumeL: '2', count: '2' }
    ],
    ccpdDayBagId: '',
    ccpdNightShifts: []
  },
  times: {
    inflow: '20:00',
    outflow: '08:00'
  },
  labs: {
    pdf_urea: { value: 10.2, unit: 'mmol/L' },
    pdf_creatinine: { value: 0.9, unit: 'mg/dL' },
    pdf_protein: { value: 0.15, unit: 'g/dL' },
    urine_protein_creatinine: { value: 12, unit: 'mg/mg' },
    blood_urea: { value: 25, unit: 'mg/dL' },
    blood_creatinine: { value: 7.2, unit: 'mg/dL' },
    blood_albumin: { value: 3.6, unit: 'g/dL' },
    blood_protein: { value: 6.5, unit: 'g/dL' }
  }
};

const result = deriveFeatures(state, config);

assert.ok(Math.abs(result.bmi - 24.8) < 0.05, `BMI expected ~24.8, received ${result.bmi}`);
assert.equal(result.charlson_index, 6, 'Charlson index should include CHF, DM with complications, renal baseline, age bracket.');
assert.ok(Math.abs(result.pdf_osmolarity - 366) < 0.2, `Expected osmolarity ~366.6, received ${result.pdf_osmolarity}`);
assert.equal(result.dwell_time_minutes, 720, 'Expected dwell of 12 hours (720 minutes).');
assert.ok(Math.abs(result.pdf_creatinine - (0.9 * 88.4)) < 0.01, 'PDF creatinine conversion to umol/L failed.');
assert.ok(Math.abs(result.pdf_protein - 1.5) < 0.01, 'PDF protein conversion to g/L failed.');
assert.ok(Math.abs(result.urine_protein_creatinine - (12 * 113.12)) < 0.5, 'Urine protein/creatinine conversion to mg/mmol failed.');
assert.ok(Math.abs(result.blood_urea - (25 * 0.1665)) < 0.01, 'Blood urea conversion to mmol/L failed.');
assert.ok(Math.abs(result.blood_creatinine - (7.2 * 88.4)) < 0.05, 'Blood creatinine conversion to umol/L failed.');
assert.ok(Math.abs(result.blood_albumin - 36) < 0.01, 'Blood albumin conversion to g/L failed.');
assert.ok(Math.abs(result.blood_protein - 65) < 0.1, 'Blood protein conversion to g/L failed.');

console.log('deriveFeatures tests passed');
