import { isFiniteNumber } from './units.js';

const REQUIRED_FIELDS = [
  'age',
  'height_cm',
  'weight_kg',
  'bmi',
  'charlson_index',
  'pdf_osmolarity',
  'dwell_time_minutes',
  'pdf_urea',
  'pdf_creatinine',
  'pdf_protein',
  'urine_protein_creatinine',
  'blood_creatinine',
  'blood_albumin',
  'blood_protein'
];

export function validateFeatures(features, config) {
  const missing = [];
  const warnings = [];
  const errors = [];

  for (const field of REQUIRED_FIELDS) {
    if (!fieldInFeature(features, field)) {
      missing.push(field);
    }
  }

  applyGuards(features, config, warnings, errors);

  const isValid = missing.length === 0 && errors.length === 0;
  return { missing, warnings, errors, isValid };
}

function fieldInFeature(features, field) {
  const value = features[field];
  if (value === 0) return true;
  if (value === null || typeof value === 'undefined') return false;
  if (Number.isNaN(value)) return false;
  return true;
}

function applyGuards(features, config, warnings, errors) {
  const guardConfig = config.guards || {};
  checkNumericGuard('age', features.age, guardConfig.age, warnings);
  checkNumericGuard('height_cm', features.height_cm, guardConfig.height_cm, warnings);
  checkNumericGuard('weight_kg', features.weight_kg, guardConfig.weight_kg, warnings);

  const labGuards = guardConfig.labs || {};
  for (const [field, guard] of Object.entries(labGuards)) {
    checkNumericGuard(field, features[field], guard, warnings);
  }

  if (isFiniteNumber(features.bmi) && (features.bmi < 10 || features.bmi > 60)) {
    warnings.push({ field: 'bmi', message: 'BMI outside common clinical range (10–60).' });
  }
  if (!isFiniteNumber(features.pdf_osmolarity)) {
    errors.push({ field: 'pdf_osmolarity', message: 'Unable to compute PDF osmolarity. Check regimen inputs.' });
  }
  if (!isFiniteNumber(features.dwell_time_minutes)) {
    errors.push({ field: 'dwell_time_minutes', message: 'Provide valid inflow/outflow times to compute dwell.' });
  }
}

function checkNumericGuard(field, value, guard, warnings) {
  if (!guard) return;
  if (!isFiniteNumber(value)) return;
  if (typeof guard.min === 'number' && value < guard.min) {
    warnings.push({ field, message: `${formatField(field)} below guard minimum (${guard.min}).` });
  }
  if (typeof guard.max === 'number' && value > guard.max) {
    warnings.push({ field, message: `${formatField(field)} above guard maximum (${guard.max}).` });
  }
}

function formatField(field) {
  return field.replace(/_/g, ' ');
}
