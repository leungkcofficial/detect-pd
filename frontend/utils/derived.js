import { normalizeHeight, normalizeWeight, convertLabValue, isFiniteNumber } from './units.js';
import { computeCharlsonIndex } from './cci.js';
import { calculateOsmolarity } from './osmolarity.js';

const CHARLSON_MAP = {
  ht: 'hypertension',
  mi: 'myocardial_infarction',
  chf: 'congestive_heart_failure',
  pvd: 'peripheral_vascular_disease',
  cva: 'cerebrovascular_disease',
  dementia: 'dementia',
  copd: 'chronic_pulmonary_disease',
  rheumatic: 'connective_tissue_disease',
  mild_liver: 'mild_liver_disease',
  severe_liver: 'moderate_or_severe_liver_disease',
  dm_no_cx: 'diabetes',
  dm_cx: 'diabetes_with_end_organ_damage',
  hemiplegia: 'hemiplegia',
  malignancy: 'any_tumor',
  metastatic: 'metastatic_solid_tumor',
  hiv_aids: 'aids_hiv'
};

export function deriveFeatures(input, config) {
  const age = parseMaybeNumber(input.patient?.age);
  const heightCm = normalizeHeight(parseMaybeNumber(input.patient?.heightValue), input.patient?.heightUnit || config.units?.height?.default);
  const weightKg = normalizeWeight(parseMaybeNumber(input.patient?.weightValue), input.patient?.weightUnit || config.units?.weight?.default);

  const bmi = computeBmi(weightKg, heightCm);

  const comorbidityCodes = buildComorbidityCodes(input.comorbidities);
  const charlsonIndex = computeCharlsonIndex({
    comorbidityCodes,
    age,
    includeRenal: true,
    includeAge: true
  });

  const dwellTimeMinutes = computeDwellMinutes(input.times?.inflow, input.times?.outflow);

  const bagLookup = new Map((config.bags || []).map((bag) => [bag.id, bag]));
  const osmResult = calculateOsmolarity({
    modality: input.regimen?.modality,
    bagLookup,
    capdBagId: input.regimen?.capdBagId,
    capdVolumeL: parseMaybeNumber(input.regimen?.capdVolumeL),
    nipdShifts: input.regimen?.nipdShifts,
    ccpdDayBagId: input.regimen?.ccpdDayBagId,
    ccpdNightShifts: input.regimen?.ccpdNightShifts
  });

  const normalizedLabs = normalizeLabs(input.labs, config.units?.labs);

  return {
    patient_id: generateEphemeralId(),
    age,
    height_cm: heightCm,
    weight_kg: weightKg,
    bmi,
    charlson_index: charlsonIndex,
    pdf_osmolarity: osmResult.value,
    dwell_time_minutes: dwellTimeMinutes,
    ...normalizedLabs.values,
    _meta: {
      comorbidityCodes,
      osmMeta: osmResult.meta,
      labConversions: normalizedLabs.meta,
      dwellSource: buildDwellMeta(input.times)
    }
  };
}

export function computeBmi(weightKg, heightCm) {
  if (!isFiniteNumber(weightKg) || !isFiniteNumber(heightCm) || heightCm <= 0) {
    return null;
  }
  const heightM = heightCm / 100;
  return weightKg / (heightM ** 2);
}

function computeDwellMinutes(inflowTime, outflowTime) {
  if (!inflowTime || !outflowTime) return null;
  const inflow = parseTime(inflowTime);
  const outflow = parseTime(outflowTime);
  if (inflow == null || outflow == null) return null;
  let delta = outflow - inflow;
  if (delta <= 0) {
    delta += 24 * 60;
  }
  return delta;
}

function parseTime(value) {
  if (typeof value !== 'string') return null;
  const [hh, mm] = value.split(':');
  const hours = Number.parseInt(hh, 10);
  const mins = Number.parseInt(mm, 10);
  if (!Number.isInteger(hours) || !Number.isInteger(mins)) return null;
  if (hours < 0 || hours > 23 || mins < 0 || mins > 59) return null;
  return hours * 60 + mins;
}

function buildComorbidityCodes(comorbidities = {}) {
  const codes = [];
  for (const [key, selected] of Object.entries(comorbidities)) {
    if (!selected) continue;
    const mapped = CHARLSON_MAP[key];
    if (mapped) {
      codes.push(mapped);
    }
  }
  return codes;
}

function normalizeLabs(labs = {}, labUnitConfig = {}) {
  const values = {};
  const meta = {};
  for (const [field, payload] of Object.entries(labs)) {
    if (!payload) continue;
    const expected = labUnitConfig[field]?.expected || payload.unit;
    const fromUnit = payload.unit || expected;
    const numericValue = parseMaybeNumber(payload.value);
    if (numericValue == null) {
      values[field] = null;
      continue;
    }
    const converted = convertLabValue(field, numericValue, fromUnit, expected);
    values[field] = converted;
    meta[field] = fromUnit === expected ? null : `${numericValue} ${fromUnit} → ${converted.toFixed(3)} ${expected}`;
  }
  return { values, meta };
}

function buildDwellMeta(times) {
  if (!times?.inflow || !times?.outflow) return 'Incomplete';
  return `${times.inflow} → ${times.outflow}`;
}

function parseMaybeNumber(value) {
  if (value === '' || value === null || typeof value === 'undefined') {
    return null;
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return null;
  }
  return num;
}

function generateEphemeralId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `patient-${Math.random().toString(36).slice(2, 10)}`;
}
