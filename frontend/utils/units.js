export function normalizeHeight(value, unit) {
  if (!isFiniteNumber(value)) return null;
  if (unit === 'cm') return value;
  if (unit === 'in') return value * 2.54;
  return null;
}

export function normalizeWeight(value, unit) {
  if (!isFiniteNumber(value)) return null;
  if (unit === 'kg') return value;
  if (unit === 'lb') return value / 2.20462;
  return null;
}

export function convertLabValue(analyte, value, fromUnit, toUnit) {
  if (!isFiniteNumber(value) || fromUnit === toUnit) return value;
  const key = `${analyte}:${fromUnit}->${toUnit}`.toLowerCase();
  const converters = {
    'pdf_urea:mg/dl->mmol/l': (v) => v * 0.1665,
    'pdf_urea:mmol/l->mg/dl': (v) => v / 0.1665,
    'pdf_creatinine:umol/l->mg/dl': (v) => v / 88.4,
    'pdf_creatinine:mg/dl->umol/l': (v) => v * 88.4,
    'pdf_protein:g/l->g/dl': (v) => v / 10,
    'pdf_protein:g/dl->g/l': (v) => v * 10,
    'urine_protein_creatinine:mg/mg->mg/mmol': (v) => v * 113.12,
    'urine_protein_creatinine:mg/mmol->mg/mg': (v) => v / 113.12,
    'blood_urea:mg/dl->mmol/l': (v) => v * 0.1665,
    'blood_urea:mmol/l->mg/dl': (v) => v / 0.1665,
    'blood_creatinine:umol/l->mg/dl': (v) => v / 88.4,
    'blood_creatinine:mg/dl->umol/l': (v) => v * 88.4,
    'blood_albumin:g/l->g/dl': (v) => v / 10,
    'blood_albumin:g/dl->g/l': (v) => v * 10,
    'blood_protein:g/l->g/dl': (v) => v / 10,
    'blood_protein:g/dl->g/l': (v) => v * 10
  };
  const converter = converters[key];
  if (!converter) {
    return value;
  }
  return converter(value);
}

export function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}
