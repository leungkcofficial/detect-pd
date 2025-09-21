const CCI_BASE_WEIGHTS = {
  myocardial_infarction: 1,
  congestive_heart_failure: 1,
  peripheral_vascular_disease: 1,
  cerebrovascular_disease: 1,
  dementia: 1,
  chronic_pulmonary_disease: 1,
  connective_tissue_disease: 1,
  peptic_ulcer_disease: 1,
  mild_liver_disease: 1,
  diabetes: 1,
  diabetes_with_end_organ_damage: 2,
  hemiplegia: 2,
  moderate_or_severe_renal_disease: 2,
  any_tumor: 2,
  leukemia: 2,
  lymphoma: 2,
  moderate_or_severe_liver_disease: 3,
  metastatic_solid_tumor: 6,
  aids_hiv: 6
};

export function computeCharlsonIndex({
  comorbidityCodes,
  age,
  includeRenal = true,
  includeAge = true,
  customWeights
}) {
  const weights = { ...CCI_BASE_WEIGHTS, ...(customWeights || {}) };
  let score = 0;
  if (Array.isArray(comorbidityCodes)) {
    for (const code of comorbidityCodes) {
      score += weights[code] || 0;
    }
  }
  if (includeRenal) {
    score += weights['moderate_or_severe_renal_disease'] || 0;
  }
  if (includeAge && typeof age === 'number' && Number.isFinite(age)) {
    score += ageAdjustment(age);
  }
  return score;
}

export function ageAdjustment(age) {
  if (age < 50) return 0;
  if (age < 60) return 1;
  if (age < 70) return 2;
  if (age < 80) return 3;
  if (age < 90) return 4;
  return 5;
}
