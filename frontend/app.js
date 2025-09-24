import { loadConfig } from './utils/config.js';
import { deriveFeatures } from './utils/derived.js';
import { validateFeatures } from './utils/validation.js';
import { createInferenceClient } from './utils/inference.js';
import { showToast } from './utils/toast.js';
import { formatNumber, formatInteger, formatProbability, featureLabel } from './utils/format.js';

const COMORBIDITY_FIELDS = [
  { key: 'ht', label: 'Hypertension (HT)' },
  { key: 'mi', label: 'Myocardial infarction' },
  { key: 'chf', label: 'Congestive heart failure' },
  { key: 'pvd', label: 'Peripheral vascular disease' },
  { key: 'cva', label: 'Cerebrovascular disease' },
  { key: 'dementia', label: 'Dementia' },
  { key: 'copd', label: 'Chronic pulmonary disease' },
  { key: 'rheumatic', label: 'Rheumatic disease' },
  { key: 'mild_liver', label: 'Mild liver disease' },
  { key: 'severe_liver', label: 'Severe liver disease' },
  { key: 'dm_no_cx', label: 'DM w/o complications' },
  { key: 'dm_cx', label: 'DM with complications' },
  { key: 'hemiplegia', label: 'Hemi/Paraplegia' },
  { key: 'malignancy', label: 'Any malignancy' },
  { key: 'metastatic', label: 'Metastatic cancer' },
  { key: 'hiv_aids', label: 'HIV / AIDS' }
];

const LAB_FIELDS = [
  { key: 'pdf_urea', label: 'PDF Urea', helper: 'Default mmol/L; converter available for mg/dL.' },
  { key: 'pdf_creatinine', label: 'PDF Creatinine', helper: 'Default umol/L; converter available for mg/dL.' },
  { key: 'pdf_protein', label: 'PDF Protein', helper: 'Default g/L; converter available for g/dL.' },
  { key: 'urine_protein_creatinine', label: 'Urine Protein/Creatinine Ratio', helper: 'Default mg/mmol; mg/mg supported.' },
  { key: 'blood_urea', label: 'Blood Urea', helper: 'Default mmol/L; converter available for mg/dL.' },
  { key: 'blood_creatinine', label: 'Blood Creatinine', helper: 'Default umol/L; converter available for mg/dL.' },
  { key: 'blood_albumin', label: 'Blood Albumin', helper: 'Default g/L; converter available for g/dL.' },
  { key: 'blood_protein', label: 'Blood Total Protein', helper: 'Default g/L; converter available for g/dL.' }
];

const PET_CLASS_LABELS = {
  0: 'Low',
  1: 'Low Average', 
  2: 'High Average',
  3: 'High'
};

function getPetClassLabel(classIndex) {
  return PET_CLASS_LABELS[classIndex] || `Class ${classIndex}`;
}

const state = {
  config: null,
  inference: null,
  patient: {
    age: '',
    heightValue: '',
    heightUnit: 'cm',
    weightValue: '',
    weightUnit: 'kg'
  },
  comorbidities: COMORBIDITY_FIELDS.reduce((acc, item) => ({ ...acc, [item.key]: false }), {}),
  regimen: {
    modality: '',
    capdBagId: '',
    capdVolumeL: '',
    nipdShifts: [],
    ccpdDayBagId: '',
    ccpdNightShifts: []
  },
  times: {
    inflow: '',
    outflow: ''
  },
  labs: {},
  derived: {},
  validation: { missing: [], warnings: [], errors: [], isValid: false },
  prediction: null
};

window.addEventListener('DOMContentLoaded', init);

async function init() {
  try {
    const config = await loadConfig();
    state.config = config;
    state.inference = createInferenceClient(config);
    initialiseState();
    setupTabs();
    renderPatientPanel();
    renderRegimenPanel();
    renderLabsPanel();
    renderResults(null);
    bindGlobalActions();
    recomputeDerived();
    showToast('Configuration loaded. Ready for inputs.', 'info', 3500);
  } catch (error) {
    console.error(error);
    showToast('Failed to initialize application. See console for details.', 'error');
  }
}

function initialiseState() {
  const { config } = state;
  state.patient.heightUnit = config.units?.height?.default || 'cm';
  state.patient.weightUnit = config.units?.weight?.default || 'kg';
  state.times.inflow = config.dwell?.defaultInflow || '';
  state.times.outflow = config.dwell?.defaultOutflow || '';
  // Seed lab units with defaults
  for (const field of LAB_FIELDS) {
    const expected = config.units?.labs?.[field.key]?.expected;
    const defaultUnit = expected || config.units?.labs?.[field.key]?.options?.[0];
    state.labs[field.key] = { value: '', unit: defaultUnit };
  }
}

function setupTabs() {
  const tabs = document.querySelectorAll('.tab-button');
  const panels = document.querySelectorAll('.tab-panel');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('aria-controls');
      tabs.forEach((btn) => btn.setAttribute('aria-selected', String(btn === tab)));
      panels.forEach((panel) => {
        if (panel.id === target) {
          panel.classList.add('active');
        } else {
          panel.classList.remove('active');
        }
      });
    });
  });
}

function bindGlobalActions() {
  document.getElementById('predict-button').addEventListener('click', handlePredict);
  document.getElementById('reset-button').addEventListener('click', resetForm);
  document.getElementById('download-button').addEventListener('click', downloadJson);
}

function renderPatientPanel() {
  const panel = document.getElementById('panel-patient');
  const { patient } = state;
  panel.innerHTML = `
    <div class="form-grid">
      <div class="field">
        <label for="age-input">Age (years)</label>
        <input type="number" id="age-input" min="0" max="120" value="${patient.age}" />
        <p class="helper-text">Recommended 18–110 years.</p>
      </div>
      <div class="field">
        <label for="height-input">Height</label>
        <div class="unit-group">
          <input type="number" id="height-input" min="0" step="0.1" value="${patient.heightValue}" />
          <select id="height-unit">
            ${renderOptions(state.config.units?.height?.options, patient.heightUnit)}
          </select>
        </div>
        <p class="helper-text">Auto-converted to cm.</p>
      </div>
      <div class="field">
        <label for="weight-input">Weight</label>
        <div class="unit-group">
          <input type="number" id="weight-input" min="0" step="0.1" value="${patient.weightValue}" />
          <select id="weight-unit">
            ${renderOptions(state.config.units?.weight?.options, patient.weightUnit)}
          </select>
        </div>
        <p class="helper-text">Auto-converted to kg.</p>
      </div>
    </div>
    <div class="field" aria-live="polite">
      <label>Charlson comorbidities</label>
      <div class="toggle-grid">
        ${COMORBIDITY_FIELDS.map((item) => renderComorbidityToggle(item, state.comorbidities[item.key])).join('')}
      </div>
    </div>
  `;

  panel.querySelector('#age-input').addEventListener('input', (event) => {
    state.patient.age = event.target.value;
    recomputeDerived();
  });
  panel.querySelector('#height-input').addEventListener('input', (event) => {
    state.patient.heightValue = event.target.value;
    recomputeDerived();
  });
  panel.querySelector('#weight-input').addEventListener('input', (event) => {
    state.patient.weightValue = event.target.value;
    recomputeDerived();
  });
  panel.querySelector('#height-unit').addEventListener('change', (event) => {
    state.patient.heightUnit = event.target.value;
    recomputeDerived();
  });
  panel.querySelector('#weight-unit').addEventListener('change', (event) => {
    state.patient.weightUnit = event.target.value;
    recomputeDerived();
  });

  panel.querySelectorAll('.toggle input').forEach((input) => {
    input.addEventListener('change', (event) => {
      const key = event.target.dataset.key;
      state.comorbidities[key] = event.target.checked;
      recomputeDerived();
    });
  });
}

function renderComorbidityToggle(item, checked) {
  return `
    <label class="toggle">
      <input type="checkbox" data-key="${item.key}" ${checked ? 'checked' : ''} />
      <span>${item.label}</span>
    </label>
  `;
}

function renderRegimenPanel() {
  const panel = document.getElementById('panel-regimen');
  const { regimen, times } = state;
  panel.innerHTML = `
    <div class="form-grid">
      <div class="field">
        <label for="modality-select">PD Modality</label>
        <select id="modality-select">
          <option value="">Select modality</option>
          ${['CAPD', 'NIPD', 'CCPD'].map((mod) => `<option value="${mod}" ${regimen.modality === mod ? 'selected' : ''}>${mod}</option>`).join('')}
        </select>
      </div>
      <div class="field" id="capd-group" style="${regimen.modality === 'CAPD' ? '' : 'display:none;'}">
        <label for="capd-bag">CAPD day dwell bag</label>
        <select id="capd-bag">
          <option value="">Select bag</option>
          ${renderBagOptions(regimen.capdBagId)}
        </select>
        <p class="helper-text">Optional volume override (L):</p>
        <input type="number" id="capd-volume" min="0" step="0.1" value="${regimen.capdVolumeL || ''}" placeholder="e.g., 2.0" />
      </div>
      <div class="field" id="ccpd-day-group" style="${regimen.modality === 'CCPD' ? '' : 'display:none;'}">
        <label for="ccpd-day-bag">CCPD day dwell bag</label>
        <select id="ccpd-day-bag">
          <option value="">Select bag (optional)</option>
          ${renderBagOptions(regimen.ccpdDayBagId)}
        </select>
      </div>
    </div>

    <div class="field" id="nipd-shifts-block" style="${regimen.modality === 'NIPD' ? '' : 'display:none;'}">
      <label>NIPD night cycles</label>
      <div class="helper-text">Specify up to six cycles (bag × volume × count).</div>
      <div class="shift-list" id="nipd-shifts"></div>
      <button type="button" class="secondary" id="add-nipd-shift">Add cycle</button>
    </div>

    <div class="field" id="ccpd-night-block" style="${regimen.modality === 'CCPD' && !regimen.ccpdDayBagId ? '' : 'display:none;'}">
      <label>CCPD machine cycles</label>
      <div class="helper-text">Required when no day dwell bag selected.</div>
      <div class="shift-list" id="ccpd-shifts"></div>
      <button type="button" class="secondary" id="add-ccpd-shift">Add cycle</button>
    </div>

    <div class="form-grid">
      <div class="field">
        <label for="inflow-time">Inflow time</label>
        <input type="time" id="inflow-time" value="${times.inflow || ''}" />
      </div>
      <div class="field">
        <label for="outflow-time">Outflow time</label>
        <input type="time" id="outflow-time" value="${times.outflow || ''}" />
      </div>
    </div>
  `;

  panel.querySelector('#modality-select').addEventListener('change', (event) => {
    state.regimen.modality = event.target.value;
    if (state.regimen.modality !== 'CAPD') {
      state.regimen.capdBagId = '';
    }
    if (state.regimen.modality !== 'NIPD') {
      state.regimen.nipdShifts = [];
    }
    if (state.regimen.modality !== 'CCPD') {
      state.regimen.ccpdDayBagId = '';
      state.regimen.ccpdNightShifts = [];
    }
    renderRegimenPanel();
    recomputeDerived();
  });

  const capdBag = panel.querySelector('#capd-bag');
  if (capdBag) {
    capdBag.addEventListener('change', (event) => {
      state.regimen.capdBagId = event.target.value;
      recomputeDerived();
    });
  }
  const capdVolume = panel.querySelector('#capd-volume');
  if (capdVolume) {
    capdVolume.addEventListener('input', (event) => {
      state.regimen.capdVolumeL = event.target.value;
      recomputeDerived();
    });
  }

  const ccpdDayBag = panel.querySelector('#ccpd-day-bag');
  if (ccpdDayBag) {
    ccpdDayBag.addEventListener('change', (event) => {
      state.regimen.ccpdDayBagId = event.target.value;
      renderRegimenPanel();
      recomputeDerived();
    });
  }

  panel.querySelector('#inflow-time').addEventListener('change', (event) => {
    state.times.inflow = event.target.value;
    recomputeDerived();
  });
  panel.querySelector('#outflow-time').addEventListener('change', (event) => {
    state.times.outflow = event.target.value;
    recomputeDerived();
  });

  renderShiftSection('nipd', state.regimen.nipdShifts, panel.querySelector('#nipd-shifts'));
  renderShiftSection('ccpd', state.regimen.ccpdNightShifts, panel.querySelector('#ccpd-shifts'));

  const nipdAdd = panel.querySelector('#add-nipd-shift');
  if (nipdAdd) {
    nipdAdd.addEventListener('click', () => {
      if (state.regimen.nipdShifts.length >= 6) {
        showToast('Maximum of six cycles supported.', 'warning');
        return;
      }
      state.regimen.nipdShifts.push(createEmptyShift());
      renderRegimenPanel();
      recomputeDerived();
    });
  }

  const ccpdAdd = panel.querySelector('#add-ccpd-shift');
  if (ccpdAdd) {
    ccpdAdd.addEventListener('click', () => {
      if (state.regimen.ccpdNightShifts.length >= 6) {
        showToast('Maximum of six cycles supported.', 'warning');
        return;
      }
      state.regimen.ccpdNightShifts.push(createEmptyShift());
      renderRegimenPanel();
      recomputeDerived();
    });
  }
}

function createEmptyShift() {
  return { bagId: '', volumeL: '', count: '' };
}

function renderShiftSection(prefix, shifts, container) {
  if (!container) return;
  if (!Array.isArray(shifts) || shifts.length === 0) {
    container.innerHTML = '<p class="helper-text">No cycles configured.</p>';
    return;
  }
  container.innerHTML = shifts
    .map((shift, index) => `
      <div class="card" data-shift-index="${index}" data-shift-prefix="${prefix}">
        <div class="form-grid">
          <div class="field">
            <label>Bag</label>
            <select data-action="bag">
              <option value="">Select bag</option>
              ${renderBagOptions(shift.bagId)}
            </select>
          </div>
          <div class="field">
            <label>Volume (L)</label>
            <input type="number" min="0" step="0.1" data-action="volume" value="${shift.volumeL || ''}" />
          </div>
          <div class="field">
            <label>Count</label>
            <input type="number" min="1" max="10" step="1" data-action="count" value="${shift.count || ''}" />
          </div>
        </div>
        <button type="button" class="secondary" data-action="remove">Remove</button>
      </div>
    `)
    .join('');

  container.querySelectorAll('[data-action="bag"]').forEach((element) => {
    element.addEventListener('change', (event) => {
      const { index, prefix: pref } = shiftIndexFromElement(event.target);
      const list = pref === 'nipd' ? state.regimen.nipdShifts : state.regimen.ccpdNightShifts;
      list[index].bagId = event.target.value;
      recomputeDerived();
    });
  });
  container.querySelectorAll('[data-action="volume"]').forEach((element) => {
    element.addEventListener('input', (event) => {
      const { index, prefix: pref } = shiftIndexFromElement(event.target);
      const list = pref === 'nipd' ? state.regimen.nipdShifts : state.regimen.ccpdNightShifts;
      list[index].volumeL = event.target.value;
      recomputeDerived();
    });
  });
  container.querySelectorAll('[data-action="count"]').forEach((element) => {
    element.addEventListener('input', (event) => {
      const { index, prefix: pref } = shiftIndexFromElement(event.target);
      const list = pref === 'nipd' ? state.regimen.nipdShifts : state.regimen.ccpdNightShifts;
      list[index].count = event.target.value;
      recomputeDerived();
    });
  });
  container.querySelectorAll('[data-action="remove"]').forEach((element) => {
    element.addEventListener('click', (event) => {
      const { index, prefix: pref } = shiftIndexFromElement(event.target);
      const list = pref === 'nipd' ? state.regimen.nipdShifts : state.regimen.ccpdNightShifts;
      list.splice(index, 1);
      renderRegimenPanel();
      recomputeDerived();
    });
  });
}

function shiftIndexFromElement(element) {
  const wrapper = element.closest('[data-shift-index]');
  const index = Number(wrapper?.dataset.shiftIndex || 0);
  const prefix = wrapper?.dataset.shiftPrefix;
  return { index, prefix };
}

function renderLabsPanel() {
  const panel = document.getElementById('panel-labs');
  panel.innerHTML = `
    <div class="form-grid">
      ${LAB_FIELDS.map((field) => renderLabField(field)).join('')}
    </div>
  `;
  LAB_FIELDS.forEach((field) => {
    const valueInput = panel.querySelector(`#${field.key}-input`);
    const unitSelect = panel.querySelector(`#${field.key}-unit`);
    valueInput.addEventListener('input', (event) => {
      state.labs[field.key].value = event.target.value;
      recomputeDerived();
    });
    unitSelect.addEventListener('change', (event) => {
      state.labs[field.key].unit = event.target.value;
      recomputeDerived();
    });
  });
}

function renderLabField(field) {
  const labState = state.labs[field.key] || { value: '', unit: '' };
  const unitConfig = state.config.units?.labs?.[field.key];
  const options = unitConfig?.options || [unitConfig?.expected].filter(Boolean);
  return `
    <div class="field">
      <label for="${field.key}-input">${field.label}</label>
      <div class="unit-group">
        <input type="number" step="0.01" id="${field.key}-input" value="${labState.value || ''}" />
        <select id="${field.key}-unit">
          ${renderOptions(options, labState.unit)}
        </select>
      </div>
      <p class="helper-text">Expected: ${unitConfig?.expected || labState.unit}</p>
      ${field.helper ? `<p class="helper-text">${field.helper}</p>` : ''}
    </div>
  `;
}

function renderBagOptions(selectedId) {
  return state.config.bags
    .map((bag) => `<option value="${bag.id}" ${bag.id === selectedId ? 'selected' : ''}>${bag.brand} — ${bag.concentration} (${bag.osm_mOsmL} mOsm/L)</option>`)
    .join('');
}

function renderOptions(options = [], selected) {
  return options
    .map((option) => `<option value="${option}" ${option === selected ? 'selected' : ''}>${option}</option>`)
    .join('');
}

function recomputeDerived() {
  const derived = deriveFeatures(state, state.config);
  state.derived = derived;
  const validation = validateFeatures(derived, state.config);
  state.validation = validation;
  updateDerivedChips(derived);
  updateValidationSummary(validation);
  updateActions(validation);
}

function updateDerivedChips(derived) {
  const mapping = {
    bmi: (value) => formatNumber(value, { digits: 1 }),
    charlson_index: (value) => formatInteger(value),
    dwell_time_minutes: (value) => formatNumber(value, { digits: 0 }),
    pdf_osmolarity: (value) => formatNumber(value, { digits: 1 })
  };
  document.querySelectorAll('.derived-pill').forEach((pill) => {
    const key = pill.dataset.derived;
    const formatter = mapping[key] || ((value) => formatNumber(value));
    pill.querySelector('.derived-value').textContent = formatter(derived[key]);
  });

  const ktvCalibration = document.querySelector('[data-ktv="calibration"]');
  if (ktvCalibration) {
    const ktvNote = state.config.calibration?.ktv;
    const noteText = typeof ktvNote === 'string' ? ktvNote : ktvNote?.note;
    ktvCalibration.textContent = `Calibration: ${noteText || 'Refer to CatBoost bundle'}`;
  }
  const petReliability = document.querySelector('[data-pet="reliability"]');
  if (petReliability && !state.prediction) {
    const petCal = state.config.calibration?.pet_class_idx || {};
    petReliability.textContent = buildReliabilityText(petCal.ece, petCal.note);
  }
}

function updateValidationSummary(validation) {
  const container = document.getElementById('validation-summary');
  if (!container) return;
  container.innerHTML = '';
  if (validation.missing.length > 0) {
    const missingList = validation.missing.map((field) => featureLabel(field)).join(', ');
    const el = document.createElement('div');
    el.className = 'validation-error';
    el.textContent = `Missing required inputs: ${missingList}`;
    container.appendChild(el);
  }
  validation.errors.forEach((error) => {
    const el = document.createElement('div');
    el.className = 'validation-error';
    el.textContent = error.message;
    container.appendChild(el);
  });
  validation.warnings.forEach((warning) => {
    const el = document.createElement('div');
    el.className = 'validation-warning';
    el.textContent = warning.message;
    container.appendChild(el);
  });
}

function updateActions(validation) {
  const predictButton = document.getElementById('predict-button');
  predictButton.disabled = !validation.isValid;

  const downloadButton = document.getElementById('download-button');
  downloadButton.disabled = !state.prediction;
}

async function handlePredict() {
  if (!state.validation.isValid) {
    showToast('Please resolve validation issues before predicting.', 'warning');
    return;
  }
  const button = document.getElementById('predict-button');
  const originalText = button.textContent;
  button.textContent = 'Predicting…';
  button.disabled = true;
  try {
    const response = await state.inference.predict(state.derived);
    state.prediction = {
      requestedAt: new Date().toISOString(),
      request: state.derived,
      response
    };
    updateActions(state.validation);
    renderResults(response);
    showToast('Prediction updated.', 'info', 3000);
  } catch (error) {
    console.error('Prediction error', error);
  } finally {
    button.textContent = originalText;
    button.disabled = !state.validation.isValid;
  }
}

function renderResults(response) {
  const ktvCard = document.getElementById('ktv-card');
  const petCard = document.getElementById('pet-card');
  if (!response) {
    resetResultCard(ktvCard, petCard);
    return;
  }
  const ktv = response.ktv;
  if (ktv) {
    ktvCard.querySelector('[data-ktv="value"]').textContent = formatNumber(ktv.prediction, { digits: 2 });
    ktvCard.querySelector('[data-ktv="interval"]').textContent = ktv.pi_95
      ? `95% PI: ${formatNumber(ktv.pi_95[0], { digits: 2 })} – ${formatNumber(ktv.pi_95[1], { digits: 2 })}`
      : 'Prediction interval unavailable';
    renderDrivers(ktvCard.querySelector('[data-ktv="drivers"]'), ktv.explanation);
  } else {
    ktvCard.querySelector('[data-ktv="value"]').textContent = '–';
    ktvCard.querySelector('[data-ktv="interval"]').textContent = 'Waiting for prediction';
    renderDrivers(ktvCard.querySelector('[data-ktv="drivers"]'), []);
  }
  const pet = response.pet_class_idx;
  if (pet) {
    const classLabel = typeof pet.pred_class === 'number' ? getPetClassLabel(pet.pred_class) : '–';
    petCard.querySelector('[data-pet="class"]').textContent = classLabel;
    const eceValue = Number.isFinite(pet.ece_bin) ? pet.ece_bin : state.config.calibration?.pet_class_idx?.ece;
    const note = state.config.calibration?.pet_class_idx?.note;
    petCard.querySelector('[data-pet="reliability"]').textContent = buildReliabilityText(eceValue, note);
    renderProbabilities(petCard.querySelector('[data-pet="probabilities"]'), pet.probs, pet.top2);
    renderDrivers(petCard.querySelector('[data-pet="drivers"]'), pet.explanation);
  } else {
    const petCal = state.config.calibration?.pet_class_idx || {};
    petCard.querySelector('[data-pet="class"]').textContent = '–';
    petCard.querySelector('[data-pet="reliability"]').textContent = buildReliabilityText(petCal.ece, petCal.note);
    renderProbabilities(petCard.querySelector('[data-pet="probabilities"]'), []);
    renderDrivers(petCard.querySelector('[data-pet="drivers"]'), []);
  }
}

function renderProbabilities(container, probs = [], highlights = []) {
  if (!container) return;
  if (!Array.isArray(probs) || probs.length === 0) {
    container.innerHTML = '<p class="helper-text">Probability details appear after prediction.</p>';
    return;
  }
  const highlightSet = new Set(highlights || []);
  container.innerHTML = probs
    .map((prob, index) => {
      const percent = Math.round(prob * 100);
      const classes = ['probability-item'];
      if (highlightSet.has(index)) classes.push('highlight');
      return `
      <div class="${classes.join(' ')}">
        <div class="probability-label">
          <span>${getPetClassLabel(index)}</span>
          <span>${formatProbability(prob)}</span>
        </div>
        <div class="probability-bar"><span style="width:${percent}%"></span></div>
      </div>
    `;
    })
    .join('');
}

function buildReliabilityText(ece, note) {
  const parts = [];
  if (Number.isFinite(ece)) {
    parts.push(`ECE ${formatNumber(ece, { digits: 2, fallback: '–' })}`);
  }
  if (note) {
    parts.push(note);
  }
  if (parts.length === 0) {
    return 'Reliability: –';
  }
  return `Reliability: ${parts.join(' · ')}`;
}

function renderDrivers(container, drivers = []) {
  if (!container) return;
  if (!Array.isArray(drivers) || drivers.length === 0) {
    container.innerHTML = '<p class="helper-text">No driver details available.</p>';
    return;
  }
  container.innerHTML = `
    <h3>Top drivers</h3>
    <ul>
      ${drivers
        .map((driver) => `<li>${featureLabel(driver.feature)} (${driver.direction}${formatNumber(driver.weight, { digits: 2 })})</li>`)
        .join('')}
    </ul>
  `;
}

function downloadJson() {
  if (!state.prediction) {
    showToast('Run a prediction first.', 'warning');
    return;
  }
  const payload = {
    version: state.config.version,
    generated_at: new Date().toISOString(),
    request: state.prediction.request,
    response: state.prediction.response
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `detect_pd_prediction_${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function resetForm() {
  state.patient.age = '';
  state.patient.heightValue = '';
  state.patient.weightValue = '';
  state.comorbidities = COMORBIDITY_FIELDS.reduce((acc, item) => ({ ...acc, [item.key]: false }), {});
  state.regimen = {
    modality: '',
    capdBagId: '',
    capdVolumeL: '',
    nipdShifts: [],
    ccpdDayBagId: '',
    ccpdNightShifts: []
  };
  state.times = {
    inflow: state.config.dwell?.defaultInflow || '',
    outflow: state.config.dwell?.defaultOutflow || ''
  };
  LAB_FIELDS.forEach((field) => {
    const defaultUnit = state.config.units?.labs?.[field.key]?.expected || state.config.units?.labs?.[field.key]?.options?.[0];
    state.labs[field.key] = { value: '', unit: defaultUnit };
  });
  state.prediction = null;
  renderPatientPanel();
  renderRegimenPanel();
  renderLabsPanel();
  recomputeDerived();
  renderResults(null);
  document.getElementById('download-button').disabled = true;
  showToast('Form reset.', 'info', 2500);
}

function resetResultCard(ktvCard, petCard) {
  if (ktvCard) {
    ktvCard.querySelector('[data-ktv="value"]').textContent = '–';
    ktvCard.querySelector('[data-ktv="interval"]').textContent = 'Waiting for prediction';
    renderDrivers(ktvCard.querySelector('[data-ktv="drivers"]'), []);
  }
  if (petCard) {
    petCard.querySelector('[data-pet="class"]').textContent = '–';
    const petCal = state.config?.calibration?.pet_class_idx || {};
    petCard.querySelector('[data-pet="reliability"]').textContent = buildReliabilityText(petCal.ece, petCal.note);
    renderProbabilities(petCard.querySelector('[data-pet="probabilities"]'), []);
    renderDrivers(petCard.querySelector('[data-pet="drivers"]'), []);
  }
}
