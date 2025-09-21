import { showToast } from './toast.js';

export function createInferenceClient(config) {
  const { api } = config;
  async function predict(features) {
    if (api.useMock) {
      return mockPredict(features);
    }
    return await callRemote(api, features);
  }
  return { predict };
}

async function callRemote(api, features) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), api.timeoutMs || 5000);
  try {
    const response = await fetch(`${api.baseUrl}${api.predictEndpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: features.patient_id, features }),
      signal: controller.signal
    });
    if (!response.ok) {
      const message = await safeParseError(response);
      throw new Error(message || `Prediction request failed with ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    showToast(error.message || 'Prediction failed. Check console for details.', 'error');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function safeParseError(response) {
  try {
    const data = await response.json();
    return data?.detail || data?.message;
  } catch (error) {
    return null;
  }
}

function mockPredict(features) {
  const { bmi = 0, pdf_osmolarity = 0, blood_creatinine = 0, blood_albumin = 0, dwell_time_minutes = 0 } = features;
  const ktvBase = 1.2 + (0.002 * (dwell_time_minutes || 0)) / 60 + 0.001 * (pdf_osmolarity - 300) - 0.02 * (bmi - 24);
  const ktv = clamp(round(ktvBase, 2), 0.5, 3.5);
  const ktvResponse = {
    ktv: {
      model: 'mock-catboost',
      prediction: ktv,
      pi_95: [clamp(ktv - 0.25, 0.3, 4.0), clamp(ktv + 0.25, 0.3, 4.0)],
      explanation: buildMockDrivers(features, [
        ['blood_creatinine', -0.12],
        ['pdf_urea', 0.08],
        ['bmi', -0.06]
      ])
    }
  };

  const score =
    -3 +
    0.015 * (blood_creatinine || 0) -
    0.04 * (blood_albumin || 0) +
    0.002 * (dwell_time_minutes || 0) +
    0.003 * (pdf_osmolarity || 0) +
    0.01 * (bmi || 0);

  const probabilities = softmax([score - 1, score, score + 0.5, score + 1.2]);
  const predClass = probabilities.indexOf(Math.max(...probabilities));
  const top2 = findTopIndices(probabilities, 2);
  const petResponse = {
    pet_class_idx: {
      model: 'mock-xgboost',
      pred_class: predClass,
      probs: probabilities.map((p) => round(p, 3)),
      ece_bin: 0.39,
      top2,
      explanation: buildMockDrivers(features, [
        ['blood_creatinine', 0.11],
        ['dwell_time_minutes', 0.09],
        ['pdf_osmolarity', 0.05]
      ])
    }
  };

  return { ...ktvResponse, ...petResponse };
}

function buildMockDrivers(features, template) {
  return template
    .filter(([feature]) => typeof features[feature] !== 'undefined')
    .map(([feature, weight]) => ({
      feature,
      direction: weight >= 0 ? '+' : '-',
      weight: Math.abs(weight)
    }));
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
}

function findTopIndices(values, count) {
  return values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count)
    .map((item) => item.index);
}

function round(value, precision = 2) {
  const factor = 10 ** precision;
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
