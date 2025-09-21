export function formatNumber(value, { digits = 2, fallback = '–' } = {}) {
  if (value === null || typeof value === 'undefined' || Number.isNaN(value)) {
    return fallback;
  }
  return Number(value).toFixed(digits);
}

export function formatInteger(value, { fallback = '–' } = {}) {
  if (!Number.isFinite(value)) return fallback;
  return Math.round(value).toString();
}

export function formatProbability(prob, { fallback = '0.0%' } = {}) {
  if (!Number.isFinite(prob)) return fallback;
  return `${(prob * 100).toFixed(1)}%`;
}

export function titleCase(text) {
  if (!text) return '';
  return text
    .toLowerCase()
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function featureLabel(feature) {
  return feature
    .replace(/_/g, ' ')
    .replace(/\b[a-z]/g, (c) => c.toUpperCase());
}
