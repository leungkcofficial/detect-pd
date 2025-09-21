export async function loadConfig() {
  const response = await fetch('./config/frontend_config.json', { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error(`Failed to load configuration: ${response.status}`);
  }
  const data = await response.json();
  validateConfig(data);
  return freezeConfig(data);
}

function validateConfig(config) {
  if (!config || typeof config !== 'object') {
    throw new Error('Configuration payload malformed.');
  }
  if (!Array.isArray(config.bags) || config.bags.length === 0) {
    throw new Error('Configuration missing peritoneal dialysate bag dictionary.');
  }
  const requiredApiFields = ['baseUrl', 'predictEndpoint', 'timeoutMs'];
  for (const field of requiredApiFields) {
    if (!config.api || typeof config.api[field] === 'undefined') {
      throw new Error(`Configuration missing api.${field}`);
    }
  }
}

function freezeConfig(config) {
  return {
    ...config,
    api: Object.freeze({ ...config.api }),
    bags: config.bags.map((bag) => Object.freeze({ ...bag })),
    units: deepFreeze(config.units || {}),
    guards: deepFreeze(config.guards || {}),
    calibration: deepFreeze(config.calibration || {}),
    dwell: deepFreeze(config.dwell || {})
  };
}

function deepFreeze(target) {
  if (!target || typeof target !== 'object') {
    return target;
  }
  Object.freeze(target);
  for (const key of Object.keys(target)) {
    deepFreeze(target[key]);
  }
  return target;
}
