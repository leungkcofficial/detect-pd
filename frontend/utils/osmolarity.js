export function calculateOsmolarity(params) {
  const { modality, bagLookup, capdBagId, capdVolumeL, nipdShifts = [], ccpdDayBagId, ccpdNightShifts = [] } = params;
  if (!modality) return { value: null, meta: 'Select modality' };
  const normalizedModality = modality.toUpperCase();
  switch (normalizedModality) {
    case 'CAPD':
      return calcCapd({ bagLookup, bagId: capdBagId, volumeL: capdVolumeL });
    case 'NIPD':
      return calcShifts({ bagLookup, shifts: nipdShifts });
    case 'CCPD':
      if (ccpdDayBagId) {
        return calcCapd({ bagLookup, bagId: ccpdDayBagId });
      }
      return calcShifts({ bagLookup, shifts: ccpdNightShifts });
    default:
      return { value: null, meta: 'Unsupported modality' };
  }
}

function calcCapd({ bagLookup, bagId, volumeL }) {
  if (!bagId) {
    return { value: null, meta: 'Select bag' };
  }
  const bag = bagLookup.get(bagId);
  if (!bag) {
    return { value: null, meta: 'Unknown bag' };
  }
  const osm = bag.osm_mOsmL;
  if (!isFinite(osm)) {
    return { value: null, meta: 'Bag missing osmolarity' };
  }
  return {
    value: Number(osm),
    meta: `${bag.brand} ${bag.concentration}${volumeL ? ` @ ${volumeL} L` : ''}`
  };
}

function calcShifts({ bagLookup, shifts }) {
  if (!Array.isArray(shifts)) {
    return { value: null, meta: 'Configure shifts' };
  }
  let totalWeighted = 0;
  let totalVolume = 0;
  for (const shift of shifts) {
    if (!shift || !shift.bagId) continue;
    const bag = bagLookup.get(shift.bagId);
    if (!bag || !isFinite(bag.osm_mOsmL)) continue;
    const volume = parseFloat(shift.volumeL);
    const count = parseInt(shift.count, 10);
    if (!isFinite(volume) || volume <= 0 || !isFinite(count) || count <= 0) continue;
    const total = volume * count;
    totalVolume += total;
    totalWeighted += bag.osm_mOsmL * total;
  }
  if (totalVolume === 0) {
    return { value: null, meta: 'Add night cycles' };
  }
  return {
    value: totalWeighted / totalVolume,
    meta: `Volume-weighted mean across ${totalVolume.toFixed(1)} L`
  };
}
