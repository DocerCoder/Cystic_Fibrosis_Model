const fs = require('fs');
const path = require('path');

const DATA_DIR = __dirname;
const INPUT_PATH = path.join(DATA_DIR, 'cftr_batch_results_merged.json');

let raw;
try {
  raw = JSON.parse(fs.readFileSync(INPUT_PATH, 'utf8'));
} catch (err) {
  console.error('Failed to read or parse', INPUT_PATH, err.message);
  process.exit(1);
}

// Flatten to variant objects
function asVariants(x) {
  if (Array.isArray(x)) return x.flat(Infinity);
  if (x && typeof x === 'object') return Object.values(x).flat(Infinity);
  return [];
}

const variants = asVariants(raw).filter(v => v && typeof v === 'object');
console.log('Detected variants:', variants.length);

// --- Class defaults for priors ---
function classDefaults(predictedClass) {
  const c = String(predictedClass || '').toUpperCase();
  if (c.includes('CLASS I')) return { therapy: 'ETI', fev1: 5, bmi: 0.5 };
  if (c.includes('CLASS II')) return { therapy: 'ETI', fev1: 12, bmi: 1.2 };
  if (c.includes('CLASS III')) return { therapy: 'Ivacaftor', fev1: 8, bmi: 0.8 };
  if (c.includes('CLASS IV')) return { therapy: 'Tez/Iva', fev1: 6, bmi: 0.6 };
  if (c.includes('CLASS V')) return { therapy: 'Tez/Iva', fev1: 6, bmi: 0.6 };
  return { therapy: 'Tez/Iva', fev1: 6, bmi: 0.6 };
}

const cftrOut = {};
const organoidOut = { priors: {} };

for (const v of variants) {
  const id = (v.variant || v.hgvs || v.id || '').toString().trim();
  if (!id) continue;

  // Skip if we already have a Strong record
  if (cftrOut[id] && cftrOut[id].evidence === 'Strong') continue;

  // --- Functional summary mapping ---
  let gating = 0, cond = 0, proc = 0;
  if (v.functional_summary && typeof v.functional_summary === 'object') {
    gating = Number(v.functional_summary.avg_function ?? 0);
    cond   = Number(v.functional_summary.avg_function ?? 0);
    proc   = (v.functional_summary.avg_quantity != null)
      ? Number(v.functional_summary.avg_quantity) / 100
      : 0;
  }

  const evidence = v.clinical_context ? 'Strong' : 'Heuristic';

  // Only overwrite if we don’t have this variant yet,
  // or if this record has stronger evidence
  if (!cftrOut[id] || evidence === 'Strong') {
    cftrOut[id] = {
      gating, cond, proc,
      evidence,
      source: 'cftr_batch_results_merged',
      last_update: new Date().toISOString().slice(0,10)
    };

    const anchor = classDefaults(v.predicted_class);
    organoidOut.priors[id] = {
      [anchor.therapy]: {
        effect_fev1_gain_mean: anchor.fev1,
        effect_fev1_gain_sd: Math.round(anchor.fev1 * 0.25),
        effect_bmi_gain_mean: anchor.bmi,
        effect_bmi_gain_sd: 0.3,
        assay: null,
        evidence,
        last_update: new Date().toISOString().slice(0,10)
      }
    };
  }
}

// --- Write outputs ---
const OUT_CFTR = path.join(DATA_DIR, 'cftr_function_scores.json');
const OUT_ORG  = path.join(DATA_DIR, 'organoid_priors.json');
fs.writeFileSync(OUT_CFTR, JSON.stringify(cftrOut, null, 2));
fs.writeFileSync(OUT_ORG, JSON.stringify(organoidOut, null, 2));
console.log('Wrote', OUT_CFTR, 'and', OUT_ORG);
