/**
 * microbiome.js
 * Calculates microbiome penalty for CF severity scoring.
 */

function microbiomePenalty(microbiome) {
  const rules = {
    diversityThreshold: 2.0,
    lowDiversityPenalty: 6,
    proteobacteriaPenalty: 8,
    bothBonus: 4
  };
  if (
    microbiome.shannon_diversity == null ||
    microbiome.dominant_phylum == null ||
    microbiome.dominant_phylum === ""
  ) {
    return 0;
  }
  const diversity = parseFloat(microbiome.shannon_diversity);
  const phylum = microbiome.dominant_phylum.trim().toLowerCase();
  const isLowDiversity = diversity < rules.diversityThreshold;
  const isProteobacteria = phylum === "proteobacteria";

  let penalty = 0;
  if (isLowDiversity) penalty += rules.lowDiversityPenalty;
  if (isProteobacteria) penalty += rules.proteobacteriaPenalty;
  if (isLowDiversity && isProteobacteria) penalty += rules.bothBonus;
  return penalty; // 0–18
}
function readMicrobiomeInputs() {
  const diversityStr = val('shannon_diversity');
  const diversity = diversityStr === "" ? null : parseFloat(diversityStr);
  const phylum = val('dominant_phylum');
  return {
    shannon_diversity: Number.isFinite(diversity) ? diversity : null,
    dominant_phylum: phylum || null
  };
}