import { microbiomePenalty } from './microbiome.js';

export function computeMicrobiomeScore() {
  const diversityInput = document.getElementById('shannon_diversity');
  const phylumInput = document.getElementById('dominant_phylum');

  const diversity = parseFloat(diversityInput.value);
  const phylum = phylumInput.value;

  const microbiomeData = {
    shannon_diversity: isNaN(diversity) ? null : diversity,
    dominant_phylum: phylum || null
  };

  const score = microbiomePenalty(microbiomeData);

  // Update score display
  const scoreDisplay = document.getElementById('microbiome_score');
  if (scoreDisplay) {
    scoreDisplay.textContent = `Microbiome Score: ${score}/18`;
  }

  // Update risk badges
  const badges = [];
  if (microbiomeData.shannon_diversity < 2.0) {
    badges.push("Low diversity ↑ severity");
  }
  if ((microbiomeData.dominant_phylum || "").toLowerCase() === "proteobacteria") {
    badges.push("Proteobacteria dominance ↑ inflammation");
  }

  const badgeContainer = document.getElementById('microbiome_badges');
  if (badgeContainer) {
    badgeContainer.innerHTML = badges.map(b => `<span class="badge">${b}</span>`).join('');
  }

  return score;
}
