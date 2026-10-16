const $ = (id) => document.getElementById(id);
const regimes = [
  { key: 'high_growth_low_inflation', value: 'High growth / low inflation', name: 'Goldilocks', color: '#43856a', growth: 'Higher', inflation: 'Lower', text: 'Growth is above its threshold while inflation is below it. Explore a disinflationary expansion as the opening state of your scenario.' },
  { key: 'high_growth_high_inflation', value: 'High growth / high inflation', name: 'Overheating', color: '#b8873c', growth: 'Higher', inflation: 'Higher', text: 'Growth and inflation are both above their thresholds. Explore an inflationary expansion as the opening state of your scenario.' },
  { key: 'low_growth_high_inflation', value: 'Low growth / high inflation', name: 'Stagflation', color: '#b97761', growth: 'Lower', inflation: 'Higher', text: 'Growth is below its threshold while inflation is above it. Explore an inflationary slowdown as the opening state of your scenario.' },
  { key: 'low_growth_low_inflation', value: 'Low growth / low inflation', name: 'Recession', color: '#628da4', growth: 'Lower', inflation: 'Lower', text: 'Growth and inflation are both below their thresholds. Explore a deflationary slowdown as the opening state of your scenario.' },
];

export function refreshScenarioPreview() {
  const months = Number($('periods').value);
  const paths = Number($('paths').value);
  const selected = $('start-state').value;
  const regime = regimes.find((item) => item.value === selected);
  $('preview-horizon').textContent = months > 0 ? `${+(months / 12).toFixed(1)} ${months === 12 ? 'year' : 'years'}` : 'Set horizon';
  $('preview-paths').textContent = paths > 0 ? paths.toLocaleString() : 'Set paths';
  const isHMM = $('model-kind').value === 'hmm';
  document.querySelector('[data-preview-control="start-state"], [data-preview-control="model-kind"]').dataset.previewControl = isHMM ? 'model-kind' : 'start-state';
  $('preview-regime').textContent = isHMM ? 'Learned from returns' : regime?.name || $('start-state').selectedOptions[0]?.textContent || 'Latest inferred';
  $('hero-path-count').textContent = Math.max(0, paths || 0).toLocaleString();
}

export function refreshAllocationPreview(assets) {
  const bar = $('allocation-composition');
  const total = assets.reduce((sum, asset) => sum + Math.max(0, asset.weight), 0);
  const existing = new Map([...bar.children].map((child) => [child.dataset.ticker, child]));
  assets.forEach(({ ticker, weight, color }) => {
    let segment = existing.get(ticker);
    if (!segment) {
      segment = document.createElement('span');
      segment.dataset.ticker = ticker;
      bar.appendChild(segment);
    }
    existing.delete(ticker);
    const share = total > 0 ? Math.max(0, weight) / total * 100 : 0;
    segment.style.width = `${share}%`;
    segment.style.backgroundColor = color;
    segment.title = `${ticker}: ${share.toFixed(1)}% normalized allocation`;
  });
  existing.forEach((segment) => segment.remove());
  bar.setAttribute('aria-label', total > 0
    ? `Normalized allocation: ${assets.map((asset) => `${asset.ticker} ${(Math.max(0, asset.weight) / total * 100).toFixed(1)}%`).join(', ')}`
    : 'No allocation selected');
  bar.classList.toggle('is-empty', total <= 0);
}

export function setupDynamicExperience() {
  const stage = $('quadrant-stage');
  const cards = [...stage.querySelectorAll('[data-regime]')];
  const tour = $('regime-tour');
  const apply = $('apply-regime');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let index = 0;
  let timer = null;
  const syncApply = () => {
    const available = $('model-kind').value === 'quadrant' && [...$('start-state').options].some((option) => option.value === regimes[index].value);
    const applied = available && $('start-state').value === regimes[index].value;
    apply.disabled = !available || applied;
    apply.textContent = applied ? 'Selected as starting state ✓' : available ? 'Use as starting state →' : 'Available with quadrant model';
  };
  const selectRegime = (next) => {
    index = next;
    const regime = regimes[index];
    stage.style.setProperty('--map-color', regime.color);
    cards.forEach((card) => card.setAttribute('aria-pressed', String(card.dataset.regime === regime.key)));
    $('regime-insight-title').textContent = regime.name;
    $('regime-insight-copy').textContent = regime.text;
    $('regime-step').textContent = `0${index + 1} / 04`;
    ['growth', 'inflation'].forEach((dimension) => {
      $(`explorer-${dimension}`).textContent = regime[dimension];
      $(`explorer-${dimension}-bar`).style.width = regime[dimension] === 'Higher' ? '80%' : '25%';
    });
    if (!reducedMotion.matches) {
      $('regime-insight').getAnimations().forEach((animation) => animation.cancel());
      $('regime-insight').animate([{ opacity: .35, transform: 'translateY(7px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 320, easing: 'ease-out' });
    }
    syncApply();
  };
  const stopTour = () => {
    clearInterval(timer);
    timer = null;
    stage.classList.remove('is-touring');
    tour.textContent = '▶ Play tour';
    tour.setAttribute('aria-pressed', 'false');
    $('regime-insight').setAttribute('aria-live', 'polite');
  };
  cards.forEach((card) => card.addEventListener('click', () => {
    stopTour();
    selectRegime(regimes.findIndex((regime) => regime.key === card.dataset.regime));
  }));
  tour.addEventListener('click', () => {
    if (timer) return stopTour();
    stage.classList.add('is-touring');
    tour.textContent = 'Ⅱ Pause tour';
    tour.setAttribute('aria-pressed', 'true');
    // A tour previews categories, never probabilities or simulated transitions.
    $('regime-insight').setAttribute('aria-live', 'off');
    timer = setInterval(() => selectRegime((index + 1) % regimes.length), 4000);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopTour(); });
  reducedMotion.addEventListener('change', stopTour);
  new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) stopTour(); }).observe(stage);
  apply.addEventListener('click', () => {
    stopTour();
    $('start-state').value = regimes[index].value;
    $('start-state').dispatchEvent(new Event('change', { bubbles: true }));
    syncApply();
  });
  new MutationObserver(syncApply).observe($('start-state'), { childList: true });
  document.addEventListener('change', syncApply);
  document.querySelectorAll('[data-preview-control]').forEach((button) => {
    button.addEventListener('click', () => {
      const control = $(button.dataset.previewControl);
      control.closest('details').open = true;
      control.focus({ preventScroll: true });
      control.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'center' });
    });
  });
  selectRegime(0);
  refreshScenarioPreview();
}
