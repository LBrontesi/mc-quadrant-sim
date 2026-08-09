"use strict";

export const MEMORY_LIMIT_MB = 384;

export function estimateSimulationResources({ periods, paths, assets, workers, jointMacro = false, dynamicCorrelation = false }) {
  const safePeriods = Math.max(1, Number(periods) || 1);
  const safePaths = Math.max(1, Number(paths) || 1);
  const safeAssets = Math.max(1, Number(assets) || 1);
  const safeWorkers = Math.max(1, Number(workers) || 1);
  const targetChunk = Math.max(500, Math.round((5000 * 120 * 8) / (safePeriods * safeAssets)));
  const chunkSize = Math.min(5000, targetChunk, safePaths);
  const wealthBytes = safePeriods * safePaths * 8;
  const regimeBytes = safePeriods * safePaths;
  const responseBytes = safePaths * 8 * 2;
  const macroPathBytes = jointMacro ? safePeriods * safePaths * 2 * 8 : 0;
  let transientPerWorker = safePeriods * chunkSize * safeAssets * 8 * 3;
  if (jointMacro) transientPerWorker += safePeriods * chunkSize * 2 * 8;
  if (dynamicCorrelation) transientPerWorker += chunkSize * safeAssets * safeAssets * 8 * 3;
  const transientBytes = transientPerWorker * safeWorkers;
  const workerOverhead = safeWorkers * 32 * 1024 ** 2;
  const fixedOverhead = 64 * 1024 ** 2;
  const memoryMb = (wealthBytes + regimeBytes + responseBytes + macroPathBytes + transientBytes + workerOverhead + fixedOverhead) / 1024 ** 2;
  const ratio = memoryMb / MEMORY_LIMIT_MB;
  return {
    memoryMb,
    ratio,
    level: ratio > 1 ? "over" : ratio >= 0.75 ? "warn" : "good",
    workUnits: safePeriods * safePaths * safeAssets,
    chunkSize,
  };
}

export function formatWorkUnits(value) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(Math.round(value));
}
