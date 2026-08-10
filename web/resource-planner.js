"use strict";

export function estimateSimulationResources({ periods, paths, assets }) {
  const safePeriods = Math.max(1, Number(periods) || 1);
  const safePaths = Math.max(1, Number(paths) || 1);
  const safeAssets = Math.max(1, Number(assets) || 1);
  const targetChunk = Math.max(500, Math.round((5000 * 120 * 8) / (safePeriods * safeAssets)));
  const chunkSize = Math.min(5000, targetChunk, safePaths);
  return {
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
