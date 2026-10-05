export function nearestTick(candidates, target) {
  let best = null;
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate.tick)) continue;
    if (best === null || Math.abs(candidate.tick - target) < Math.abs(best.tick - target)) best = candidate;
  }
  return best;
}

export function evalRatio(evalReward, evalBase) {
  return evalBase ? evalReward / evalBase : null;
}
