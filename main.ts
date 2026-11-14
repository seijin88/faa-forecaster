// polymarket-faa-attribution-02 entry: adaptive quantile forecaster
//
// Strategy:
//   - Recent volatility (mean |change| over last 60 min) sets sigma, with a floor
//   - Recent drift (mean change over last 60 min) decays exponentially over 2 hours
//   - Quantiles: last + drift(h) + sigma * z_tau * sqrt(h), clamped to [0,1]
//   - When prices have been flat (recent vol ~ 0), spread stays narrow
//   - When prices have been moving, spread widens accordingly
//
// Runs offline, Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the content scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854];

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** Mean absolute change over the last N minutes of a price series. */
function recentVol(prices: number[], N: number): number {
  const n = Math.min(N, prices.length - 1);
  if (n <= 0) return 0;
  let sum = 0;
  for (let i = prices.length - n; i < prices.length; i++) {
    sum += Math.abs(prices[i] - prices[i - 1]);
  }
  return sum / n;
}

/** Mean change per minute over the last N minutes (drift). */
function recentDrift(prices: number[], N: number): number {
  const n = Math.min(N, prices.length - 1);
  if (n <= 0) return 0;
  return (prices[prices.length - 1] - prices[prices.length - 1 - n]) / n;
}

/** Forecast one market: horizon rows of five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;

  // Adaptive parameters from recent history
  const vol = recentVol(m.prices, 60);        // mean |change| over last 60 min
  const driftPerMin = recentDrift(m.prices, 60); // mean change/min over last 60 min

  // Sigma: blend recent vol with a floor. vol already captures per-minute move size.
  // Scale up a bit because future vol tends to match recent vol.
  const sigma = Math.max(vol * 1.5, 0.0005);

  // Drift decays exponentially: recent trend matters most for near minutes
  const driftTau = 120; // minutes over which drift decays

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const drift = driftPerMin * h * Math.exp(-h / driftTau);
    const spread = sigma * Math.sqrt(h);
    const row: number[] = [];
    for (const z of Z) {
      row.push(price6(x + drift + z * spread));
    }
    // Ensure non-decreasing (should already be, but clamp for safety)
    for (let j = 1; j < row.length; j++) {
      if (row[j] < row[j - 1]) row[j] = row[j - 1];
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
