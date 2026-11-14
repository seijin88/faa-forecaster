// polymarket-faa-attribution-02 entry: adaptive quantile forecaster
//
// Strategy:
//   - Volatility is the root-mean-square of recent minute ticks, blended
//     across a 1-hour, 6-hour and full-history window (recent ticks weigh
//     most), with a floor so a quiet market keeps a tight but honest spread
//   - Drift is the recent mean change per minute, capped so one printed
//     jump cannot drag the median across the whole horizon, decaying
//     exponentially over the horizon
//   - Quantiles: last + drift(h) + sigma * z_tau * sqrt(h), clamped to [0,1]
//
// Runs offline, Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854];
/** Per-minute volatility floor, in price units per square-root minute. */
const SIGMA_FLOOR = 0.0005;
/** Cap on the assumed trend, in price units per minute. */
const MAX_DRIFT_PER_MIN = 0.002;
/** Minutes over which the trend decays across the horizon. */
const DRIFT_TAU = 100;

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

/** Root-mean-square of the last n minute-to-minute changes. */
function rmsTick(prices: number[], n: number): number {
  const k = Math.min(n, prices.length - 1);
  if (k <= 0) return 0;
  let sum = 0;
  for (let i = prices.length - k; i < prices.length; i++) {
    const d = prices[i] - prices[i - 1];
    sum += d * d;
  }
  return Math.sqrt(sum / k);
}

/** Mean change per minute over the last n minutes. */
function meanTick(prices: number[], n: number): number {
  const k = Math.min(n, prices.length - 1);
  if (k <= 0) return 0;
  return (prices[prices.length - 1] - prices[prices.length - 1 - k]) / k;
}

/** Share of the last n minutes whose price changed at all. */
function tickRate(prices: number[], n: number): number {
  const k = Math.min(n, prices.length - 1);
  if (k <= 0) return 0;
  let hits = 0;
  for (let i = prices.length - k; i < prices.length; i++) {
    if (prices[i] !== prices[i - 1]) hits++;
  }
  return hits / k;
}

/** Forecast one market: horizon rows of five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const p = m.prices;
  const x = p[p.length - 1] as number;

  // Volatility clustering: a market ticking more than its own base rate runs
  // hot and widens; one gone quiet against its history cools and narrows.
  // Clamped so a single silent hour on an old market cannot collapse the spread.
  const heat = Math.max(0.6, Math.min(1.5, (tickRate(p, 120) + 0.03) / (tickRate(p, p.length) + 0.03)));
  // Recent ticks weigh most; the full history keeps short samples honest.
  const sigma = Math.max(
    (0.5 * rmsTick(p, 60) + 0.3 * rmsTick(p, 360) + 0.2 * rmsTick(p, p.length)) * heat,
    SIGMA_FLOOR,
  );

  // Trend: mostly the last hour, backed up by the last four, hard-capped.
  const drift = Math.max(
    -MAX_DRIFT_PER_MIN,
    Math.min(MAX_DRIFT_PER_MIN, 0.6 * meanTick(p, 60) + 0.4 * meanTick(p, 240)),
  );

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const trend = drift * h * Math.exp(-h / DRIFT_TAU);
    const spread = sigma * Math.sqrt(h);
    const row: number[] = [];
    for (const z of Z) row.push(price6(x + trend + z * spread));
    // Quantiles must never decrease across the row.
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