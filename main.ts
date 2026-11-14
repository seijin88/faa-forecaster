// polymarket-faa-attribution-02 entry: adaptive quantile forecaster
//
// Strategy:
//   - Volatility (sigma) is an equal-weight (0.5 / 0.5) blend of the 1-hour
//     (60-minute) and 6-hour (360-minute) root-mean-square minute ticks,
//     scaled down by 0.4 and clamped between 0.0003 and 0.005 price units
//     per square-root minute
//   - Center / Median: stays at the market's last observed price (persistence)
//   - Quantiles: last + sigma * z_tau * sqrt(h), clamped to [0,1], where the
//     spread widens with the square root of the horizon minute h
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
/** Per-minute volatility floor of 0.0003, in price units per square-root minute. */
const SIGMA_FLOOR = 0.0003;
/** Per-minute volatility cap of 0.005, in price units per square-root minute. */
const MAX_SIGMA = 0.005;

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

/** Forecast one market: horizon rows of five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const p = m.prices;
  const x = p[p.length - 1] as number;

  // Equal-weight blend of 60-minute and 360-minute RMS ticks, scaled by 0.4.
  const sigma = Math.min(
    MAX_SIGMA,
    Math.max(SIGMA_FLOOR, 0.4 * (0.5 * rmsTick(p, 60) + 0.5 * rmsTick(p, 360))),
  );

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const spread = sigma * Math.sqrt(h);
    const row: number[] = [];
    for (const z of Z) row.push(price6(x + z * spread));
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