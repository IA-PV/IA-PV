# Genetic agent — hyperparameter tuning protocol

This document is the trustworthy way to tune the genetic agent. It exists
because the naive loop — "change a knob, look at `best_fitness`, keep the
bigger number" — is misleading here for three measured reasons:

1. **Censored selection metric.** `task_return` is (weighted) lines cleared
   over a *finite* horizon. Once a policy reliably survives the horizon, its
   return is pinned near a throughput ceiling (~0.40 lines/piece), so a wide
   band of very different policies score almost the same. Empirically, a full
   Dellacherie policy and one that *ignores* holes/transitions/wells were
   statistically indistinguishable up to 600 pieces (both survived every seed).
2. **Winner's curse.** `best_fitness` is a `max` over ~12–16 candidates scored
   on the *same* validation seeds, so it is optimistically biased. Comparing two
   configurations by their `best_fitness` compounds the bias.
3. **Flat landscape vs premature convergence.** An early plateau in the search
   can mean either the landscape is genuinely flat (population still diverse) or
   the population collapsed. The two demand opposite responses.

The trainer now has guard-rails for each; use them as follows.

## Guard-rails (in the trainer/report)

- **Held-out TEST bank** (`--test-episodes`, `--test-max-pieces`). Disjoint
  seeds, placed after the monitoring batch, used **once** to score the
  already-selected model. It never influences selection, so it is unbiased.
  Enabled by default (`--test-episodes 24`); `--test-episodes 0` restores the
  old behaviour bit-for-bit. Surfaced in `summary.json → test_evaluation` with a
  95% CI, and in `model.json` (`test_*`).
- **Discrimination metrics** in `summary.json → validation_discrimination` and
  inside `test_evaluation.discrimination`: `reward_per_piece` (removes the
  horizon-length confound that flattens `task_return`), `score_per_line`
  (clearing STYLE: ≈100 for singles, up to ≈300 for back-to-back tetrises),
  `lines_per_piece`, `score_per_piece`.
- **Population diversity** per generation in `history.csv` +
  `summary.json → convergence`, read against the **fixed-seed monitoring**
  curve (comparable across generations, unlike the rotating-seed `best_fitness`).

## Methodology (non-negotiables)

- **Rank on the shared TEST bank, not on `best_fitness`.** Use
  `compare_genetic_runs` so every model is scored on the *same* fixed seeds
  (base `200000`, far from any training `--seed`) at the decensored horizon.
- **Report CIs; require separation.** Do not declare a winner unless the paired
  common-random-number CI excludes 0.
- **Compare at a long horizon** (≥1000 pieces), never at the censored 300.
- **Repeat over ≥2 training seeds** per configuration (e.g. `--seed 0`, then
  `--seed 1000`) and require the winner to hold across seeds.
- **Freeze the four hold/I genes only after re-validating at the long horizon**
  (see below) — those genes are the tetris machinery whose value only appears
  once the metric is decensored.

## The first experiment — 2×2 (freeze × horizon)

This tests the reviewer's hypothesis (raise the horizon) *and* the concern that
freezing the hold/I genes caps the ceiling, in one shot. Keep `--freeze-genes`
and `--validation-max-pieces` as the only variables.

```bash
FROZEN="use_hold hold_store_i hold_retrieve_i i_well_match"
COMMON="--population-size 24 --generations 15 --episodes-per-individual 6 \
  --monitoring-episodes 8 --validation-episodes 24 \
  --elite-count 2 --elite-archive-capacity 4 \
  --lookahead-depth 2 --lookahead-beam-width 3 \
  --mutation-stddev 0.30 --final-mutation-stddev 0.06 --mutation-schedule exponential \
  --test-episodes 30 --workers 0"

# horizon 300, frozen vs free
python -m tetris_ai.cli.train_genetic_agent $COMMON --max-pieces 300 --validation-max-pieces 300 --freeze-genes $FROZEN --seed 0
python -m tetris_ai.cli.train_genetic_agent $COMMON --max-pieces 300 --validation-max-pieces 300 --seed 0
# horizon 900, frozen vs free
python -m tetris_ai.cli.train_genetic_agent $COMMON --max-pieces 300 --validation-max-pieces 900 --freeze-genes $FROZEN --seed 0
python -m tetris_ai.cli.train_genetic_agent $COMMON --max-pieces 300 --validation-max-pieces 900 --seed 0
```

Then rank all four on the **same** held-out bank at a long horizon:

```bash
python -m tetris_ai.cli.compare_genetic_runs \
  --models reports/genetic_agent/<RUN_1>/model.json \
           reports/genetic_agent/<RUN_2>/model.json \
           reports/genetic_agent/<RUN_3>/model.json \
           reports/genetic_agent/<RUN_4>/model.json \
  --labels frozen@300 free@300 frozen@900 free@900 \
  --episodes 40 --max-pieces 1000 --seed 200000 \
  --json-out reports/genetic_agent/compare_2x2.json
```

Read it as:
- **`reward_per_piece` and `score_per_line`**, not raw `task_return`. If frozen
  and free tie on `task_return` but free wins on `score_per_line`, the hold/I
  genes are buying tetrises — do **not** freeze.
- **Paired CI** decides significance. Overlapping-with-0 ⇒ the gap is noise;
  widen episodes or accept "no difference".

## Diagnosing a plateau

```bash
python -m tetris_ai.cli.diagnose_genetic_run --report-dir reports/genetic_agent/<RUN_ID>
```

- **Flat landscape** (early plateau, diversity still high) ⇒ more
  population/generations won't help; give the metric gradient (longer horizon,
  rank on `reward_per_piece`/`score_per_line`).
- **Premature convergence** (diversity collapsed) ⇒ raise `--mutation-stddev`
  / `--final-mutation-stddev` / `--population-size`, or slow the annealing.

Only after this do the population/generation/beam sweeps (`compare_genetic_runs`
between each), one small grid at a time, always ranked on the shared TEST bank
with CIs.
