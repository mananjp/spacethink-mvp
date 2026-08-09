# SPEC — SBC/PPC must consult the scorer

**Status:** proposed (draft PR, for review before merge)
**Stacked on:** `feat/calibration-gated-autonomy` (PR #3)
**Touches the patentable core** — the calibration gate's semantics. Review carefully.

## Problem

PR #3 wired the AutonomyGate to a *real* `CalibrationStatus` derived from
`run_sbc` / `run_ppc` instead of a hardcoded confidence constant. But those two
functions had a latent defect:

> `run_sbc(scorer, …)` and `run_ppc(scorer, …)` accepted a `scorer` argument and
> **never used it.**

Verified mechanically: the token `scorer` appeared only in each function's
signature and docstring, never in its body. SBC computed rank-uniformity from a
hardcoded normalized-RMSE distance between `ToySimulator` runs; PPC checked real
summary stats against the twin ensemble's band. **Both measured the twin + a fixed
metric, not the learned scorer.**

### Why it mattered

The roadmap milestone PR #3 set up — *"train the SBIScorer until real SBC passes,
then flip the default gate to PASSIVE"* — was **unachievable**. Training the scorer
changed nothing SBC/PPC looked at, so the gate could never flip no matter how good
the scorer got. The claim language in the patent doc ("a calibrated confidence
value derived from the scorer's calibration diagnostics") was not true of the code.

## Fix

Make both checks a function of the scorer, respecting the existing `Scorer`
protocol (`score(hyp, real, simulated) -> SimResult`, lower `distance` = better fit):

- **`run_sbc`** — for each prior draw θ\*, simulate `real ~ p(x|θ*)`, then rank θ\*
  among candidate draws θⱼ **by the scorer's `distance`** (`scorer.score(hypⱼ, real,
  [simⱼ]).distance`). Uniform ranks ⇒ well-calibrated scorer ⇒ pass. Same simulation
  budget as before; only the ranking metric changed (distance is now asked of the
  scorer instead of computed inline).

- **`run_ppc`** — form the predictive band from the scorer's **importance-weighted**
  ensemble: draw θⱼ from the prior, weight each by `exp(-distance)` (self-normalized,
  matching `normalize_posteriors`), and check the real summary stats against the
  weighted 95% band. A scorer that concentrates weight on well-fitting parameters
  yields a tight, well-centered band.

`derive_calibration_status` is unchanged — it still reduces `(SBCResult, PPCResult)`
to a `CalibrationStatus`. Only the two producers changed.

### Secondary changes
- `duration_s` is now a parameter of both functions (was a magic `1000` / `2000`),
  so callers and tests can run cheaply.
- `diagnostics["scorer"]` records which scorer produced the result.

## Backward compatibility

- All callers (`reaction_wheel.py`, `tests/test_phases.py`) use keyword args; the new
  parameters have defaults. No caller changes required.
- Full suite: **63 passed** (was 59; +4 new tests), ~6s. No regressions.
- The default reaction-wheel calibration still **fails** honestly under the corrected
  SBC (`passed=False`, `p=0.0`, rank histogram `[12,0,0,0,0,0,0]`), so the gate stays
  **ACTIVE** — behavior unchanged for the shipped default.

## What the corrected SBC reveals (important)

`DistanceScorer` fails SBC because it behaves like a point estimate: the true
parameter's twin always fits `real` best, so θ\*'s rank collapses to 0 every time —
a spiked, non-uniform histogram. **This is SBC working as designed: it rejects
overconfident "posteriors" with no uncertainty.** Passing SBC therefore requires a
scorer that expresses genuine posterior *uncertainty* — i.e. the amortized NPE
`SBIScorer` (`sbi` package), trained on adequate hardware.

## Deferred (not in this PR)

1. **Train the SBIScorer** (`scripts/train_sbi_scorer.py`, needs `pip install sbi` +
   compute) so its posterior has calibrated width, then re-run SBC on it.
2. **Flip the default gate to PASSIVE** only once real SBC passes on the trained
   scorer — never before. No change here fakes that flip.
3. **Native posterior-predictive PPC**: if a scorer exposes a posterior sampler,
   replace the prior-proposal importance sampling with true posterior draws.

## Open question for review

Importance-weighted PPC from a prior proposal is the pragmatic, protocol-respecting
form. Is that the semantics we want to commit to for the patent claim, or should PPC
require a native posterior sampler (and thus be defined only for `SBIScorer`)? This
is a claim-scope decision, not just an implementation detail.
