# SPEC — Make the Calibration Gate Real and Wired

**Problem.** The patentable core (Verifier → CalibrationStatus → AutonomyGate) exists,
but two things make it a facade rather than a working safeguard:

1. `calibration_status()` in both verifiers returns a **hardcoded constructor constant**
   (`confidence=0.92` / `0.94`, `passed=True`). It never runs the SBC/PPC code that
   already lives in `evaluate/calibration.py`. The patent's central novelty —
   *"a calibrated confidence value derived from calibration diagnostics, **not** the
   scorer's self-reported confidence"* — is therefore **not true of the code**.
2. `AutonomyGate` / `decide_oversight` is **never called from the live loop**
   (`run_closed_loop` escalates only via the old `evaluate_human_gate` posterior check).
   The flagship architecture is tested in isolation but does not run.

**Goal.** Derive `CalibrationStatus.confidence` from real SBC/PPC (reaction wheel) and a
real self-benchmark (astro), and route every `run_closed_loop` event through
`decide_oversight`, **without breaking** the existing human-gate semantics.

## Contracts

- `derive_calibration_status(domain, sbc: SBCResult, ppc: PPCResult, method) -> CalibrationStatus`
  - `passed = sbc.passed and ppc.passed`
  - `confidence = 0.5 * ppc_coverage + 0.5 * sbc_uniformity_score` where
    `sbc_uniformity_score = min(1.0, uniformity_p_value / 0.05)` — bounded [0,1], derived
    purely from diagnostics, never self-reported.
  - `diagnostics` carries the raw SBC/PPC numbers for audit.
- Verifiers: an **injected** `confidence=`/`calibrated=` still wins (keeps the existing
  autonomy-gate unit tests, which inject known statuses). When nothing is injected, the
  verifier **computes the real status** and caches it (`lru_cache`, deterministic seed).
- `run_closed_loop(..., oversight_policy=None, calibration_status=None)`:
  - computes the run-level `CalibrationStatus` once (injected or cached default),
  - calls `decide_oversight` per event,
  - **adds** `autonomy_mode`, `calibrated_confidence`, `calibration_passed`,
    `calibration_method`, and a combined `requires_human` flag to each report event,
  - leaves `validation_status` (human_gate) untouched — additive, non-breaking.

## Edge cases / constraints
- Empty PPC coverage → confidence 0.0, passed False (fail closed).
- Calibration is a **family-level** property (the SBC code is twin-driven, scorer-agnostic),
  so it is computed once and cached, not per event — keeps the suite fast.
- Immutable `CalibrationStatus` (frozen dataclass) — never mutated.

## Honest result this surfaces
Real SBC on the current default scorer **fails** (`p≈0`, ranks cluster at 0): a
distance/SBI scorer with the true parameter fits best, so it is not a rank-uniform
posterior. The gate therefore correctly returns **ACTIVE (human required)** by default —
the hardcoded `0.92/passed=True` was masking this. This is the gate doing its job, and it
is the strongest demonstration that it does something. It also sets the bar for "train the
SBI scorer until it passes SBC" as the concrete next milestone.
