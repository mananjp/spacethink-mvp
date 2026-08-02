# Fix notes — diagnosis engine now discriminates (and an evaluation harness)

Short version: the MVP's architecture was right and the pipeline ran, but the
diagnosis step didn't actually work — it couldn't tell the fault types apart and
confidently "diagnosed" faults on healthy telemetry. This change fixes that and
adds an honest accuracy measurement.

## What was wrong (v0)

Running the loop on each known fault and looking at the verdict showed:

| Ground truth | v0 diagnosis | |
|---|---|---|
| friction increase | bearing_friction_increase | ✅ |
| encoder dropout | encoder_dropout (barely) | ⚠️ |
| stiction | encoder_dropout | ❌ |
| **nominal (no fault)** | **encoder_dropout, "conf" 0.50** | ❌❌ |

Two root causes, both in how `evaluate/scorer.py` fed into `plan/planner.py`:

1. **The scorer ignored the detected event.** It compared each hypothesis's
   simulation against the *entire* telemetry series, not the anomaly window the
   detector found — so every event in a run got an identical posterior, and the
   detect→hypothesize→test steps weren't actually connected.
2. **The distance was noise-dominated.** It compared raw, differently-seeded
   telemetry, so normalized RMSE was ≈1 from noise alone and the tiny fault
   signature barely moved it. (Compounded by a data bug: the generator's speed
   noise was ~200× the twin's, so the two weren't even on the same scale.)

The LLM council and human gate sat on top of these meaningless numbers, so their
"consensus" was rigor-theater over a coin flip.

## What changed

- **`evaluate/signature.py` (new):** reduces a telemetry window (vs. a pre-fault
  baseline) to a physically meaningful fault *signature* — per-channel level
  shift and trend, plus dropout-zero fraction, stiction step-down rate, and
  current-spike rate. This is the missing link between detection and diagnosis.
- **`evaluate/scorer.py`:** `SignatureScorer` compares the real signature to each
  hypothesis's twin-simulated signature (Euclidean distance → softmax posterior).
- **`plan/planner.py`:** extracts the real signature the *same symmetric way* as
  the simulated ones, simulates each candidate, and ranks by signature distance.
- **`hypothesize/generator.py`:** added a `nominal_no_fault` candidate so the loop
  can conclude "benign / sensor noise" instead of being forced to name a fault.
- **`ingest/synthetic_generator.py` + `twin/simulator.py`:** fixed the noise
  mismatch so real and simulated telemetry are on the same scale.
- **`explore/detector.py`:** events now carry their `start_idx`/`end_idx`.
- **`evaluate/harness.py` (new):** measures diagnostic accuracy against ground
  truth — confusion matrix, per-class accuracy, and nominal false-positive rate.
- **Tests:** `tests/test_diagnosis.py` locks in correct diagnosis + zero nominal
  false-positives; existing tests updated to the new interfaces.

## Result (`python -m evaluate.harness`)

~97% overall accuracy, **0% nominal false-positive rate**. Remaining misses are
very-sparse stiction runs (2–3 events in 4000 points) that are genuinely
near-indistinguishable from noise — conservatively called nominal, which is the
right failure mode for an alarm system. `pytest` is green (13 tests); `ruff` and
`mypy` clean on the changed modules.

## Detector tightening (follow-up, now done)

The v0 z-score detector fired ~150 events/run because a trailing rolling window
lagged the telemetry's normal sinusoid (and it also adapted to slow drifts like
friction and missed them). `ZScoreDetector` is now **baseline-referenced**: it
compares each channel to robust median/MAD statistics of a nominal baseline at the
start of the run, groups flags into events, and merges nearby blips. Result: ~2
low-severity events on nominal runs and a handful on fault runs (down from ~150),
with diagnosis accuracy unchanged (97.5%, 0% nominal false-positives).
`tests/test_diagnosis.py` guards this (nominal stays quiet; every fault is flagged).

## Detection metrics + real-data seam (follow-up, now done)

Added honest, event-wise detection scoring and the plug-in point for real data:

- `evaluate/detection_metrics.py` — range/overlap-based precision, recall, and
  **F0.5 (never point-adjusted)**, the metric the literature review calls out as
  the field's inflated standard.
- `evaluate/detection_harness.py` — `python -m evaluate.detection_harness`. Current
  synthetic result: 100% recall, ~59% precision, F0.5 ≈ 0.64 (honest; precision
  headroom is what the LSTM swap targets).
- `ingest/sources.py` — `TelemetrySource` protocol with `SyntheticSource` and
  `CsvTelemetrySource`. Real ESA-ADB / OPS-SAT-AD telemetry drops in as CSVs +
  an anomaly-interval labels file, with **no change to the detector or harness**
  (verified by a round-trip test). The diagnosis harness now shares this source.

## Still open (intentionally out of scope)

- Download + wire real ESA-ADB / OPS-SAT-AD telemetry through `CsvTelemetrySource`
  (needs the multi-GB dataset, so it happens on your machine, not here).
- The remaining swaps as planned: LSTM detector → `sbi`/NPE scorer → Basilisk
  twin. The `Protocol` interfaces already make these drop-in.
- Diagnosis on real data also needs a matching twin + fault library per subsystem;
  detection scoring works on real labels today.
