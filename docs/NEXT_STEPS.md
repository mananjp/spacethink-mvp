# spaceThink — Current State & Next Implementation Options

Living doc tying the code to the roadmap and the product decisions. Companion to
`FIX_NOTES.md` (what the diagnosis fix changed), `docs/STRATEGIC_DIRECTIONS.md`
(A/B/C product directions), and `docs/IMPLEMENTATION_PLAN.md` (phase plan).

## Where the code is now

The EXHYTE loop closes end-to-end and **diagnoses correctly** on synthetic
reaction-wheel data: ~97% mechanism accuracy, 0% nominal false-positives
(`python -m evaluate.harness`). Detection is measured honestly, event-wise, never
point-adjusted (`python -m evaluate.detection_harness`).

Pipeline components are now **pluggable** (`plan/components.py`), selectable from
the default path and the CLI:

```
python -m cli.main run <csv> --detector zscore|telemanom --scorer signature|distance|sbi
```

- **Detectors:** `zscore` (baseline-referenced robust z-score, default) · `telemanom` (LSTM/exp-smoothing forecaster lineage, ESA-ADB-oriented)
- **Scorers:** `signature` (event-window fault-signature, default) · `distance` (whole-series RMSE) · `sbi` (amortized NPE; kernel fallback until trained)

Advanced components load their heavy deps (torch/sbi) lazily and fall back
gracefully, so the default path stays hermetic.

## Measured component comparison (synthetic, 20 runs)

Honest numbers — this is *why* the defaults are what they are:

| detector | scorer | mechanism accuracy | nominal false-positives |
|---|---|---|---|
| **zscore** | **signature** (default) | **95%** | 0/5 |
| zscore | distance | 50% | 0/5 |
| zscore | sbi | 50% | **5/5** |
| telemanom | signature | 65% | 0/5 |
| telemanom | sbi | 70% | 0/5 |

**Read:** the proven default wins by a wide margin today. `SBIScorer` is *worse*
than default on this data — it is untrained (kernel fallback) and scores
whole-series summary stats, re-introducing nominal false-positives. `telemanom`
under-performs `zscore` here because it is designed for real, non-stationary
telemetry (ESA-ADB), not this clean synthetic signal. **So they are wired in and
selectable, but not default — flip the default once the evidence changes.**

## Next implementation options (roughly by ROI)

1. **Train SBIScorer** (`scripts/train_sbi_scorer.py`, needs `sbi`), commit the
   posteriors, and re-run the comparison. If it beats `signature`, make it default
   via `plan/components.py` — one-line change, no planner edits.
2. **Real data:** point `ingest/opssat_ad.py` / `CsvTelemetrySource` at real
   ESA-ADB, and run the detection harness on real labels. This is where
   `telemanom` should start to win, and where the "honest metrics" story becomes
   a real benchmark number.
3. **Higher-fidelity twin:** wire `twin/basilisk_twin.py` (needs Basilisk) behind
   the existing `Twin` seam; keep `ToySimulator` as the hermetic-CI default.
4. **Detector precision:** close the residual nominal blips (temp channel) —
   per-channel thresholds / telecommand conditioning (`telemanom` already models
   telecommands).
5. **Onboard autonomy:** `explore/onboard.py` + `evaluate/autonomy_gate.py` — the
   "robot runs its own experiments under comms delay" path; later-stage in all
   directions.

## Open product decisions (see docs/STRATEGIC_DIRECTIONS.md)

- **Direction A / B / C** — science-first vs. mission+wedge vs. ops-first. This
  changes the customer, the demo, and the metrics. The code is direction-agnostic
  (same engine); the current demo tells a Direction-C (diagnosis) story, while the
  Verifier/calibration/autonomy work leans toward A.
- **Default scorer** — `signature` now; `sbi` once trained.
- **Non-code (matters most for YC):** team/founder story, unit economics
  (cost-per-analysis vs. price), twin provenance (SaaS vs. consulting), data
  cold-start. Not in the code; tracked here so they don't get lost.
