"""Detector precision contract — nominal runs must not raise events.

The regression these guard against: ``ZScoreDetector`` referenced a fixed nominal
baseline taken from the *start* of the run, sized as a fraction of the run length
with no regard for how fast each channel oscillates. On the synthetic
reaction-wheel data the baseline spanned 2.00 periods of speed and current but
only 0.50 periods of temperature, so the baseline never observed the temperature
trough (baseline range [24.47, 28.63] against a full range reaching 21.20). The
entire negative lobe of normal thermal oscillation then read as a level shift:
1720 points flagged at max robust-z 10.8, grouped into 2 events, on every seed.

The invariant that fixes it, and that these tests pin: a channel's baseline must
span at least one full cycle of that channel's own dominant periodicity, so that
"normal" includes the whole normal excursion. This is not synthetic-data trivia —
real spacecraft telemetry carries orbital (~90 min) and thermal cycles far slower
than the attitude-control channels sharing the same frame, so a single run-length
fraction cannot serve every channel.

Precision is only half the contract: a detector that flags nothing has zero false
positives. Every test here is paired with a recall assertion.
"""

from __future__ import annotations

import warnings

import pytest

from explore.detector import ZScoreDetector
from ingest.synthetic_generator import generate_reaction_wheel_telemetry

CHANNELS = ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
SEEDS = list(range(10))
FAULTS = ["friction_increase", "encoder_dropout", "stiction"]


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_nominal_run_raises_no_events(seed):
    """Normal oscillation is not an anomaly, on any channel."""
    df = generate_reaction_wheel_telemetry(fault_type="none", seed=seed)
    events = ZScoreDetector().detect(df, CHANNELS)
    assert events == [], (
        f"seed={seed}: {len(events)} false positive(s) on "
        f"{sorted({e.channel for e in events})}"
    )


def test_temperature_channel_specifically_is_clean():
    """The channel that carried the whole regression, called out by name."""
    offenders = 0
    for seed in SEEDS:
        df = generate_reaction_wheel_telemetry(fault_type="none", seed=seed)
        events = ZScoreDetector().detect(df, CHANNELS)
        offenders += sum(1 for e in events if e.channel == "wheel_temp_c")
    assert offenders == 0, f"{offenders} spurious wheel_temp_c events across {len(SEEDS)} runs"


def test_slow_channel_baseline_covers_a_full_cycle():
    """The mechanism, not just the symptom.

    The temperature channel oscillates with a 2000-point period; a baseline shorter
    than that cannot contain the channel's normal range, whatever the threshold.
    """
    df = generate_reaction_wheel_telemetry(fault_type="none", seed=0)
    detector = ZScoreDetector()
    baseline_n = detector.baseline_length(df["wheel_temp_c"].to_numpy(dtype=float))
    assert baseline_n >= 2000, (
        f"temperature baseline is {baseline_n} points, shorter than its 2000-point cycle"
    )


def test_fast_channels_keep_a_valid_baseline():
    """The slow channel's fix must not cost the fast channels their baseline."""
    df = generate_reaction_wheel_telemetry(fault_type="none", seed=0)
    detector = ZScoreDetector()
    for ch in ("wheel_speed_rpm", "wheel_current_a"):
        assert detector.baseline_length(df[ch].to_numpy(dtype=float)) > 0, (
            f"{ch} lost its baseline"
        )


# ---------------------------------------------------------------------------
# Recall — the other half of the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_faults_are_still_detected(fault, seed):
    """Precision must not be bought with recall."""
    df = generate_reaction_wheel_telemetry(fault_type=fault, seed=seed)
    events = ZScoreDetector().detect(df, CHANNELS)
    assert events, f"{fault} (seed={seed}) went undetected"


@pytest.mark.parametrize("fault", FAULTS)
def test_detected_events_start_after_the_fault_onset(fault):
    """Events should localise to the fault, not to baseline-relative artefacts."""
    df = generate_reaction_wheel_telemetry(fault_type=fault, seed=0)
    fault_start = df.attrs["fault_start"]
    events = ZScoreDetector().detect(df, CHANNELS)
    assert events, f"{fault} went undetected"
    earliest = min(e.metadata["start_idx"] for e in events)
    assert earliest >= fault_start * 0.8, (
        f"{fault}: earliest event at idx {earliest} precedes fault onset {fault_start}"
    )


def test_under_characterised_channel_says_so():
    """Surface reduced confidence rather than letting it pass unremarked.

    At 3000 points the temperature baseline lands mid-cycle and the held-out tail of
    the nominal region leaves the range the baseline established, so "normal" for that
    channel rests on an unvalidated default.

    The detector warns and continues rather than abstaining. Abstention was the first
    design and it is wrong: a baseline can fail validation because the channel cycles
    slowly *or* because a fault already started inside the region, and those two are
    indistinguishable from within it. Abstaining silently drops every fault of the
    second kind — the stiction case in test_pipeline.py is exactly that shape.
    """
    df = generate_reaction_wheel_telemetry(n_points=3000, fault_type="none", seed=0)
    with pytest.warns(UserWarning, match="under-characterised"):
        ZScoreDetector().detect(df, ["wheel_temp_c"])


def test_events_record_whether_their_baseline_was_validated():
    """Reduced confidence must outlive the warning that announced it."""
    validated = generate_reaction_wheel_telemetry(fault_type="friction_increase", seed=0)
    events = ZScoreDetector().detect(validated, CHANNELS)
    assert events and all(e.metadata["baseline_validated"] for e in events)

    fallback = generate_reaction_wheel_telemetry(
        fault_type="stiction", n_points=4000, fault_start=1500, seed=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fallback_events = ZScoreDetector().detect(fallback, ["wheel_current_a"])
    assert fallback_events and not any(
        e.metadata["baseline_validated"] for e in fallback_events
    )


def test_early_fault_inside_the_nominal_region_is_still_caught():
    """The recall half of the abstention trade-off, pinned.

    The fault starts at 1500 of 4000 points — inside the presumed-nominal region — so
    no baseline in that region validates. The detector must still flag it.
    """
    df = generate_reaction_wheel_telemetry(
        fault_type="stiction", n_points=4000, fault_start=1500, seed=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = ZScoreDetector().detect(df, ["wheel_speed_rpm", "wheel_current_a"])
    assert events, "an early fault must not be swallowed by baseline growth"


def test_short_runs_are_a_known_limit_not_a_silent_one():
    """Document the residual limitation, in code, so it cannot be forgotten.

    Below roughly one nominal cycle the detector genuinely cannot separate a slow
    nominal excursion from a slow fault: the information is not in the presumed-nominal
    region. It either abstains (held-out check fires) or may still raise a flag on the
    descent. What it must never do is silently emit the 2-per-run temperature artefact
    that motivated this work.
    """
    detector = ZScoreDetector()
    for n_points in (1200, 1500, 2000):
        df = generate_reaction_wheel_telemetry(n_points=n_points, fault_type="none", seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events = detector.detect(df, ["wheel_temp_c"])
        assert len(events) <= 1, (
            f"n_points={n_points}: {len(events)} temperature events on a nominal run — "
            f"the original regression produced 2 per run and is expected to be gone"
        )
