"""Evaluation harness — measures diagnostic accuracy against ground truth.

The synthetic generator knows the true fault for every run, so we can score the
whole closed loop honestly: generate labelled runs, diagnose each, and compare
the predicted mechanism to the truth. This is the project's stated differentiator
("we evaluate honestly in a field full of inflated numbers") made concrete — and
it is what would have caught the v0 engine confidently diagnosing nominal data.

Reported metrics (no point-adjustment, per the literature review):
  - overall accuracy (correct mechanism / total runs),
  - a confusion matrix over the four mechanisms,
  - the nominal false-positive rate (nominal runs diagnosed as some fault) —
    the single most operationally important number for an alarm system.

Run directly:  python -m evaluate.harness
"""

from __future__ import annotations

from collections import Counter

from ingest.sources import FAULT_TO_MECHANISM, SyntheticSource
from plan.planner import run_closed_loop
from runstore import RunStore

# De-duplicated mechanism labels, in a stable order (dict preserves insertion).
MECHANISMS = list(dict.fromkeys(FAULT_TO_MECHANISM.values()))


def _run_prediction(report: dict) -> str:
    """Collapse a run's per-event diagnoses to one run-level mechanism.

    No detected events -> nominal. Otherwise the most common top hypothesis
    across events (majority vote).
    """
    if not report["events"]:
        return "nominal_no_fault"
    votes = Counter(e["top_hypothesis"] for e in report["events"] if e["top_hypothesis"])
    if not votes:
        return "nominal_no_fault"
    return votes.most_common(1)[0][0]


def evaluate(
    n_per_class: int = 5,
    n_points: int = 4000,
    fault_start: int = 2000,
    n_sims_per_hypothesis: int = 8,
    base_seed: int = 100,
    detector_name: str | None = None,
    scorer_name: str | None = None,
) -> dict:
    """Diagnose ``n_per_class`` runs of each fault type and score the result.

    ``detector_name`` / ``scorer_name`` select pipeline components (None = the
    proven defaults, zscore + signature) so combinations can be compared honestly.
    """
    from plan.components import build_detector, build_scorer

    # Confusion matrix keyed [truth_mechanism][predicted_mechanism].
    confusion: dict[str, Counter] = {t: Counter() for t in MECHANISMS}
    per_class_correct: Counter = Counter()
    per_class_total: Counter = Counter()
    store = RunStore()

    source = SyntheticSource(
        n_per_class=n_per_class,
        n_points=n_points,
        fault_start=fault_start,
        base_seed=base_seed,
    )
    for run in source.runs():
        truth = run.truth_mechanism
        assert truth is not None  # synthetic runs are always labelled
        report = run_closed_loop(
            run.telemetry,
            n_sims_per_hypothesis=n_sims_per_hypothesis,
            store=store,
            detector=build_detector(detector_name),
            scorer=build_scorer(scorer_name),
        )
        predicted = _run_prediction(report)
        confusion[truth][predicted] += 1
        per_class_total[truth] += 1
        if predicted == truth:
            per_class_correct[truth] += 1

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())
    accuracy = correct / total if total else 0.0

    # Nominal false-positive rate: nominal runs diagnosed as any real fault.
    nominal_total = per_class_total["nominal_no_fault"]
    nominal_fp = nominal_total - confusion["nominal_no_fault"]["nominal_no_fault"]
    nominal_fp_rate = nominal_fp / nominal_total if nominal_total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": {
            m: (per_class_correct[m] / per_class_total[m] if per_class_total[m] else 0.0)
            for m in MECHANISMS
        },
        "nominal_false_positive_rate": nominal_fp_rate,
        "confusion": {t: dict(confusion[t]) for t in MECHANISMS},
    }


def format_report(result: dict) -> str:
    """Render the evaluation result as a readable confusion-matrix report."""
    lines = [
        "=" * 68,
        "spaceThink diagnostic evaluation (synthetic, ground-truth labelled)",
        "=" * 68,
    ]
    short = {m: m.replace("bearing_", "").replace("_no_fault", "")[:9] for m in MECHANISMS}
    header = "truth \\ pred |" + "".join(f"{short[m]:>10}" for m in MECHANISMS)
    lines.append(header)
    lines.append("-" * len(header))
    for t in MECHANISMS:
        row = f"{short[t]:>12} |" + "".join(
            f"{result['confusion'][t].get(m, 0):>10}" for m in MECHANISMS
        )
        lines.append(row)
    lines.append("-" * len(header))
    lines.append(
        f"overall accuracy         : {result['accuracy']:.1%} ({result['correct']}/{result['total']})"
    )
    lines.append(f"nominal false-positive   : {result['nominal_false_positive_rate']:.1%}")
    lines.append("per-class accuracy       :")
    for m in MECHANISMS:
        lines.append(f"    {short[m]:>10}: {result['per_class_accuracy'][m]:.1%}")
    lines.append("=" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    result = evaluate()
    print(format_report(result))
