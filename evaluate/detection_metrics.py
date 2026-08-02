"""Detection metrics — honest, event-wise (range-based) precision/recall/F-beta.

The literature review is explicit that the field's standard "point-adjusted F1"
inflates scores to the point where random noise looks state-of-the-art, and that
evaluating honestly is a differentiator. So detection is scored **event-wise**, on
whether detected *intervals* overlap labelled anomaly *intervals* — never
point-adjusted.

Definitions (overlap / existence based):
  - A labelled anomaly is **recalled** if at least one detected interval overlaps it.
  - A detection is a **true positive** if it overlaps at least one labelled anomaly.
  - recall    = recalled anomalies / total labelled anomalies
  - precision = true-positive detections / total detections
  - F-beta    = (1+b^2) * P * R / (b^2 * P + R),  default beta=0.5 (precision-weighted,
    because on an operations console a false alarm is costlier than a slightly late catch).

Nominal data (no labelled anomalies) contributes only to precision: any detection
there is a false positive. This is the number that exposes an alarm system that
cries wolf — the exact failure the v0 pipeline had.
"""

from __future__ import annotations

from dataclasses import dataclass

Interval = tuple[int, int]  # half-open [start, end)


def _overlaps(a: Interval, b: Interval) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@dataclass(frozen=True)
class DetectionScore:
    precision: float
    recall: float
    fbeta: float
    beta: float
    n_detected: int
    n_labelled: int
    tp_detections: int  # detections overlapping some labelled anomaly
    recalled: int  # labelled anomalies hit by some detection

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "fbeta": self.fbeta,
            "beta": self.beta,
            "n_detected": self.n_detected,
            "n_labelled": self.n_labelled,
            "tp_detections": self.tp_detections,
            "recalled": self.recalled,
        }


def score_detection(
    detected: list[Interval],
    labelled: list[Interval],
    beta: float = 0.5,
) -> DetectionScore:
    """Range-based detection score over one or many runs' intervals.

    Pass the *pooled* intervals across all runs to get a corpus-level score
    (recommended), or a single run's intervals for a per-run score.
    """
    tp_detections = sum(1 for d in detected if any(_overlaps(d, t) for t in labelled))
    recalled = sum(1 for t in labelled if any(_overlaps(d, t) for d in detected))

    precision = (
        tp_detections / len(detected) if detected else (1.0 if not labelled else 0.0)
    )
    recall = recalled / len(labelled) if labelled else 1.0

    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0

    return DetectionScore(
        precision=precision,
        recall=recall,
        fbeta=fbeta,
        beta=beta,
        n_detected=len(detected),
        n_labelled=len(labelled),
        tp_detections=tp_detections,
        recalled=recalled,
    )


def aggregate_scores(scores: list[DetectionScore], beta: float = 0.5) -> DetectionScore:
    """Corpus-level score from per-run scores.

    Counts are summed across runs and precision/recall/F-beta recomputed from the
    totals. This is the correct way to combine runs — pooling raw intervals across
    runs would let a detection in one run spuriously "cover" an anomaly in another
    (their row indices are independent).
    """
    n_detected = sum(s.n_detected for s in scores)
    n_labelled = sum(s.n_labelled for s in scores)
    tp_detections = sum(s.tp_detections for s in scores)
    recalled = sum(s.recalled for s in scores)

    precision = tp_detections / n_detected if n_detected else 1.0
    recall = recalled / n_labelled if n_labelled else 1.0
    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0

    return DetectionScore(
        precision=precision,
        recall=recall,
        fbeta=fbeta,
        beta=beta,
        n_detected=n_detected,
        n_labelled=n_labelled,
        tp_detections=tp_detections,
        recalled=recalled,
    )


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping/touching intervals — used to collapse per-channel
    detections into per-run anomaly regions before scoring.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]
