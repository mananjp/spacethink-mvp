"""Cross-fleet anomaly clustering — fleet-wide signature correlation.

"This signature appeared on 7 of your satellites in 24h."
Recommends a coordinated response (commands, downlink priority).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from domain import EventOfInterest, FleetCluster, new_id


def _extract_event_features(event: EventOfInterest) -> np.ndarray:
    """Extract a feature vector from an event for clustering."""
    return np.array([
        event.score,
        {"low": 0, "medium": 1, "high": 2}.get(event.severity.value, 0),
        hash(event.channel) % 100 / 100.0,  # channel hash as proxy
        (event.end_ts - event.start_ts).total_seconds(),
    ])


def cluster_fleet_events(
    events: list[EventOfInterest],
    distance_threshold: float = 2.0,
    min_cluster_size: int = 2,
) -> list[FleetCluster]:
    """Cluster events across a fleet by signature similarity.

    Uses simple agglomerative clustering based on event feature vectors
    (score, severity, channel, duration).
    """
    if len(events) < min_cluster_size:
        return []

    # Extract features
    features = np.array([_extract_event_features(e) for e in events])

    # Normalize features
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    normalized = (features - mu) / sigma

    # Simple greedy clustering
    assigned = [False] * len(events)
    clusters: list[FleetCluster] = []

    for i in range(len(events)):
        if assigned[i]:
            continue

        cluster_indices = [i]
        assigned[i] = True

        for j in range(i + 1, len(events)):
            if assigned[j]:
                continue
            dist = float(np.linalg.norm(normalized[i] - normalized[j]))
            if dist < distance_threshold:
                cluster_indices.append(j)
                assigned[j] = True

        if len(cluster_indices) >= min_cluster_size:
            cluster_events = [events[k] for k in cluster_indices]
            centroid = normalized[cluster_indices].mean(axis=0)
            centroid_dist = float(np.mean([
                np.linalg.norm(normalized[k] - centroid) for k in cluster_indices
            ]))

            # Determine representative mechanism from most common channel
            from collections import Counter
            channel_counts = Counter(e.channel for e in cluster_events)
            top_channel = channel_counts.most_common(1)[0][0]

            # Recommendation based on cluster characteristics
            avg_score = np.mean([e.score for e in cluster_events])
            if avg_score > 5.0:
                action = (
                    f"URGENT: {len(cluster_events)} satellites showing correlated "
                    f"high-severity anomaly on {top_channel}. Initiate fleet-wide "
                    f"diagnostic downlink and prepare contingency telecommands."
                )
            elif avg_score > 3.0:
                action = (
                    f"ATTENTION: {len(cluster_events)} satellites with similar "
                    f"anomaly pattern on {top_channel}. Schedule coordinated "
                    f"high-rate downlink for root cause analysis."
                )
            else:
                action = (
                    f"MONITOR: {len(cluster_events)} satellites showing correlated "
                    f"low-severity pattern on {top_channel}. Continue monitoring; "
                    f"no immediate action required."
                )

            clusters.append(FleetCluster(
                cluster_id=new_id(),
                event_ids=tuple(e.id for e in cluster_events),
                representative_mechanism=f"fleet_cluster_{top_channel}",
                centroid_distance=centroid_dist,
                recommended_action=action,
            ))

    return clusters


def format_fleet_report(clusters: list[FleetCluster]) -> str:
    """Format fleet clustering results as Markdown."""
    if not clusters:
        return "## Fleet Analysis\nNo correlated anomaly patterns detected across the fleet."

    lines = [
        "## 🛰️ Fleet-Wide Anomaly Clustering",
        "",
        f"**{len(clusters)} correlated cluster(s) detected**",
        "",
    ]

    for i, cluster in enumerate(clusters, 1):
        n_sats = len(cluster.event_ids)
        lines.extend([
            f"### Cluster {i}: {n_sats} satellites",
            f"- **Pattern:** `{cluster.representative_mechanism}`",
            f"- **Cohesion:** {cluster.centroid_distance:.3f}",
            f"- **Recommendation:** {cluster.recommended_action}",
            "",
        ])

    return "\n".join(lines)
