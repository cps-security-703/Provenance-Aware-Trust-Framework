"""
Reputation System for Autonomous Vehicle Update Validation (Section IV-B / IV-C)

Implements:
  - Per-source reputation scores with exponential temporal decay
  - Sybil resistance via minimum interaction thresholds and diversity checks
  - Collusion detection via correlated voting pattern analysis

The reputation score for source *s* at time *t* is:

    R_s(t) = (1/Z) * sum_{i} w_i * outcome_i * exp(-lambda * (t - t_i))

where:
    w_i       = weight of interaction i (positive for correct verdicts, negative for false ones)
    outcome_i = +1 for correct validation, -1 for incorrect
    lambda    = temporal decay constant
    Z         = normalisation factor
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_REPUTATION = 0.5
DECAY_LAMBDA = 0.01          # per-timestep exponential decay
MIN_INTERACTIONS = 5          # Sybil: require this many observations before trusting
SYBIL_DIVERSITY_THRESHOLD = 1 # Sybil: require interactions across >= N distinct channels
COLLUSION_CORRELATION_THRESH = 0.85  # flag pairs whose agreement rate exceeds this
REPUTATION_FLOOR = 0.0
REPUTATION_CEILING = 1.0


class ReputationRecord:
    """Single interaction record for a source."""

    __slots__ = ("timestamp", "outcome", "channel", "weight")

    def __init__(self, timestamp: float, outcome: float,
                 channel: str = "unknown", weight: float = 1.0):
        self.timestamp = timestamp
        self.outcome = outcome      # +1 correct, -1 incorrect
        self.channel = channel
        self.weight = weight


class ReputationSystem:
    """
    Per-source reputation tracker with temporal decay, Sybil resistance,
    and collusion detection.
    """

    def __init__(self, decay_lambda: float = DECAY_LAMBDA,
                 min_interactions: int = MIN_INTERACTIONS,
                 diversity_threshold: int = SYBIL_DIVERSITY_THRESHOLD,
                 collusion_threshold: float = COLLUSION_CORRELATION_THRESH):
        self.decay_lambda = decay_lambda
        self.min_interactions = min_interactions
        self.diversity_threshold = diversity_threshold
        self.collusion_threshold = collusion_threshold

        self._records: Dict[str, List[ReputationRecord]] = defaultdict(list)
        self._flagged_sybil: set = set()
        self._flagged_collusion: set = set()  # pairs stored as frozensets

    # ------------------------------------------------------------------
    # Core reputation update (Section IV-B)
    # ------------------------------------------------------------------
    def record_interaction(self, source_id: str, timestamp: float,
                           was_correct: bool, channel: str = "unknown",
                           weight: float = 1.0) -> None:
        """Record a single interaction outcome for *source_id*."""
        outcome = 1.0 if was_correct else -1.0
        self._records[source_id].append(
            ReputationRecord(timestamp, outcome, channel, weight)
        )

    def get_reputation(self, source_id: str,
                       current_time: Optional[float] = None) -> float:
        """
        Compute reputation score with exponential temporal decay.

            R_s(t) = (1/Z) * sum_i  w_i * o_i * exp(-lambda*(t - t_i))

        Returns a value in [0, 1].  Sources with too few interactions
        return the initial prior (0.5).
        """
        records = self._records.get(source_id)
        if not records or len(records) < self.min_interactions:
            return DEFAULT_INITIAL_REPUTATION

        if current_time is None:
            current_time = max(r.timestamp for r in records)

        weighted_sum = 0.0
        norm = 0.0
        for r in records:
            decay = math.exp(-self.decay_lambda * (current_time - r.timestamp))
            weighted_sum += r.weight * r.outcome * decay
            norm += r.weight * decay

        if norm == 0:
            return DEFAULT_INITIAL_REPUTATION

        raw = (weighted_sum / norm + 1.0) / 2.0  # map [-1,1] -> [0,1]
        return max(REPUTATION_FLOOR, min(REPUTATION_CEILING, raw))

    # ------------------------------------------------------------------
    # Sybil resistance (Section IV-C)
    # ------------------------------------------------------------------
    def check_sybil(self, source_id: str) -> Dict:
        """
        Assess whether *source_id* exhibits Sybil-like behaviour:
          1. Fewer than MIN_INTERACTIONS total observations
          2. All interactions over fewer than DIVERSITY_THRESHOLD distinct channels

        Returns a dict with 'is_suspicious', 'reason', and detail fields.
        """
        records = self._records.get(source_id, [])
        n = len(records)
        channels = set(r.channel for r in records)

        result: Dict = {
            "source_id": source_id,
            "is_suspicious": False,
            "reason": None,
            "interaction_count": n,
            "channel_diversity": len(channels),
        }

        if n < self.min_interactions:
            result["is_suspicious"] = True
            result["reason"] = (
                f"Insufficient interaction history ({n} < {self.min_interactions})"
            )
            self._flagged_sybil.add(source_id)
            return result

        if len(channels) < self.diversity_threshold:
            result["is_suspicious"] = True
            result["reason"] = (
                f"Low channel diversity ({len(channels)} < {self.diversity_threshold})"
            )
            self._flagged_sybil.add(source_id)
            return result

        # Additional heuristic: if all outcomes are identical (+1 or -1),
        # the source may be fabricating a track record.
        outcomes = [r.outcome for r in records]
        if len(set(outcomes)) == 1 and n >= self.min_interactions * 2:
            result["is_suspicious"] = True
            result["reason"] = "Uniform outcome history (potential track-record inflation)"
            self._flagged_sybil.add(source_id)
            return result

        # Source passed all checks — remove from flagged set if previously flagged
        self._flagged_sybil.discard(source_id)
        return result

    # ------------------------------------------------------------------
    # Collusion detection (Section IV-C)
    # ------------------------------------------------------------------
    def detect_collusion(self, time_window: Optional[Tuple[float, float]] = None
                         ) -> List[Dict]:
        """
        Detect potential collusion by identifying source pairs whose
        validation verdicts agree at a rate >= COLLUSION_CORRELATION_THRESH
        within the specified time window.

        Returns a list of flagged pairs with their agreement rate.
        """
        sources = list(self._records.keys())
        if len(sources) < 2:
            return []

        def _outcomes_in_window(src: str) -> Dict[float, float]:
            recs = self._records[src]
            out = {}
            for r in recs:
                if time_window and not (time_window[0] <= r.timestamp <= time_window[1]):
                    continue
                out[r.timestamp] = r.outcome
            return out

        source_outcomes = {s: _outcomes_in_window(s) for s in sources}
        flagged = []

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s1, s2 = sources[i], sources[j]
                o1, o2 = source_outcomes[s1], source_outcomes[s2]

                common_times = set(o1.keys()) & set(o2.keys())
                if len(common_times) < self.min_interactions:
                    continue

                agreements = sum(
                    1 for t in common_times if o1[t] == o2[t]
                )
                rate = agreements / len(common_times)

                if rate >= self.collusion_threshold:
                    pair = frozenset([s1, s2])
                    self._flagged_collusion.add(pair)
                    flagged.append({
                        "source_pair": (s1, s2),
                        "agreement_rate": round(rate, 4),
                        "common_interactions": len(common_times),
                    })

        return flagged

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def get_all_reputations(self, current_time: Optional[float] = None
                            ) -> Dict[str, float]:
        """Return reputation scores for every known source."""
        return {
            src: self.get_reputation(src, current_time)
            for src in self._records
        }

    def get_flagged_sources(self) -> Dict:
        """Return all flagged Sybil and collusion entities."""
        return {
            "sybil_suspects": list(self._flagged_sybil),
            "collusion_pairs": [
                tuple(pair) for pair in self._flagged_collusion
            ],
        }

    def summary(self, current_time: Optional[float] = None) -> Dict:
        """Full summary of the reputation system state."""
        reps = self.get_all_reputations(current_time)
        return {
            "total_sources": len(self._records),
            "reputation_scores": reps,
            "sybil_suspects": list(self._flagged_sybil),
            "collusion_pairs": [tuple(p) for p in self._flagged_collusion],
            "mean_reputation": (
                sum(reps.values()) / len(reps) if reps else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Temporal decay analysis (R-3 reviewer response)
    # ------------------------------------------------------------------
    def get_temporal_decay_analysis(self, current_time: Optional[float] = None) -> Dict:
        """
        Provide per-source temporal decay breakdown for transparency.

        For each source returns:
          - reputation: current score
          - total_interactions: number of recorded interactions
          - channel_diversity: number of distinct channels used
          - recent_weight_fraction: fraction of total weight from the most
            recent 20% of interactions (shows how much recent data dominates)
          - oldest_interaction: timestamp of first interaction
          - newest_interaction: timestamp of last interaction
          - effective_decay: 1 - exp(-lambda * observation_span)
        """
        analysis = {}
        for src, records in self._records.items():
            if not records:
                continue

            if current_time is None:
                t_now = max(r.timestamp for r in records)
            else:
                t_now = current_time

            timestamps = sorted(r.timestamp for r in records)
            channels = set(r.channel for r in records)

            # Compute total decayed weight and recent-fraction
            total_weight = 0.0
            for r in records:
                total_weight += r.weight * math.exp(-self.decay_lambda * (t_now - r.timestamp))

            # Recent 20% of interactions
            n_recent = max(1, len(records) // 5)
            recent_recs = sorted(records, key=lambda r: r.timestamp)[-n_recent:]
            recent_weight = sum(
                r.weight * math.exp(-self.decay_lambda * (t_now - r.timestamp))
                for r in recent_recs
            )

            span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

            analysis[src] = {
                "reputation": self.get_reputation(src, t_now),
                "total_interactions": len(records),
                "channel_diversity": len(channels),
                "channels": sorted(channels),
                "recent_weight_fraction": round(recent_weight / total_weight, 4) if total_weight > 0 else 0.0,
                "oldest_interaction": timestamps[0],
                "newest_interaction": timestamps[-1],
                "observation_span": span,
                "effective_decay": round(1.0 - math.exp(-self.decay_lambda * span), 4) if span > 0 else 0.0,
            }

        return analysis


# ---------------------------------------------------------------------------
# Demonstration / self-test
# ---------------------------------------------------------------------------
def main():
    print("=== Reputation System Demo ===\n")

    rs = ReputationSystem()

    # Simulate 3 sources with varying behaviour
    # Source A: mostly correct, diverse channels
    for t in range(20):
        rs.record_interaction("SourceA", timestamp=float(t),
                              was_correct=(t % 5 != 0),  # 80% correct
                              channel=["OTA", "P2P", "charging_station", "nearby_car"][t % 4])

    # Source B: always correct but only 1 channel (Sybil-suspicious)
    for t in range(15):
        rs.record_interaction("SourceB", timestamp=float(t),
                              was_correct=True,
                              channel="OTA")

    # Source C: mimics Source B exactly at same timestamps (collusion candidate)
    for t in range(15):
        rs.record_interaction("SourceC", timestamp=float(t),
                              was_correct=True,
                              channel="OTA")

    # Source D: low history
    for t in range(3):
        rs.record_interaction("SourceD", timestamp=float(t),
                              was_correct=True, channel="P2P")

    current_time = 25.0

    # --- Reputation scores ---
    print("Reputation Scores:")
    for src in ["SourceA", "SourceB", "SourceC", "SourceD"]:
        rep = rs.get_reputation(src, current_time)
        print(f"  {src}: {rep:.4f}")

    # --- Sybil checks ---
    print("\nSybil Analysis:")
    for src in ["SourceA", "SourceB", "SourceC", "SourceD"]:
        result = rs.check_sybil(src)
        status = "SUSPICIOUS" if result["is_suspicious"] else "OK"
        reason = result.get("reason", "-")
        print(f"  {src}: {status}  ({reason})")

    # --- Collusion detection ---
    print("\nCollusion Detection:")
    colluders = rs.detect_collusion()
    if colluders:
        for c in colluders:
            print(f"  Pair {c['source_pair']}: agreement={c['agreement_rate']:.2%} "
                  f"over {c['common_interactions']} interactions")
    else:
        print("  No collusion detected.")

    # --- Summary ---
    print(f"\nFlagged entities: {rs.get_flagged_sources()}")
    print(f"\nFull summary: {rs.summary(current_time)}")

    return rs


if __name__ == "__main__":
    rs = main()
