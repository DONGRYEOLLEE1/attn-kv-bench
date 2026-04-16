"""Multi-turn benchmark workload definitions."""

from __future__ import annotations


def multi_turn_workload():
    long_a = (
        "System: You are a routing assistant for aquarium observations. "
        "Keep every prior detail about water current, feeding noise, light changes, visitor movement, "
        "glass reflections, and maintenance events in working memory before answering the next question."
    )
    long_b = (
        "User: Read this synthetic incident narrative carefully. During the last two hours the tank received food, "
        "then a vibration near the glass, then a brief shadow from the left side, then a stronger water current near the corner, "
        "then a quieter interval, then another small disturbance around the bubble wall. Keep this sequence in memory."
    )
    long_c = (
        "User: Extend the same context with a follow-up. Assume one observer noted that the fish changed direction three times, "
        "two fish stayed near the plant, one moved toward the top, and the filter noise increased slightly after the feeding event."
    )
    long_d = (
        "User: Add another turn. Suppose the room lights dimmed, a person approached the tank, and a small tap happened near the right side, "
        "while the current remained uneven between the heater and the bridge decoration."
    )
    return [
        [
            long_a, long_b, long_c, long_d,
            "User: Given the full running context, explain which signal should dominate routing if a new vibration arrives right after the shadow event.",
            "User: Using all previous turns, identify the safest short operational response and explain the reason in one concise paragraph.",
            "User: Now keep all prior turns in memory and decide whether the current, the shadow, or the tap is the primary trigger.",
        ],
        [
            "System: You are a concise support assistant. Preserve every prior constraint across turns and answer using the full running context.",
            "User: Read this long synthetic project log. Team A owns ingestion, Team B owns storage, Team C owns evaluation, deadlines are staggered, and two blockers affect downstream latency and release readiness.",
            "User: Extend the same log. Assume storage slipped, evaluation depends on storage, and release notes cannot finalize until both metrics and dashboards are regenerated.",
            "User: Add another long turn. A PM asks for reprioritization after a reliability incident, a partial rollback, and a delayed dependency patch in the deployment pipeline.",
            "User: Given all earlier turns, identify the highest-priority blocker.",
            "User: Now summarize the likely schedule impact using all previous context.",
            "User: Finally provide the shortest action memo that still reflects the entire conversation history.",
        ],
        [
            "System: You are a planner for multi-step operations. Keep all prior turns in memory.",
            "User: Here is a long synthetic inventory log with recurring anomalies in shipment, storage temperature, packaging variance, dock delays, and repeated handoff failures between staging and dispatch.",
            "User: Add another turn. Assume dock three had a delay, cold-chain compliance briefly degraded, and one package batch was rerouted twice before scan confirmation.",
            "User: Extend the log. The warehouse manager noticed label mismatches, repeated pallet swaps, and a mismatch between sensor time and dispatch time.",
            "User: Based on the full context so far, which anomaly matters most if cold-chain compliance becomes the top priority?",
            "User: Re-evaluate the likely root cause after the packaging incident and dock delay are both included.",
            "User: Produce a compact final response that still depends on the whole running context.",
        ],
        [
            "System: You are a concise analyst. Keep all previous turns in memory.",
            "User: Read this long synthetic conversation history about retries, cache misses, rate limits, queueing delay, background compaction, and latency regressions across multiple windows.",
            "User: Add detail. Assume cache warm-up completed, but queue depth rose before retry storms began and one downstream shard became imbalanced.",
            "User: Extend again. A second shard shows intermittent misses, rate limiting tightens, and the control plane introduces backoff after an operator intervention.",
            "User: Using all prior context, what is the most likely cause of the latency regression?",
            "User: If the retry budget is cut in half, what symptom changes first?",
            "User: Produce a compact final answer for an operations dashboard using the full multi-turn history.",
        ],
    ]
