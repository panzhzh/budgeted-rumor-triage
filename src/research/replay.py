"""Calendar-time queue replay using saved 1h, 6h and 24h priority scores."""
from __future__ import annotations

import math
import numpy as np

WINDOWS = (1, 6, 24)
MIN_AGE = 1.0
MAX_AGE = 48.0


def replay(ids, source_hours, reaction_times, scores, capacity, policy):
    """Reward reactions strictly after review; denominator is fixed at age 1 h."""
    source_hours = np.asarray(source_hours, float)
    if capacity <= 0:
        raise ValueError('Positive review capacity required')
    start = float(source_hours.min() + MIN_AGE)
    end = float(source_hours.max() + MAX_AGE)
    service_times = start + np.arange(math.floor((end - start) * capacity) + 1) / capacity
    selected = set()
    records = []
    possible = sum((np.count_nonzero(t > MIN_AGE) for t in reaction_times))
    for now in service_times:
        age = now - source_hours
        candidates = [i for i in range(len(ids)) if i not in selected and MIN_AGE <= age[i] <= MAX_AGE]
        if not candidates:
            continue
        available_window = np.asarray([max((w for w in WINDOWS if w <= age[i])) for i in candidates])
        if policy == 'combined_catboost_ranker':
            priority = np.asarray([scores[i, WINDOWS.index(int(w))] for i, w in zip(candidates, available_window)])
        elif policy == 'early_volume':
            priority = np.asarray([np.searchsorted(reaction_times[i], age[i], side='right') for i in candidates], float)
        elif policy == 'fifo':
            priority = -source_hours[candidates]
        else:
            raise ValueError(policy)
        order = sorted(range(len(candidates)), key=lambda j: (-priority[j], source_hours[candidates[j]], str(ids[candidates[j]])))
        j = order[0]
        i = candidates[j]
        selected.add(i)
        reward = int(np.count_nonzero(reaction_times[i] > age[i]))
        records.append(dict(thread_id=str(ids[i]), review_time_hours=now, thread_age_hours=float(age[i]), snapshot_window_hours=int(available_window[j]), priority=float(priority[j]), observed_reactions=int(np.searchsorted(reaction_times[i], age[i], side='right')), post_review_growth=reward, policy=policy))
    captured = sum((row['post_review_growth'] for row in records))
    summary = dict(policy=policy, capacity_per_hour=capacity, threads=len(ids), reviewed=len(records), expired_or_unreviewed=len(ids) - len(records), future_growth_denominator=int(possible), captured_post_review_growth=captured, capture_rate=captured / possible if possible else np.nan, mean_review_age_hours=float(np.mean([r['thread_age_hours'] for r in records])) if records else np.nan, median_review_age_hours=float(np.median([r['thread_age_hours'] for r in records])) if records else np.nan, service_slots=len(service_times))
    return (summary, records)
