"""Small, dependency-free CTC substring forced-alignment Viterbi core.

The function operates on model-produced log probabilities.  It deliberately
does not load an acoustic model, so model extraction can be cached/replaced
without changing the scoring contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CTCAlignment:
    align_logp_sum: float
    align_logp_mean: float
    token_score_min: float
    token_score_mean: float
    start_frame: int
    end_frame: int
    aligned_frames: int
    blank_ratio: float
    token_scores: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def forced_align_subsequence(
    log_probs: np.ndarray,
    tokens: Sequence[int],
    *,
    blank_id: int = 0,
) -> CTCAlignment:
    """Find the best CTC path for ``tokens`` anywhere in ``log_probs``.

    Leading and trailing frames are free context; the returned path still has
    to visit every target token in order.  Repeated labels require an
    intervening blank according to the CTC transition rule.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    target = [int(x) for x in tokens]
    if lp.ndim != 2 or lp.shape[0] <= 0 or lp.shape[1] <= 0:
        raise ValueError(f"log_probs must be [T,V], got {lp.shape}")
    if not target:
        raise ValueError("target tokens must be non-empty")
    if not np.isfinite(lp).all():
        raise ValueError("log_probs contains non-finite values")
    if any(x < 0 or x >= lp.shape[1] for x in target) or not (0 <= blank_id < lp.shape[1]):
        raise ValueError("token or blank id is outside vocabulary")
    t_count, token_count = lp.shape[0], len(target)
    states = [blank_id]
    for token in target:
        states.extend([token, blank_id])
    n_states = len(states)
    neg_inf = -np.inf
    scores = np.full(n_states, neg_inf)
    starts = np.full(n_states, -1, dtype=np.int64)
    back = np.full((t_count, n_states), -3, dtype=np.int32)
    prev_states = np.full((t_count, n_states), -1, dtype=np.int32)
    emissions = np.full((t_count, n_states), neg_inf)

    for t in range(t_count):
        next_scores = np.full(n_states, neg_inf)
        next_starts = np.full(n_states, -1, dtype=np.int64)
        for s, label in enumerate(states):
            emit = float(lp[t, label])
            emissions[t, s] = emit
            candidates: list[tuple[float, int, int]] = []
            if np.isfinite(scores[s]):
                candidates.append((float(scores[s]), s, int(starts[s])))
            if s > 0 and np.isfinite(scores[s - 1]):
                candidates.append((float(scores[s - 1]), s - 1, int(starts[s - 1])))
            if s > 1 and s % 2 == 1 and states[s - 2] != label and np.isfinite(scores[s - 2]):
                candidates.append((float(scores[s - 2]), s - 2, int(starts[s - 2])))
            # Free-context restart: this frame may be the beginning of the
            # target (blank or first token), without scoring prior frames.
            if s in (0, 1):
                candidates.append((0.0, -1, t))
            if not candidates:
                continue
            best, prev, start = max(candidates, key=lambda item: item[0])
            next_scores[s] = best + emit
            next_starts[s] = start if start >= 0 else t
            prev_states[t, s] = prev
        scores, starts = next_scores, next_starts

    end_candidates = [(float(scores[n_states - 1]), n_states - 1), (float(scores[n_states - 2]), n_states - 2)]
    score, state = max(end_candidates, key=lambda item: item[0])
    if not np.isfinite(score) or starts[state] < 0:
        raise ValueError("target cannot be aligned in log_probs")
    path_states: list[int] = []
    t = t_count - 1
    while t >= 0:
        path_states.append(state)
        prev = int(prev_states[t, state])
        if prev < 0:
            break
        state = prev
        t -= 1
    path_states.reverse()
    start_frame = int(starts[max(end_candidates, key=lambda item: item[0])[1]])
    end_frame = t_count - 1
    # Trim frames before the first token-start and after the final token was
    # reached; restart/trailing context is not part of the aligned span.
    first = next((i for i, s in enumerate(path_states) if s % 2 == 1), 0)
    last = len(path_states) - 1 - next((i for i, s in enumerate(reversed(path_states)) if s % 2 == 1), 0)
    path_states = path_states[first : last + 1]
    start_frame += first
    end_frame = start_frame + len(path_states) - 1
    token_scores = []
    for token_index in range(token_count):
        state_index = 2 * token_index + 1
        values = [float(emissions[start_frame + i, state_index]) for i, s in enumerate(path_states) if s == state_index]
        if not values:
            raise ValueError(f"token {token_index} has no aligned frame")
        token_scores.append(float(np.mean(values)))
    align_sum = float(sum(float(emissions[start_frame + i, s]) for i, s in enumerate(path_states)))
    return CTCAlignment(
        align_logp_sum=align_sum,
        align_logp_mean=float(np.mean(token_scores)),
        token_score_min=float(min(token_scores)),
        token_score_mean=float(np.mean(token_scores)),
        start_frame=start_frame,
        end_frame=end_frame,
        aligned_frames=len(path_states),
        blank_ratio=float(sum(s % 2 == 0 for s in path_states) / max(1, len(path_states))),
        token_scores=tuple(token_scores),
    )
