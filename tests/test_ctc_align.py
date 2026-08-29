from __future__ import annotations

import numpy as np
import pytest

from quick.ctc_align import forced_align_subsequence


def _log_probs(t: int = 8, v: int = 4) -> np.ndarray:
    x = np.full((t, v), 0.01, dtype=np.float64)
    x[:, 0] = 0.95
    x /= x.sum(axis=1, keepdims=True)
    return x


def test_subsequence_alignment_allows_context_and_returns_span():
    lp = _log_probs()
    lp[2] = [0.01, 0.95, 0.02, 0.02]
    lp[4] = [0.95, 0.01, 0.02, 0.02]
    lp[5] = [0.01, 0.02, 0.95, 0.02]
    lp /= lp.sum(axis=1, keepdims=True)
    result = forced_align_subsequence(np.log(lp), [1, 2], blank_id=0)
    assert result.start_frame <= result.end_frame
    assert len(result.token_scores) == 2
    assert result.token_score_min > -1.0


def test_repeated_ctc_token_requires_blank():
    lp = _log_probs(6, 3)
    lp[1] = [0.02, 0.9, 0.08]
    lp[2] = [0.9, 0.02, 0.08]
    lp[3] = [0.02, 0.9, 0.08]
    lp /= lp.sum(axis=1, keepdims=True)
    result = forced_align_subsequence(np.log(lp), [1, 1], blank_id=0)
    assert result.start_frame == 1
    assert result.end_frame == 3


def test_nonfinite_and_empty_target_rejected():
    with pytest.raises(ValueError):
        forced_align_subsequence(np.zeros((2, 3)), [], blank_id=0)
    with pytest.raises(ValueError):
        forced_align_subsequence(np.array([[0.0, np.nan, 0.0]]), [1], blank_id=0)
