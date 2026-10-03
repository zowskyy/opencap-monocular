"""Per-frame confidence and unreliable-segment flagging (pure Python)."""

from typing import List, Sequence, Tuple


def frame_confidence(
    keypoint_conf: Sequence[Sequence[float]],
    valid_mask: Sequence[bool] = None,
) -> List[float]:
    """Mean 2D keypoint confidence per frame, zeroed where ``valid_mask`` is
    False (subject absent or truncated)."""
    out = []
    for i, kp in enumerate(keypoint_conf):
        c = sum(kp) / len(kp) if len(kp) else 0.0
        if valid_mask is not None and not valid_mask[i]:
            c = 0.0
        out.append(float(c))
    return out


def low_confidence_segments(
    conf: Sequence[float], threshold: float = 0.5, min_len: int = 3
) -> List[Tuple[int, int]]:
    """Half-open (start, end) runs of frames below ``threshold``."""
    segs, start = [], None
    for i, c in enumerate(list(conf) + [float("inf")]):
        if c < threshold and start is None:
            start = i
        elif c >= threshold and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    return segs
