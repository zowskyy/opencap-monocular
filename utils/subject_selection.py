"""Target-person selection for multi-person videos.

Operates on tracks of per-frame 2D boxes (xyxy) so it is independent of the
detector/tracker used. Handles choosing the target (by track id, a user click,
or largest subject), bridging track loss by re-identification and flagging
frames where the subject is truncated or out of frame.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Track:
    track_id: int
    boxes: Dict[int, Box] = field(default_factory=dict)  # frame -> box

    @property
    def frames(self) -> List[int]:
        return sorted(self.boxes)


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def point_in_box(pt: Tuple[float, float], b: Box) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def select_target_track(
    tracks: Sequence[Track],
    track_id: Optional[int] = None,
    click: Optional[Tuple[float, float, int]] = None,
) -> Track:
    """Pick the target track.

    ``track_id`` selects explicitly; ``click`` is ``(x, y, frame)`` and picks
    the track whose box contains the point on that frame (smallest box wins
    for overlaps); otherwise the track with the largest total box area.
    """
    if not tracks:
        raise ValueError("No person tracks available")
    if track_id is not None:
        for t in tracks:
            if t.track_id == track_id:
                return t
        raise ValueError(f"Track id {track_id} not found")
    if click is not None:
        x, y, frame = click
        hits = [
            (box_area(t.boxes[frame]), t)
            for t in tracks
            if frame in t.boxes and point_in_box((x, y), t.boxes[frame])
        ]
        if not hits:
            raise ValueError(f"No person at ({x}, {y}) on frame {frame}")
        return min(hits, key=lambda h: h[0])[1]
    return max(tracks, key=lambda t: sum(box_area(b) for b in t.boxes.values()))


def merge_tracks_reid(
    target: Track,
    others: Sequence[Track],
    max_gap: int = 30,
    min_iou: float = 0.3,
    max_area_ratio: float = 2.0,
) -> Track:
    """Re-link fragments to the target after track loss.

    A fragment is appended when it starts within ``max_gap`` frames after the
    target's last frame, its first box overlaps the target's last box
    (IoU >= ``min_iou``) and its size is similar. Greedy and repeated until no
    more fragments link.
    """
    merged = Track(target.track_id, dict(target.boxes))
    pool = [t for t in others if t.track_id != target.track_id and t.boxes]
    changed = True
    while changed:
        changed = False
        last_f = merged.frames[-1]
        last_box = merged.boxes[last_f]
        best, best_iou = None, 0.0
        for t in pool:
            first_f = t.frames[0]
            if not 0 < first_f - last_f <= max_gap:
                continue
            fb = t.boxes[first_f]
            a1, a2 = box_area(last_box), box_area(fb)
            if min(a1, a2) <= 0 or max(a1, a2) / min(a1, a2) > max_area_ratio:
                continue
            iou = box_iou(last_box, fb)
            if iou >= min_iou and iou > best_iou:
                best, best_iou = t, iou
        if best is not None:
            merged.boxes.update(best.boxes)
            pool.remove(best)
            changed = True
    return merged


def is_truncated(
    box: Box, width: int, height: int, margin: float = 2.0, tol: float = 0.0
) -> bool:
    """True if the box touches/leaves the frame border."""
    return (
        box[0] < margin - tol
        or box[1] < margin - tol
        or box[2] > width - margin + tol
        or box[3] > height - margin + tol
    )


def frame_validity(
    track: Track, n_frames: int, width: int, height: int, margin: float = 2.0
) -> List[bool]:
    """Per-frame mask: True where the subject is present and fully in frame."""
    return [
        (f in track.boxes) and not is_truncated(track.boxes[f], width, height, margin)
        for f in range(n_frames)
    ]


def valid_segments(mask: Sequence[bool], min_len: int = 10, max_gap: int = 3):
    """Contiguous valid (start, end) half-open segments, bridging gaps of up
    to ``max_gap`` invalid frames and dropping segments shorter than
    ``min_len``."""
    segs: List[List[int]] = []
    for i, ok in enumerate(mask):
        if not ok:
            continue
        if segs and i - segs[-1][1] <= max_gap + 1:
            segs[-1][1] = i + 1
        else:
            segs.append([i, i + 1])
    return [(a, b) for a, b in segs if b - a >= min_len]
