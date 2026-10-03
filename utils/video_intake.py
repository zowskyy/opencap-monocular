"""Video intake helpers for arbitrary single-view videos.

Provides metadata probing/validation, shot-boundary detection and fallback
camera intrinsics estimation (from a field-of-view prior) so that videos which
are not from a calibrated smartphone can be processed.
"""

import math
import os
import pickle
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

DEFAULT_HFOV_DEG = 63.0  # typical smartphone main camera, long side
MIN_SIDE_PX = 240
MIN_DURATION_S = 0.5
MIN_FPS = 5.0


@dataclass
class VideoInfo:
    width: int
    height: int  # displayed (rotation-corrected) height
    fps: float
    n_frames: int
    duration_s: float
    rotation: int = 0  # degrees, display rotation tag (0/90/180/270)
    is_variable_fps: bool = False
    warnings: List[str] = field(default_factory=list)


class VideoValidationError(ValueError):
    pass


def _parse_rate(rate: Optional[str]) -> float:
    if not rate:
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_probe(meta: dict) -> VideoInfo:
    """Build VideoInfo from an ``ffmpeg.probe`` style dict."""
    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise VideoValidationError("No video stream found")
    s = streams[0]
    w, h = int(s.get("width", 0)), int(s.get("height", 0))

    rotation = 0
    tags = s.get("tags", {}) or {}
    if "rotate" in tags:
        rotation = int(float(tags["rotate"])) % 360
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(-float(sd["rotation"])) % 360
    if rotation in (90, 270):
        w, h = h, w

    avg = _parse_rate(s.get("avg_frame_rate"))
    real = _parse_rate(s.get("r_frame_rate"))
    fps = avg or real
    duration = float(s.get("duration") or meta.get("format", {}).get("duration") or 0)
    try:
        n_frames = int(s["nb_frames"])
    except (KeyError, ValueError, TypeError):
        n_frames = int(round(duration * fps)) if fps else 0
    if not duration and fps and n_frames:
        duration = n_frames / fps

    info = VideoInfo(w, h, fps, n_frames, duration, rotation)
    info.is_variable_fps = bool(avg and real and abs(avg - real) / real > 0.02)
    return info


def probe_video(video_path: str) -> VideoInfo:
    import ffmpeg

    return parse_probe(ffmpeg.probe(video_path))


def validate_video_info(info: VideoInfo) -> VideoInfo:
    """Raise on unusable videos; attach warnings for borderline ones."""
    if info.width <= 0 or info.height <= 0:
        raise VideoValidationError("Video has invalid resolution")
    if min(info.width, info.height) < MIN_SIDE_PX:
        raise VideoValidationError(
            f"Resolution {info.width}x{info.height} is too low (min side {MIN_SIDE_PX}px)"
        )
    if info.duration_s < MIN_DURATION_S or info.n_frames < 2:
        raise VideoValidationError("Video is too short to process")
    if info.fps and info.fps < MIN_FPS:
        raise VideoValidationError(f"Frame rate {info.fps:.1f} fps is too low")
    if info.is_variable_fps:
        info.warnings.append(
            "Variable frame rate detected; resample to constant fps for best results"
        )
    if info.fps > 125:
        info.warnings.append(f"Very high frame rate ({info.fps:.0f} fps)")
    return info


def normalization_scale(info: VideoInfo, max_long_side: int = 1920) -> float:
    """Downscale factor (<=1) that brings the long side to ``max_long_side``."""
    long_side = max(info.width, info.height)
    return min(1.0, max_long_side / long_side) if long_side else 1.0


# ---------------------------------------------------------------- shots ----

def _hist_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Half L1 distance between two normalised histograms, in [0, 1]."""
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def detect_shot_boundaries(
    histograms: Sequence[Sequence[float]],
    threshold: float = 0.5,
    min_shot_len: int = 5,
) -> List[int]:
    """Return frame indices at which a new shot starts (excluding frame 0).

    ``histograms`` are per-frame normalised colour histograms. A cut is a jump
    in histogram distance above ``threshold``; cuts that would create shots
    shorter than ``min_shot_len`` frames are ignored.
    """
    cuts: List[int] = []
    last = 0
    for i in range(1, len(histograms)):
        if (
            _hist_distance(histograms[i - 1], histograms[i]) > threshold
            and i - last >= min_shot_len
        ):
            cuts.append(i)
            last = i
    if cuts and len(histograms) - cuts[-1] < min_shot_len:
        cuts.pop()
    return cuts


def split_into_shots(
    n_frames: int, cuts: Sequence[int], min_shot_len: int = 5
) -> List[Tuple[int, int]]:
    """Convert cut indices to half-open (start, end) frame ranges."""
    edges = [0, *[c for c in cuts if 0 < c < n_frames], n_frames]
    shots = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_shot_len]
    return shots


def compute_frame_histograms(video_path: str, bins: int = 16, stride: int = 1):
    """Per-frame normalised HSV histograms (requires OpenCV)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    hists = []
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                small = cv2.resize(frame, (64, 64))
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
                h = h.flatten()
                hists.append((h / max(h.sum(), 1e-8)).tolist())
            idx += 1
    finally:
        cap.release()
    return hists


# ----------------------------------------------------------- intrinsics ----

def intrinsics_from_fov(
    width: int, height: int, hfov_deg: float = DEFAULT_HFOV_DEG
) -> dict:
    """Pinhole intrinsics (zero distortion) assuming a horizontal FOV along
    the image's long side, principal point at the image centre."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    if not 10 <= hfov_deg <= 170:
        raise ValueError("Field of view must be between 10 and 170 degrees")
    long_side = max(width, height)
    f = (long_side / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}


def focal_from_exif(focal_35mm_mm: float, width: int, height: int) -> float:
    """Focal length in pixels from a 35mm-equivalent focal length."""
    if focal_35mm_mm <= 0:
        raise ValueError("Focal length must be positive")
    return focal_35mm_mm / 36.0 * max(width, height)


def write_fallback_intrinsics(
    info: VideoInfo,
    out_path: str,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    focal_px: Optional[float] = None,
) -> str:
    """Write an OpenCap-style intrinsics pickle for an arbitrary video.

    The pipeline expects portrait-oriented calibrations (``imageSize`` is
    ``[[h], [w]]`` with h >= w), so the pinhole model is built for the portrait
    version of the frame.
    """
    import numpy as np

    ph, pw = max(info.width, info.height), min(info.width, info.height)
    k = intrinsics_from_fov(pw, ph, hfov_deg)
    if focal_px:
        k["fx"] = k["fy"] = float(focal_px)
    data = {
        "intrinsicMat": np.array(
            [[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=float
        ),
        "distortion": np.zeros((1, 5)),
        "imageSize": np.array([[ph], [pw]]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    return out_path
