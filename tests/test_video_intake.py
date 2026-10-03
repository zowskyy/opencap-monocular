import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import video_intake as vi
from utils import subject_selection as ss


class TestIntake(unittest.TestCase):
    def test_parse_probe_rotation(self):
        meta = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                    "nb_frames": "90",
                    "duration": "3.0",
                    "tags": {"rotate": "90"},
                }
            ]
        }
        info = vi.validate_video_info(vi.parse_probe(meta))
        self.assertEqual((info.width, info.height), (1080, 1920))
        self.assertFalse(info.is_variable_fps)

    def test_validate_rejects(self):
        with self.assertRaises(vi.VideoValidationError):
            vi.validate_video_info(vi.VideoInfo(100, 100, 30, 100, 3.0))
        with self.assertRaises(vi.VideoValidationError):
            vi.validate_video_info(vi.VideoInfo(640, 480, 30, 1, 0.03))

    def test_vfr_warning(self):
        info = vi.VideoInfo(640, 480, 30, 100, 3.0, is_variable_fps=True)
        self.assertTrue(vi.validate_video_info(info).warnings)

    def test_shots(self):
        a, b = [1.0, 0.0], [0.0, 1.0]
        hists = [a] * 10 + [b] * 10
        cuts = vi.detect_shot_boundaries(hists)
        self.assertEqual(cuts, [10])
        self.assertEqual(vi.split_into_shots(20, cuts), [(0, 10), (10, 20)])
        self.assertEqual(vi.detect_shot_boundaries([a] * 5 + [b] * 2), [])

    def test_intrinsics(self):
        k = vi.intrinsics_from_fov(1080, 1920, 60)
        self.assertAlmostEqual(k["fx"], 960 / math.tan(math.radians(30)))
        self.assertEqual((k["cx"], k["cy"]), (540, 960))
        with self.assertRaises(ValueError):
            vi.intrinsics_from_fov(10, 10, 5)

    def test_scale(self):
        self.assertEqual(vi.normalization_scale(vi.VideoInfo(1920, 1080, 30, 1, 1)), 1.0)
        self.assertEqual(vi.normalization_scale(vi.VideoInfo(3840, 2160, 30, 1, 1)), 0.5)


class TestSubject(unittest.TestCase):
    def setUp(self):
        self.a = ss.Track(1, {f: (0, 0, 100, 200) for f in range(0, 20)})
        self.b = ss.Track(2, {f: (300, 0, 340, 80) for f in range(0, 40)})
        self.c = ss.Track(3, {f: (5, 0, 105, 200) for f in range(25, 40)})

    def test_select(self):
        self.assertEqual(ss.select_target_track([self.a, self.b]).track_id, 1)
        self.assertEqual(ss.select_target_track([self.a, self.b], track_id=2).track_id, 2)
        self.assertEqual(
            ss.select_target_track([self.a, self.b], click=(320, 40, 5)).track_id, 2
        )
        with self.assertRaises(ValueError):
            ss.select_target_track([self.a], click=(900, 900, 5))
        with self.assertRaises(ValueError):
            ss.select_target_track([])

    def test_reid(self):
        m = ss.merge_tracks_reid(self.a, [self.b, self.c])
        self.assertIn(30, m.boxes)
        self.assertEqual(m.track_id, 1)
        far = ss.merge_tracks_reid(self.a, [self.b])
        self.assertEqual(far.frames[-1], 19)

    def test_validity(self):
        t = ss.Track(1, {0: (0, 10, 50, 90), 1: (10, 10, 50, 90), 2: (10, 10, 50, 90)})
        mask = ss.frame_validity(t, 4, 100, 100)
        self.assertEqual(mask, [False, True, True, False])
        self.assertEqual(ss.valid_segments([True] * 5 + [False] + [True] * 5), [(0, 11)])
        self.assertEqual(ss.valid_segments([True] * 3, min_len=10), [])


if __name__ == "__main__":
    unittest.main()
