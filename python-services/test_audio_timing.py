import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from audio_timing import fit_audio_to_samples, plan_timing_slots


class AudioTimingTests(unittest.TestCase):
    sr = 44100

    def tone(self, seconds=3):
        time = np.arange(round(seconds * self.sr)) / self.sr
        return (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

    def test_short_take_keeps_its_speed_and_pause(self):
        audio = self.tone(0.2)
        with patch("audio_timing.subprocess.run") as ffmpeg:
            fitted, speed, resampled = fit_audio_to_samples(audio, self.sr, self.sr)
        np.testing.assert_array_equal(fitted, audio)
        self.assertEqual(speed, 1)
        self.assertFalse(resampled)
        ffmpeg.assert_not_called()

    def test_real_ffmpeg_fits_long_takes_without_a_speed_ceiling(self):
        audio = self.tone()
        for ratio in [1.005, 1.26, 2.5, 6, 12]:
            with self.subTest(ratio=ratio):
                budget = int(len(audio) / ratio)
                fitted, speed, _ = fit_audio_to_samples(audio, self.sr, budget)
                self.assertGreater(len(fitted), 0)
                self.assertLessEqual(len(fitted), budget)
                self.assertGreaterEqual(speed, ratio)
                self.assertTrue(np.isfinite(fitted).all())
                self.assertGreater(float(np.max(np.abs(fitted))), 0.01)

    def test_very_short_window_still_produces_audio(self):
        fitted, speed, _ = fit_audio_to_samples(self.tone(), self.sr, 441)
        self.assertGreater(len(fitted), 0)
        self.assertLessEqual(len(fitted), 441)
        self.assertGreaterEqual(speed, 300)
        self.assertTrue(np.isfinite(fitted).all())

    def test_atempo_overshoot_resamples_the_complete_output_instead_of_cropping(self):
        # A marker in the tail would disappear if the output were simply sliced.
        stretched = np.zeros(2000, dtype=np.float32)
        stretched[1500:] = 0.25
        result = SimpleNamespace(returncode=0, stdout=stretched.tobytes(), stderr=b"")
        with patch("audio_timing.subprocess.run", return_value=result):
            fitted, _, resampled = fit_audio_to_samples(self.tone(), self.sr, 1000)
        self.assertEqual(len(fitted), 1000)
        self.assertTrue(resampled)
        self.assertGreater(float(np.mean(fitted[-200:])), 0.2)

    def test_empty_atempo_result_uses_the_complete_original(self):
        result = SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        with patch("audio_timing.subprocess.run", return_value=result):
            fitted, _, resampled = fit_audio_to_samples(self.tone(), self.sr, 1000)
        self.assertEqual(len(fitted), 1000)
        self.assertTrue(resampled)
        self.assertGreater(float(np.max(np.abs(fitted))), 0.01)

    def test_corrupt_audio_and_ffmpeg_errors_still_fail(self):
        for audio in [np.array([]), np.array([np.nan]), np.zeros((2, 2))]:
            with self.subTest(audio=audio), self.assertRaises(ValueError):
                fit_audio_to_samples(audio, self.sr, 1000)
        result = SimpleNamespace(returncode=1, stdout=b"", stderr=b"decoder failed")
        with patch("audio_timing.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                fit_audio_to_samples(self.tone(), self.sr, 1000)

    def test_same_speaker_never_pushes_the_next_phrase_or_video_boundary(self):
        segments = [
            {"speaker": "A", "start_ms": 0, "end_ms": 1500},
            {"speaker": "A", "start_ms": 1000, "end_ms": 2300},
        ]
        slots = plan_timing_slots(segments, self.sr, 2 * self.sr)
        self.assertEqual([(s.start_sample, s.end_sample) for s in slots], [
            (0, self.sr), (self.sr, 2 * self.sr),
        ])
        audio = self.tone(6)
        for slot in slots:
            fitted, speed, _ = fit_audio_to_samples(audio, self.sr, slot.end_sample - slot.start_sample)
            self.assertLessEqual(slot.start_sample + len(fitted), slot.end_sample)
            self.assertGreaterEqual(speed, 6)

    def test_source_overlap_over_250ms_does_not_shift_other_speakers(self):
        segments = [
            {"speaker": "A", "start_ms": 0, "end_ms": 1500},
            {"speaker": "B", "start_ms": 500, "end_ms": 2000},
        ]
        slots = plan_timing_slots(segments, self.sr, 2 * self.sr)
        self.assertEqual(slots[1].start_sample, self.sr // 2)
        self.assertEqual(slots[0].end_sample - slots[1].start_sample, self.sr)

    def test_duplicate_starts_share_the_window_without_shifting_later_dialogue(self):
        segments = [
            {"speaker": "A", "start_ms": 0, "end_ms": 1000},
            {"speaker": "A", "start_ms": 0, "end_ms": 1000},
            {"speaker": "A", "start_ms": 1000, "end_ms": 2000},
        ]
        slots = plan_timing_slots(segments, self.sr, 2 * self.sr)
        self.assertEqual([(s.start_sample, s.end_sample) for s in slots], [
            (0, self.sr // 2), (self.sr // 2, self.sr), (self.sr, 2 * self.sr),
        ])

    def test_sub_200ms_slots_are_not_artificially_extended(self):
        slots = plan_timing_slots([{"start_ms": 950, "end_ms": 1000}], self.sr, self.sr)
        self.assertEqual(slots[0].end_sample - slots[0].start_sample, 2205)

    def test_invalid_timestamps_remain_an_input_error(self):
        for start, end in [(1, 0), (0, float("nan")), (1000, 2000)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                plan_timing_slots([{"start_ms": start, "end_ms": end}], self.sr, self.sr)


if __name__ == "__main__":
    unittest.main()
