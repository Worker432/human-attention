"""Regression: eye features must not depend on head pose."""
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import numpy as np
import vision_camera as camera


class VisionTests(unittest.TestCase):
    def test_preview_converts_bottom_left_to_top_left(self):
        face = MagicMock()
        face.boundingBox.return_value = SimpleNamespace(
            origin=SimpleNamespace(x=.2, y=.3), size=SimpleNamespace(width=.4, height=.5))
        eye = np.array([[.1, .5], [.9, .5]])
        pupil = np.array([[.5, .6]])
        preview = camera.eye_preview(np.zeros((200, 200, 3), np.uint8), face,
                                     eye, pupil, eye, pupil)
        self.assertIsNotNone(preview)
        # Red pupil is higher than the green contour in the top-left image.
        red_y = np.where(preview[:, :, 0] == 255)[0].mean()
        green_y = np.where(preview[:, :, 1] == 255)[0].mean()
        self.assertLess(red_y, green_y)

    def test_eye_features_work_without_any_head_pose(self):
        detected = MagicMock()
        detected.confidence.return_value = .99
        detected.yaw.return_value = None
        detected.pitch.return_value = None
        detected.roll.return_value = None
        detector = MagicMock()
        detector.results.return_value = [detected]
        face = MagicMock()
        face.pitch.return_value = None
        face.boundingBox.return_value = SimpleNamespace(
            origin=SimpleNamespace(x=.2, y=.3), size=SimpleNamespace(width=.4, height=.5))
        request = MagicMock()
        request.results.return_value = [face]
        handler = MagicMock()
        handler.performRequests_error_.return_value = (True, None)
        eye = np.array([[0, .1], [.5, 0], [1, .1], [.5, .2]])
        pupil = np.array([[.5, .1]])
        with patch.object(camera.Vision, "VNDetectFaceRectanglesRequest") as detector_type, \
             patch.object(camera.Vision, "VNDetectFaceLandmarksRequest") as landmarks_type, \
             patch.object(camera.Vision, "VNImageRequestHandler") as handler_type, \
             patch.object(camera, "points", side_effect=[eye, pupil, eye, pupil]):
            detector_type.alloc.return_value.init.return_value = detector
            landmarks_type.alloc.return_value.init.return_value = request
            handler_type.alloc.return_value.initWithData_options_.return_value = handler
            features, status = camera.extract(np.zeros((20, 20, 3), dtype=np.uint8))
        np.testing.assert_allclose(features, [0, 0, .2, 0, 0, .2])
        detected.yaw.assert_not_called()
        detected.pitch.assert_not_called()
        detected.roll.assert_not_called()
        self.assertEqual(len(features), 6)
        detector.setRevision_.assert_called_once_with(3)
        request.setInputFaceObservations_.assert_called_once_with([detected])
        self.assertEqual(status, "Лицо и зрачки видны")
