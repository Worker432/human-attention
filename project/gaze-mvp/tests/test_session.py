"""Offscreen integration test with synthetic features; never opens a camera."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication
import app


class FakeCamera:
    def __init__(self, index):
        self.packet = None

    def start(self):
        pass

    def snapshot(self):
        return self.packet

    def requestInterruption(self):
        pass

    def wait(self):
        pass


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_calibration_validation_logging_loss_and_restart(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app, "Camera", FakeCamera):
            window = app.Window(SimpleNamespace(output=folder, camera=0, samples=10, smoothing=.15))
            try:
                for _ in range(39):
                    target = window.target()
                    window.collect_since = time.monotonic() - 2
                    # Independent target features with constant nuisance columns.
                    feature = np.array([*target, *([1.] * 4)])
                    for _ in range(10):
                        window.collect(feature, time.monotonic())
                self.assertEqual(window.mode, "tracking")
                self.assertEqual(len(window.train_x), 250)
                self.assertEqual(len(window.errors), 90)
                self.assertEqual(len(window.tune_x), 50)
                self.assertEqual(len(window.region_errors), 9)
                self.assertEqual(len(window.model_scores), 10)
                now = time.monotonic()
                window.camera.packet = (1, feature, "valid", 1234567890., now)
                window.tick()
                self.assertIsNotNone(window.point)
                window.camera.packet = (2, None, "lost", 1234567891., now)
                window.tick()
                self.assertIsNone(window.point)
                window.camera.packet = (3, feature, "stale", 1234567892., now - 1)
                window.tick()
                self.assertIsNone(window.point)
                with (Path(folder)/"trajectory.csv").open() as recorded:
                    rows = list(csv.reader(recorded))
                self.assertEqual(rows[0], ["timestamp", "x", "y"])
                self.assertEqual(len(rows), 2)
                self.assertEqual(float(rows[1][0]), 1234567890.)
                metadata = json.loads((Path(folder)/"session.json").read_text())
                self.assertEqual(len(metadata["calibration_rounds"]), 1)
                self.assertTrue((Path(folder)/"calibration-1.npz").exists())
                window.show()
                self.qt.processEvents()  # Exercise QPainter in the offscreen backend.
                from PySide6.QtGui import QKeyEvent
                from PySide6.QtCore import QEvent, Qt
                window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_C,
                                               Qt.KeyboardModifier.NoModifier))
                self.assertEqual(window.mode, "calibration")
                self.assertIsNone(window.model)
                self.assertEqual(window.train_x, [])
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
