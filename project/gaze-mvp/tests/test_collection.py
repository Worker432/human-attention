"""Resume, coordinate mapping, real presentation transitions and raw sample tests."""
import csv
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from beam_source import BeamSource
from collection_store import CollectionStore, export_dataset, image_coordinates
from collect_app import Collector


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def make_images(self, folder):
        root = Path(folder) / "images"
        root.mkdir()
        for i in range(4):
            image = QImage(640, 480, QImage.Format.Format_RGB32)
            image.fill(0xFF7799AA)
            image.save(str(root / f"{i}.png"))
        return root

    def test_resume_never_repeats_including_crash_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_images(folder)
            output = Path(folder) / "recording"
            store = CollectionStore(output, root)
            store.begin_run({})
            first = store.next_image()["id"]
            trial = store.begin_trial(first, "hash", {}, 100., 10.)
            store.add_samples([(trial, 1, 100.1, 10.1, 50., .1, "HIGH", 1, 1, .5, .5, 100., 50., 40., 20.)])
            store.close()  # Represents an interrupted process without finishing its trial.
            store = CollectionStore(output, root)
            self.assertNotEqual(first, store.next_image()["id"])
            self.assertEqual(store.db.execute("SELECT status FROM trials").fetchone()[0], "interrupted_crash")
            self.assertEqual(store.progress(), (4, 1, 0))
            with self.assertRaises(ValueError):
                CollectionStore(output, root)
            export_dataset(output / "dataset.sqlite3", Path(folder) / "export")
            with (Path(folder) / "export/samples.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["image_x"], "40.0")
            store.close()
            with self.assertRaises(ValueError):
                CollectionStore(output, root, duration=7.)

    def test_letterboxing_preserves_off_image_and_unclipped_points(self):
        sx, sy, ix, iy, inside = image_coordinates((.5, .5), (1001, 801), (100, 100, 800, 600), (1600, 1200))
        self.assertEqual((sx, sy, ix, iy, inside), (500., 400., 800., 600., True))
        self.assertLess(image_coordinates((0., .5), (1001, 801), (100, 100, 800, 600), (1600, 1200))[2], 0)
        self.assertFalse(image_coordinates((1.1, .5), (1001, 801), (100, 100, 800, 600), (1600, 1200))[4])

    def test_complete_pause_tracking_loss_close_and_next_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_images(folder)
            output = Path(folder) / "recording"
            store = CollectionStore(output, root, duration=5.)
            window = Collector(store, self.qt.primaryScreen(), SimpleNamespace(duration=5., break_every=50, ui_only=True))
            window.timer.stop()
            try:
                window.show()
                self.qt.processEvents()
                window.start()
                self.assertEqual(window.mode, "ready")  # Cannot record without Beam.
                window.last_good = time.monotonic()
                window.start()
                self.qt.processEvents()
                window.last_good = time.monotonic()
                window.interval_start = time.monotonic() - 1.1
                window.tick()
                self.qt.processEvents()
                self.assertEqual(window.mode, "viewing")
                first_id = window.entry["id"]
                trial = window.trial
                now = time.monotonic()
                window.source.sdk_timestamp = 123.
                window.source.publish((.5, .5), "HIGH", "ok", 1000., now)
                window.source.publish((-1., .5), "LOW", "low", 1000.01, now + .001)
                window.source.publish(None, "LOST_TRACKING", "lost", 1000.02, now + .002)
                window.tick()
                rows = store.db.execute("SELECT * FROM samples WHERE trial_id=?", (trial,)).fetchall()
                self.assertEqual(len(rows), 3)
                self.assertEqual([r["valid"] for r in rows], [1, 0, 0])
                self.assertLess(rows[1]["image_x"], 0)
                self.assertIsNone(rows[2]["image_x"])
                window.onset -= 6
                window.last_good = time.monotonic()
                window.tick()
                self.assertEqual(store.progress(), (4, 1, 1))
                window.last_good = time.monotonic()
                window.interval_start = time.monotonic() - 1.1
                window.tick()
                self.qt.processEvents()
                self.assertEqual(window.mode, "viewing")
                self.assertNotEqual(window.entry["id"], first_id)
                window.last_good = time.monotonic() - 1
                window.tick()
                self.assertEqual(window.mode, "paused")
                self.assertEqual(store.progress(), (4, 2, 1))
                window.grab().save(str(Path(folder) / "preview.png"))
            finally:
                window.close()
            store = CollectionStore(output, root)
            self.assertEqual(store.progress(), (4, 2, 1))
            store.close()

    def test_source_queue_does_not_lose_samples_between_ui_ticks(self):
        source = BeamSource((0, 0, 100, 100), capture_queue=True)
        for i in range(20):
            source.sdk_timestamp = i
            source.publish((.1, .2), "HIGH", "ok")
        self.assertEqual(len(source.drain()), 20)
        self.assertEqual(source.drain(), [])
        self.assertEqual(len(source.snapshot()), 6)  # Existing overlay remains compatible.


if __name__ == "__main__":
    unittest.main()
