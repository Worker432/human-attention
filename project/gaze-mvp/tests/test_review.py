"""Playback and read-only access must preserve the original collection."""
import hashlib
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from collection_store import CollectionStore
from review_app import ReviewPanel, RecordingReader


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def recording(self, folder):
        images = Path(folder) / "images"
        images.mkdir()
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        image.fill(0xFF7799AA)
        image.save(str(images / "ad.png"))
        store = CollectionStore(Path(folder) / "data", images)
        store.begin_run({})
        trial = store.begin_trial(store.next_image()["id"],
                                  hashlib.sha256((images / "ad.png").read_bytes()).hexdigest(),
                                  dict(image_width=640, image_height=480), 100., 10.)
        rows = []
        for sequence, elapsed, valid, x, y in [(1, .1, 1, 100., 200.), (2, .2, 0, None, None), (3, .3, 1, 300., 250.)]:
            rows.append((trial, sequence, 100+elapsed, 10+elapsed, 50+elapsed,
                         elapsed, "HIGH" if valid else "LOST_TRACKING", valid,
                         valid, .5 if valid else None, .5 if valid else None, x, y, x, y))
        store.add_samples(rows)
        store.finish_trial(trial, "completed", 105., 5.)
        store.close()
        return Path(folder) / "data/dataset.sqlite3", images

    def test_read_only_playback_and_slider_preserve_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            database, images = self.recording(folder)
            reader = RecordingReader(database)
            db = reader.connect()
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    db.execute("DELETE FROM samples")
            finally:
                db.close()
            panel = ReviewPanel(database)
            try:
                panel.show()
                self.qt.processEvents()
                self.assertEqual(panel.list.count(), 1)
                self.assertEqual(len(panel.canvas.visible_points()), 2)
                self.assertIn("67%", panel.summary.text())
                panel.slider.setValue(150)
                self.assertEqual(len(panel.canvas.visible_points()), 1)
                panel.slider.setValue(5000)
                with patch("review_app.time.monotonic", return_value=10.):
                    panel.toggle_playback()
                self.assertTrue(panel.playing)
                self.assertEqual(panel.slider.value(), 0)
                with patch("review_app.time.monotonic", return_value=10.25):
                    panel.advance()
                self.assertEqual(panel.slider.value(), 250)
                self.assertEqual(len(panel.canvas.visible_points()), 1)
                with patch("review_app.time.monotonic", return_value=16.):
                    panel.advance()
                self.assertFalse(panel.playing)
                self.assertEqual(panel.slider.value(), 5000)
                panel.refresh()
                self.assertEqual(panel.list.count(), 1)
                panel.canvas.grab()  # Exercise drawing of points and gap handling.
            finally:
                panel.close()
            db = reader.connect()
            self.assertEqual(db.execute("SELECT count(*) FROM samples").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT count(*) FROM trials").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
            db.close()

    def test_changed_or_missing_image_never_shows_misaligned_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            database, images = self.recording(folder)
            (images / "ad.png").write_bytes(b"changed")
            panel = ReviewPanel(database)
            try:
                self.assertTrue(panel.canvas.image.isNull())
                self.assertFalse(panel.save_button.isEnabled())
                self.assertIn("изменён", panel.notice.text())
                (images / "ad.png").unlink()
                panel.refresh()
                self.assertIn("не найдено", panel.notice.text())
            finally:
                panel.close()

    def test_no_database_is_safe_and_does_not_create_one(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "missing.sqlite3"
            panel = ReviewPanel(database)
            try:
                self.assertFalse(panel.play_button.isEnabled())
                self.assertFalse(database.exists())
                self.assertEqual(panel.list.count(), 0)
            finally:
                panel.close()


if __name__ == "__main__":
    unittest.main()
