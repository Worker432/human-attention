"""Prediction provenance, pixel mapping, cache integrity and native playback."""
import hashlib
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import sys
import unittest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess, QAbstractAnimation
from gaze_analysis import atomic_json
from scanpath_prediction import assign_steps, load_prediction, map_pixel, prediction_file, prediction_key
from predict_app import PredictionPanel
from attention_heatmap import route_heatmap
from heatmap_gallery import HeatmapGallery


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def make_prediction(self, folder):
        root = Path(folder)
        image = QImage(320, 240, QImage.Format.Format_RGB32)
        image.fill(0xFF88AABB)
        image_path = root / "banner.png"
        image.save(str(image_path))
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        heat = QImage(32, 24, QImage.Format.Format_ARGB32)
        heat.fill(0x55880000)
        heat.save(str(root / "heat.png"))
        result = dict(schema_version=1, data_source="model_prediction", image_sha256=sha,
            image_width=320, image_height=240, parameters=dict(steps=2, seed=42),
            objects=[dict(object_id=1, class_name="bottle", confidence=.8, xyxy=[0, 0, 50, 50])],
            fixations=assign_steps([(20., 20.), (200., 100.)], [dict(object_id=1, class_name="bottle", xyxy=[0, 0, 50, 50])]),
            heatmaps=["heat.png", "heat.png"])
        output = prediction_file(root, sha, 2, 42)
        atomic_json(output, result)
        return image_path, sha, output, result

    def test_mapping_and_ambiguous_boxes(self):
        self.assertEqual(map_pixel(0, 0, (100, 100), (100, 100)), (0., 0.))
        self.assertEqual(map_pixel(99, 99, (100, 100), (200, 200)), (198.5, 198.5))
        boxes = [dict(object_id=i, class_name="person", xyxy=[0, 0, 100, 100]) for i in (1, 2)]
        points = assign_steps([(30, 30), (200, 200)], boxes)
        self.assertEqual(points[0]["assignment"], "ambiguous")
        self.assertIsNone(points[0]["object_id"])
        self.assertEqual(points[1]["assignment"], "outside")

    def test_heatmap_follows_image_coordinates_and_is_transparent_away_from_points(self):
        heat = route_heatmap([dict(x=100., y=50.)], 200, 100)
        self.assertEqual((heat.width(), heat.height()), (200, 100))
        self.assertGreater(heat.pixelColor(100, 50).alpha(), 200)
        self.assertEqual(heat.pixelColor(0, 0).alpha(), 0)
        self.assertTrue(route_heatmap([], 200, 100).isNull())

    def test_cache_identity_and_source_are_checked(self):
        self.assertNotEqual(prediction_key("sha", seed=42), prediction_key("sha", seed=43))
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            self.assertIsNotNone(load_prediction(file, sha))
            self.assertIsNone(load_prediction(file, "other"))
            atomic_json(file, dict(result, data_source="human_recording"))
            self.assertIsNone(load_prediction(file, sha))
            atomic_json(file, result)
            (Path(folder)/"heat.png").unlink()
            self.assertIsNone(load_prediction(file, sha))

    def test_native_demo_load_play_seek_layers_and_parameter_change(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            panel = PredictionPanel()
            panel.output_folder = Path(folder)
            panel.steps.setValue(2)
            try:
                panel.set_image(image)
                self.assertIsNotNone(panel.result)
                self.assertEqual(panel.slider.maximum(), 2)
                self.assertEqual(panel.sequence.rowCount(), 2)
                self.assertTrue(panel.advanced.isHidden())
                panel.sequence.setCurrentCell(0, 2)
                self.assertEqual(panel.canvas.step, 1)
                self.assertIn("Бутылка", panel.step_detail.text())
                panel.slider.setValue(2)
                self.assertEqual(panel.sequence.currentRow(), 1)
                panel.play()
                self.assertEqual(panel.slider.value(), 0)
                panel.advance()
                self.assertEqual(panel.canvas.step, 1)
                panel.advance()
                self.assertFalse(panel.timer.isActive())
                panel.boxes.setChecked(False)
                panel.heatmap.setChecked(True)
                self.assertFalse(panel.canvas.show_boxes)
                self.assertTrue(panel.canvas.show_heatmap)
                panel.route.setChecked(False)
                panel.opacity.setValue(40)
                self.assertFalse(panel.canvas.show_route)
                self.assertEqual(panel.canvas.heat_opacity, .4)
                clean = panel.heatmap_image()
                self.assertEqual((clean.width(), clean.height()), (320, 240))
                self.assertNotEqual(clean.pixelColor(20, 20), panel.canvas.image.pixelColor(20, 20))
                self.assertEqual(clean.pixelColor(319, 239), panel.canvas.image.pixelColor(319, 239))
                panel.show()
                self.qt.processEvents()
                panel.canvas.grab()
                panel.seed.setValue(43)
                self.assertIsNone(panel.result)
                self.assertFalse(panel.play_button.isEnabled())
                self.assertTrue(panel.predict_button.isEnabled())
            finally:
                panel.close()

    def test_process_completion_restores_controls_and_loads_result(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            panel = PredictionPanel()
            panel.output_folder = Path(folder)
            panel.steps.setValue(2)
            try:
                panel.set_image(image)
                panel.predict()
                self.assertIsNone(panel.worker)  # Existing prediction is immediate.
                panel.worker = QProcess(panel)
                panel.set_enabled()
                self.assertFalse(panel.predict_button.isEnabled())
                panel.prediction_finished(0, QProcess.ExitStatus.NormalExit)
                self.assertIsNone(panel.worker)
                self.assertTrue(panel.predict_button.isEnabled())
                self.assertTrue(panel.play_button.isEnabled())
                self.assertEqual(panel.sequence.rowCount(), 2)
            finally:
                panel.close()

    def test_playback_interpolates_without_changing_prediction_and_pause_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            panel = PredictionPanel()
            panel.output_folder = Path(folder)
            panel.steps.setValue(2)
            panel.set_image(image)
            original = json.dumps(panel.result, sort_keys=True)
            try:
                panel.play()
                panel.advance()
                panel.advance()
                motion = panel.canvas.motion
                self.assertEqual(motion.state(), QAbstractAnimation.State.Running)
                duration = motion.duration()
                motion.setCurrentTime(duration//2)
                point = panel.canvas.current_point()
                self.assertAlmostEqual(point.x(), 110., delta=.2)
                self.assertAlmostEqual(point.y(), 60., delta=.2)
                panel.stop()
                self.assertEqual(motion.state(), QAbstractAnimation.State.Paused)
                self.assertAlmostEqual(panel.canvas.current_point().x(), 110., delta=.2)
                panel.play()
                self.assertEqual(motion.state(), QAbstractAnimation.State.Running)
                motion.setCurrentTime(duration)
                self.assertEqual(panel.canvas.current_point().x(), 200.)
                self.assertFalse(panel.timer.isActive())
                self.assertEqual(panel.play_button.text(), 'Смотреть маршрут')
                self.assertEqual(json.dumps(panel.result, sort_keys=True), original)
                panel.slider.setValue(1)
                self.assertEqual(panel.canvas.current_point().x(), 20.)
            finally:
                panel.close()

    def test_playback_chains_segments_without_dwell_at_constant_speed(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            panel = PredictionPanel()
            panel.output_folder = Path(folder)
            panel.steps.setValue(2)
            panel.set_image(image)
            result['fixations'].append(dict(result['fixations'][-1], step=3, x=290., y=100.))
            try:
                panel.show_result(result, folder)
                panel.slider.setValue(1)
                panel.advance()
                motion = panel.canvas.motion
                first_duration = motion.duration()
                motion.setCurrentTime(first_duration)
                self.assertEqual(panel.canvas.step, 3)
                self.assertEqual(motion.state(), QAbstractAnimation.State.Running)
                second_duration = motion.duration()
                self.assertAlmostEqual(first_duration/second_duration, (180**2+80**2)**.5/90, delta=.01)
                motion.setCurrentTime(second_duration//4)
                self.assertAlmostEqual(panel.canvas.current_point().x(), 222.5, delta=.3)
                motion.setCurrentTime(second_duration)
                self.assertEqual(motion.state(), QAbstractAnimation.State.Stopped)
                self.assertEqual(panel.canvas.current_point().x(), 290.)
            finally:
                panel.close()

    def test_ten_cached_variants_export_one_pooled_heatmap_at_source_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            for seed in range(42, 52):
                variant = dict(result, parameters=dict(steps=2, seed=seed))
                atomic_json(prediction_file(folder, sha, 2, seed), variant)
            gallery = HeatmapGallery(image, sha, QImage(str(image)), folder, steps=2)
            try:
                gallery.start_next()
                self.assertEqual(len(gallery.results), 10)
                self.assertIsNone(gallery.worker)
                self.assertTrue(gallery.save_button.isEnabled())
                combined = gallery.combined_image()
                self.assertEqual((combined.width(), combined.height()), (320, 240))
                expected = route_heatmap([p for r in gallery.results.values() for p in r['fixations']], 320, 240)
                self.assertEqual(gallery.heat, expected)
                gallery.opacity.setValue(20)
                self.assertNotEqual(gallery.preview.image.pixelColor(20, 20), QImage(str(image)).pixelColor(20, 20))
            finally:
                gallery.close()

    def test_closing_gallery_stops_pending_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            gallery = HeatmapGallery(image, sha, QImage(str(image)), folder, steps=2)
            process = QProcess(gallery)
            gallery.worker = process
            process.start(sys.executable, ['-c', 'import time; time.sleep(30)'])
            self.assertTrue(process.waitForStarted(3000))
            gallery.close()
            self.assertTrue(gallery.closing)
            self.assertEqual(process.state(), QProcess.ProcessState.NotRunning)

    def test_combined_heatmap_is_shown_in_main_canvas_and_clean_export(self):
        with tempfile.TemporaryDirectory() as folder:
            image, sha, file, result = self.make_prediction(folder)
            for seed in range(42, 72):
                atomic_json(prediction_file(folder, sha, 2, seed),
                            dict(result, parameters=dict(steps=2, seed=seed)))
            panel = PredictionPanel()
            panel.output_folder = Path(folder)
            panel.steps.setValue(2)
            panel.set_image(image)
            try:
                panel.open_gallery()
                self.qt.processEvents()
                self.assertTrue(panel.aggregate_complete)
                self.assertIsNone(panel.aggregate_job)
                self.assertEqual(panel.canvas.heat_mode, 2)
                self.assertTrue(panel.canvas.show_heatmap)
                self.assertFalse(panel.canvas.show_route)
                self.assertFalse(panel.canvas.aggregate_heat.isNull())
                self.assertEqual(panel.heatmap_image().size(), panel.canvas.image.size())
                panel.slider.setValue(0)
                panel.canvas.grab()
                self.assertTrue(panel.aggregate_complete)
            finally:
                panel.close()


if __name__ == "__main__":
    unittest.main()
