"""Aggregation sizes, exclusive object shares, folder exports and resume."""
import hashlib
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from gaze_analysis import atomic_json
from scanpath_prediction import assign_steps, prediction_file, run_variants
from prediction_metrics import object_attention
from predict_app import PredictionPanel
from batch_prediction import BatchPredictionPanel, image_files
from attention_heatmap import HeatmapDensity, route_heatmap
from scanpath_prediction import load_prediction
from PySide6.QtCore import QProcess


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def cache(self, image_path, folder, count=100):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(320, 240, QImage.Format.Format_RGB32)
        image.fill(0xFF88AABB)
        image.save(str(image_path), 'PNG')
        folder.mkdir(parents=True, exist_ok=True)
        heat = QImage(32, 24, QImage.Format.Format_ARGB32)
        heat.fill(0x44880000)
        heat.save(str(folder/'heat.png'))
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        boxes = [dict(object_id=1, class_name='person', xyxy=[0, 0, 160, 240]),
                 dict(object_id=2, class_name='bottle', xyxy=[100, 0, 240, 240])]
        result = dict(schema_version=1, data_source='model_prediction', image_sha256=sha,
            image_width=320, image_height=240, objects=boxes,
            fixations=assign_steps([(20, 20), (120, 20), (300, 20)], boxes),
            heatmaps=['heat.png']*3)
        for seed in range(42, 42+count):
            atomic_json(prediction_file(folder, sha, 3, seed), dict(result, parameters=dict(steps=3, seed=seed)))
        return dict(result, parameters=dict(steps=3, seed=42))

    def test_object_shares_do_not_double_count_overlaps_and_use_all_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.cache(root/'banner.png', root/'cache', 10)
            stats = object_attention([result]*10)
            self.assertEqual(stats['total_points'], 30)
            self.assertEqual([r['count'] for r in stats['rows']], [10, 0, 10, 10])
            self.assertAlmostEqual(sum(r['share'] for r in stats['rows']), 1.)
            changed = dict(result, image_sha256='another')
            with self.assertRaises(ValueError):
                object_attention([result, changed])

    def test_main_window_uses_fixed_30_variants_without_quantity_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.cache(root/'banner.png', root/'cache')
            panel = PredictionPanel()
            panel.output_folder = root/'cache'
            panel.steps.setValue(3)
            panel.set_image(root/'banner.png')
            try:
                panel.open_gallery()
                self.qt.processEvents()
                self.assertTrue(panel.aggregate_complete)
                self.assertEqual(panel.object_metrics['variants'], 30)
                self.assertEqual(panel.object_metrics['total_points'], 90)
                self.assertEqual(panel.object_table.rowCount(), 4)
                self.assertFalse(hasattr(panel, 'variant_count'))
                panel.open_gallery()
                self.qt.processEvents()
                self.assertTrue(panel.aggregate_complete)
                self.assertEqual(panel.object_metrics['variants'], 30)
                self.assertIsNone(panel.aggregate_job)
                self.assertEqual(panel.heatmap_image().width(), 320)
            finally:
                panel.close()

    def test_folder_exports_nested_images_csv_and_manifest_and_reports_bad_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination, cache = root/'input', root/'output', root/'cache'
            self.cache(source/'a/banner.png', cache, 30)
            self.cache(source/'b/banner.png', cache, 30)
            (source/'bad.jpg').write_bytes(b'not an image')
            dialog = BatchPredictionPanel(cache, steps=3)
            dialog.source, dialog.destination = source, destination
            dialog.configure()
            try:
                self.assertEqual(len(dialog.files), 3)
                dialog.start()
                for _ in range(30):
                    self.qt.processEvents()
                    if not dialog.running:
                        break
                self.assertFalse(dialog.running)
                manifest = json.loads((dialog.run_folder/'manifest.json').read_text())
                self.assertEqual(manifest['state'], 'completed')
                self.assertEqual([r['status'] for r in manifest['records']].count('completed'), 2)
                self.assertEqual([r['status'] for r in manifest['records']].count('failed'), 1)
                for sub in ('a', 'b'):
                    output = QImage(str(dialog.run_folder/sub/'banner.png.heatmap.png'))
                    self.assertEqual((output.width(), output.height()), (320, 240))
                import csv
                with (dialog.run_folder/'attention.csv').open(encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 8)
                self.assertEqual({r['image'] for r in rows}, {'a/banner.png', 'b/banner.png'})
                self.assertTrue(all(r['data_source']=='model_prediction' for r in rows))
            finally:
                dialog.close()

    def test_input_scan_excludes_export_folder_but_allows_destination_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'input'
            source.mkdir()
            (source/'banner.png').write_bytes(b'x')
            output = source/'exports'
            output.mkdir()
            (output/'other.png').write_bytes(b'x')
            self.assertEqual(len(image_files(source, output)), 1)
            self.assertEqual(len(image_files(source, root)), 2)

    def test_lightweight_cache_can_display_aggregate_but_requires_build_for_step_maps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.cache(root/'banner.png', root/'cache', 30)
            sha = first['image_sha256']
            for seed in range(42, 72):
                file = prediction_file(root/'cache', sha, 3, seed)
                value = json.loads(file.read_text())
                value['heatmaps'] = []
                value['parameters']['step_heatmaps'] = False
                atomic_json(file, value)
            self.assertIsNone(load_prediction(file, sha))
            self.assertIsNotNone(load_prediction(file, sha, require_heatmaps=False))
            panel = PredictionPanel()
            panel.output_folder = root/'cache'
            panel.steps.setValue(3)
            panel.set_image(root/'banner.png')
            try:
                panel.open_gallery()
                self.qt.processEvents()
                self.assertTrue(panel.aggregate_complete)
                self.assertFalse(panel.heatmap_image().isNull())
                panel.heat_mode.setCurrentIndex(1)
                panel.canvas.grab()
                self.assertTrue(panel.heatmap_image().isNull())
                self.assertFalse(panel.predict_button.isHidden())
                self.assertIsNone(panel.worker)
            finally:
                panel.close()

    def test_incremental_heatmap_matches_one_pass_density(self):
        points = [dict(x=20., y=20.), dict(x=210., y=100.), dict(x=100., y=200.)]
        density = HeatmapDensity(320, 240)
        density.add(points[:1])
        density.add(points[1:])
        self.assertEqual(density.image(), route_heatmap(points, 320, 240))

    def test_main_cancel_stops_background_job_and_unlocks_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.cache(root/'banner.png', root/'cache', 30)
            panel = PredictionPanel()
            panel.output_folder = root/'cache'
            panel.steps.setValue(3)
            panel.set_image(root/'banner.png')
            from heatmap_gallery import HeatmapGallery
            import sys
            job = HeatmapGallery(panel.image_path, first['image_sha256'], panel.canvas.image,
                                 root/'cache', steps=3, parent=panel, variants=30)
            worker = QProcess(job)
            job.worker = worker
            panel.aggregate_job = job
            worker.start(sys.executable, ['-c', 'import time; time.sleep(30)'])
            self.assertTrue(worker.waitForStarted(3000))
            try:
                panel.set_enabled()
                self.assertFalse(panel.cancel_button.isHidden())
                panel.cancel_prediction()
                self.assertEqual(worker.state(), QProcess.ProcessState.NotRunning)
                self.assertIsNone(panel.aggregate_job)
                self.assertTrue(panel.open_button.isEnabled())
                self.assertTrue(panel.cancel_button.isHidden())
            finally:
                panel.close()

    def test_changing_parameters_layers_or_image_never_starts_aggregate_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.cache(root/'banner.png', root/'cache', 30)
            panel = PredictionPanel()
            panel.output_folder = root/'cache'
            panel.steps.setValue(3)
            panel.set_image(root/'banner.png')
            try:
                panel.open_gallery()
                self.qt.processEvents()
                self.assertTrue(panel.aggregate_complete)
                with patch.object(panel, 'ensure_aggregate') as start:
                    panel.steps.setValue(9)
                    panel.seed.setValue(99)
                    panel.opacity.setValue(45)
                    panel.heat_mode.setCurrentIndex(1)
                    panel.heat_mode.setCurrentIndex(2)
                    panel.heatmap.setChecked(False)
                    panel.heatmap.setChecked(True)
                    panel.set_image(root/'banner.png')
                    self.qt.processEvents()
                    start.assert_not_called()
                    self.assertIsNone(panel.worker)
                    self.assertIsNone(panel.aggregate_job)
                    self.assertFalse(panel.aggregate_complete)
                    self.assertTrue(panel.predict_button.isEnabled())
                    self.assertFalse(panel.predict_button.isHidden())
                    panel.predict_button.click()
                    start.assert_called_once()
            finally:
                panel.close()

    def test_cached_single_path_does_not_hide_build_button_for_missing_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.cache(root/'banner.png', root/'cache', 10)
            panel = PredictionPanel()
            panel.output_folder = root/'cache'
            panel.steps.setValue(3)
            panel.set_image(root/'banner.png')
            try:
                with patch.object(panel, 'ensure_aggregate') as start:
                    panel.heat_mode.setCurrentIndex(2)
                    panel.heatmap.setChecked(True)
                    panel.opacity.setValue(30)
                    self.qt.processEvents()
                    start.assert_not_called()
                    self.assertIsNotNone(panel.result)
                    self.assertFalse(panel.aggregate_complete)
                    self.assertFalse(panel.predict_button.isHidden())
            finally:
                panel.close()

    def test_batch_stop_and_resume_keeps_same_run_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.cache(root/'input/banner.png', root/'cache', 30)
            dialog = BatchPredictionPanel(root/'cache', steps=3)
            dialog.source, dialog.destination = root/'input', root/'output'
            dialog.configure()
            try:
                dialog.start()
                run_folder = dialog.run_folder
                dialog.stop()
                self.assertEqual(json.loads((run_folder/'manifest.json').read_text())['state'], 'stopped')
                dialog.start()
                for _ in range(30):
                    self.qt.processEvents()
                    if not dialog.running:
                        break
                self.assertFalse(dialog.running)
                self.assertEqual(dialog.run_folder, run_folder)
                self.assertEqual(json.loads((run_folder/'manifest.json').read_text())['state'], 'completed')
            finally:
                dialog.close()

    def test_variant_cli_reuses_runtime_and_wraps_seeds(self):
        args = Namespace(seed=999995, variants=10)
        calls = []
        def fake_run(arguments, runtime):
            calls.append((arguments.seed, id(runtime)))
            return Path(str(arguments.seed))
        with patch('scanpath_prediction.run', side_effect=fake_run), patch('builtins.print'):
            self.assertEqual(len(run_variants(args)), 10)
        self.assertEqual([seed for seed, _ in calls], list(range(999995, 1000000))+list(range(5)))
        self.assertEqual(len({runtime for _, runtime in calls}), 1)
        self.assertEqual(args.seed, 999995)


if __name__ == '__main__':
    unittest.main()
