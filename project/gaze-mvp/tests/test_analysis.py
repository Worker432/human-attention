"""Object mapping preserves gaps, ambiguity, and source identity."""
from pathlib import Path
import csv
import tempfile
import unittest
from gaze_analysis import (ANALYSIS_VERSION, analysis_path, associate, atomic_json,
                           extract_fixations, load_analysis, source_digest, summarize)


def sample(t, x, y, valid=1):
    return dict(elapsed=t, image_x=x, image_y=y, valid=valid, inside_image=1)


class AnalysisTests(unittest.TestCase):
    def test_two_stable_clusters_create_object_transition(self):
        rows = [sample(i*.04, 100+i%2, 100) for i in range(6)]
        rows += [sample(.24+i*.04, 400+i%2, 200) for i in range(6)]
        fixes = extract_fixations(rows, 640, 480)
        self.assertEqual(len(fixes), 2)
        boxes = [dict(object_id=1, class_id=39, class_name="bottle", xyxy=[50, 50, 150, 150]),
                 dict(object_id=2, class_id=39, class_name="bottle", xyxy=[350, 150, 450, 250])]
        mapped = associate(fixes, boxes)
        self.assertEqual([f["object_id"] for f in mapped], [1, 2])
        transitions, stats = summarize(mapped, boxes)
        self.assertEqual(len(transitions), 1)
        self.assertEqual((transitions[0]["from_object"], transitions[0]["to_object"]), (1, 2))
        self.assertEqual([s["fixation_count"] for s in stats], [1, 1])

    def test_invalid_point_or_long_gap_prevents_false_transition(self):
        first = [sample(i*.04, 100, 100) for i in range(6)]
        second = [sample(.3+i*.04, 400, 200) for i in range(6)]
        boxes = [dict(object_id=1, class_id=0, class_name="a", xyxy=[50, 50, 150, 150]),
                 dict(object_id=2, class_id=1, class_name="b", xyxy=[350, 150, 450, 250])]
        for rows in (first+[sample(.25, None, None, 0)]+second,
                     first+[sample(1+i*.04, 400, 200) for i in range(6)]):
            mapped = associate(extract_fixations(rows, 640, 480), boxes)
            self.assertEqual(len(mapped), 2)
            self.assertEqual(summarize(mapped, boxes)[0], [])

    def test_overlap_and_background_are_not_assigned_arbitrarily(self):
        boxes = [dict(object_id=1, class_id=0, class_name="person", xyxy=[0, 0, 200, 300]),
                 dict(object_id=2, class_id=1, class_name="bottle", xyxy=[50, 50, 100, 100])]
        result = associate([dict(x=75, y=75), dict(x=400, y=300)], boxes)
        self.assertEqual(result[0]["assignment"], "ambiguous")
        self.assertEqual(result[0]["candidate_ids"], [1, 2])
        self.assertIsNone(result[0]["object_id"])
        self.assertEqual(result[1]["assignment"], "outside")

    def test_short_or_moving_points_are_not_fixations(self):
        self.assertEqual(extract_fixations([sample(0, 10, 10), sample(.03, 10, 10)], 640, 480), [])
        self.assertEqual(extract_fixations([sample(i*.04, i*100, 10) for i in range(6)], 640, 480), [])

    def test_cache_invalidates_when_source_changes(self):
        trial = dict(id="a"*32, status="completed", image_sha256="sha", geometry="{}")
        rows = [sample(0, 100, 100)]
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "dataset.sqlite3"
            path = analysis_path(database, trial["id"])
            atomic_json(path, dict(analysis_version=ANALYSIS_VERSION, image_sha256="sha",
                                  source_digest=source_digest(trial, rows)))
            self.assertIsNotNone(load_analysis(database, trial, rows))
            self.assertIsNone(load_analysis(database, trial, [sample(0, 110, 100)]))
            self.assertIsNone(load_analysis(database, dict(trial, status="skipped"), rows))
            with self.assertRaises(ValueError):
                analysis_path(database, "../escape")

    def test_export_counts_unvisited_objects_and_excludes_partial_trials(self):
        from analyze_gaze import export_csv
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "dataset.sqlite3"
            recordings = []
            for number, status in enumerate(("completed", "interrupted_pause"), 1):
                trial = dict(id=str(number)*32, status=status, image_sha256="sha", geometry="{}", image_id=number, path=f"{number}.jpg")
                recordings.append((trial, []))
                boxes = [dict(object_id=i, class_id=39, class_name="bottle", confidence=.8, xyxy=[0, 0, 10, 10]) for i in (1, 2)]
                stats = [dict(object_id=i, class_id=39, class_name="bottle", fixation_count=1 if i==1 else 0, dwell_seconds=.2 if i==1 else 0.) for i in (1, 2)]
                atomic_json(analysis_path(database, trial["id"]), dict(analysis_version=ANALYSIS_VERSION,
                    source_digest=source_digest(trial, []), image_sha256="sha", objects=boxes,
                    fixations=[], transitions=[], object_stats=stats))
            export_csv(database, recordings)
            with (Path(folder)/"analysis/class_summary.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["object_count"], "2")
            self.assertEqual(rows[0]["images_with_detection"], "1")
            self.assertEqual(rows[0]["fixated_objects"], "1")


if __name__ == "__main__":
    unittest.main()
