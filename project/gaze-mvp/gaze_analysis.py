"""Pure fixation extraction, object association, and versioned sidecar storage."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime, timezone
import uuid

ANALYSIS_VERSION = 1
DEFAULT_FIXATIONS = dict(min_duration=.1, max_gap=.1, dispersion_fraction=.04, min_samples=3)


def usable(row):
    return bool(row["valid"] and row["inside_image"] and all(
        row[k] is not None and math.isfinite(row[k]) for k in ("elapsed", "image_x", "image_y")))


def source_digest(trial, samples):
    content = dict(status=trial["status"], image_sha256=trial["image_sha256"],
                   geometry=trial["geometry"], samples=samples)
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_recordings(database, trial_id=None):
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN")
        protocol = json.loads(db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0])
        query = '''SELECT t.*,i.path FROM trials t JOIN images i ON i.id=t.image_id
                   WHERE t.status NOT IN ('presenting','unreadable')'''
        args = ()
        if trial_id:
            query += " AND t.id=?"
            args = (trial_id,)
        trials = [dict(r) for r in db.execute(query + " ORDER BY t.onset_utc", args)]
        result = []
        for trial in trials:
            samples = [dict(r) for r in db.execute('''SELECT * FROM samples WHERE trial_id=?
                         ORDER BY elapsed,sequence''', (trial["id"],))]
            result.append((trial, samples))
        return protocol, result
    finally:
        db.close()


def extract_fixations(samples, width, height, *, min_duration=.1, max_gap=.1,
                      dispersion_fraction=.04, min_samples=3):
    """I-DT in original pixels. Invalid points and >max_gap intervals split segments.

    Dispersion is (x span + y span), threshold is a fraction of the image diagonal.
    These are provisional analysis parameters, not a validated eye-tracker calibration.
    """
    if width <= 0 or height <= 0 or min_duration <= 0 or max_gap <= 0 or dispersion_fraction <= 0 or min_samples < 2:
        raise ValueError("Invalid fixation parameters")
    segments, segment = [], []
    for row in samples:
        if not usable(row):
            if segment:
                segments.append(segment)
                segment = []
            continue
        if segment and not 0 < row["elapsed"] - segment[-1]["elapsed"] <= max_gap:
            segments.append(segment)
            segment = []
        segment.append(row)
    if segment:
        segments.append(segment)
    threshold = dispersion_fraction * math.hypot(width, height)

    def dispersion(rows):
        return (max(r["image_x"] for r in rows) - min(r["image_x"] for r in rows)
                + max(r["image_y"] for r in rows) - min(r["image_y"] for r in rows))

    fixations = []
    for segment_id, rows in enumerate(segments):
        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and (j - i + 1 < min_samples or rows[j]["elapsed"] - rows[i]["elapsed"] < min_duration - 1e-9):
                j += 1
            if j >= len(rows):
                break
            if dispersion(rows[i:j+1]) > threshold:
                i += 1
                continue
            while j + 1 < len(rows) and dispersion(rows[i:j+2]) <= threshold:
                j += 1
            window = rows[i:j+1]
            fixations.append(dict(id=len(fixations)+1, segment_id=segment_id,
                start=window[0]["elapsed"], end=window[-1]["elapsed"],
                duration=window[-1]["elapsed"]-window[0]["elapsed"],
                x=sum(r["image_x"] for r in window)/len(window),
                y=sum(r["image_y"] for r in window)/len(window),
                sample_count=len(window), dispersion_px=dispersion(window)))
            i = j + 1
    return fixations


def inside(box, x, y):
    x1, y1, x2, y2 = box["xyxy"]
    return x1 <= x < x2 and y1 <= y < y2


def associate(fixations, boxes):
    mapped = []
    for fixation in fixations:
        candidates = [box for box in boxes if inside(box, fixation["x"], fixation["y"])]
        row = dict(fixation, candidate_ids=[b["object_id"] for b in candidates],
                   object_id=None, class_id=None, class_name=None,
                   assignment="outside" if not candidates else "ambiguous")
        if len(candidates) == 1:
            box = candidates[0]
            row.update(object_id=box["object_id"], class_id=box["class_id"],
                       class_name=box["class_name"], assignment="assigned")
        mapped.append(row)
    return mapped


def summarize(fixations, boxes):
    transitions = []
    for previous, current in zip(fixations, fixations[1:]):
        if (previous["assignment"] == current["assignment"] == "assigned"
                and previous["segment_id"] == current["segment_id"]
                and previous["object_id"] != current["object_id"]):
            transitions.append(dict(from_fixation=previous["id"], to_fixation=current["id"],
                from_object=previous["object_id"], to_object=current["object_id"],
                from_class=previous["class_name"], to_class=current["class_name"],
                elapsed=current["start"], gap_seconds=current["start"]-previous["end"]))
    stats = []
    for box in boxes:
        selected = [f for f in fixations if f["object_id"] == box["object_id"]]
        stats.append(dict(object_id=box["object_id"], class_id=box["class_id"], class_name=box["class_name"],
                          fixation_count=len(selected), dwell_seconds=sum(f["duration"] for f in selected)))
    return transitions, stats


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8") as file:
            temp = Path(file.name)
            json.dump(content, file, ensure_ascii=False, allow_nan=False)
            file.flush()
        temp.replace(path)
    finally:
        if temp and temp.exists():
            temp.unlink()


def analysis_path(database, trial_id):
    if len(trial_id) != 32 or any(c not in "0123456789abcdef" for c in trial_id):
        raise ValueError("Invalid trial ID")
    return Path(database).parent / "analysis/trials" / f"{trial_id}.json"


def load_analysis(database, trial, samples):
    try:
        result = json.loads(analysis_path(database, trial["id"]).read_text())
        if (result["analysis_version"] != ANALYSIS_VERSION or result["source_digest"] != source_digest(trial, samples)
                or result["image_sha256"] != trial["image_sha256"]):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def annotation_path(database, image_sha):
    if len(image_sha) != 64 or any(c not in "0123456789abcdef" for c in image_sha):
        raise ValueError("Invalid image SHA-256")
    return Path(database).parent / "annotations/images" / f"{image_sha}.json"


def load_annotation(database, trial):
    try:
        record = json.loads(annotation_path(database, trial["image_sha256"]).read_text())
        geometry = json.loads(trial["geometry"])
        if (record["schema_version"] != 1 or record["image_sha256"] != trial["image_sha256"]
                or record["width"] != geometry["image_width"] or record["height"] != geometry["image_height"]):
            return None
        validate_boxes(record["objects"], record["width"], record["height"])
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def validate_boxes(boxes, width, height):
    ids = set()
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Рамка выходит за границы изображения или имеет нулевой размер")
        if box["object_id"] in ids or not isinstance(box["object_id"], int) or box["object_id"] < 1:
            raise ValueError("ID объектов должны быть уникальными положительными числами")
        ids.add(box["object_id"])
        if not isinstance(box["class_name"], str) or not box["class_name"].strip():
            raise ValueError("У каждой рамки должен быть класс")


def class_catalog(database):
    root = Path(database).parent
    result = {"person": 0, "car": 2, "bottle": 39}
    for path in sorted((root / "analysis/trials").glob("*.json")):
        try:
            value = json.loads(path.read_text())
            for box in value.get("objects", []):
                result.setdefault(box["class_name"], box["class_id"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    try:
        result.update(json.loads((root / "annotations/classes.json").read_text()))
    except FileNotFoundError:
        pass
    return result


def save_annotation(database, trial, boxes, *, reviewed, note=""):
    import copy
    if not reviewed:
        raise ValueError("Подтвердите, что проверили все выбранные типы объектов на изображении")
    geometry = json.loads(trial["geometry"])
    width, height = geometry["image_width"], geometry["image_height"]
    boxes = copy.deepcopy(boxes)
    validate_boxes(boxes, width, height)
    catalog = class_catalog(database)
    next_class = max(999, *catalog.values()) + 1
    for box in boxes:
        label = box["class_name"].strip().casefold()
        if len(label) > 80:
            raise ValueError("Название класса слишком длинное")
        if label not in catalog:
            catalog[label] = next_class
            next_class += 1
        box.update(class_name=label, class_id=catalog[label], confidence=None, source="manual")
    record = dict(schema_version=1, image_sha256=trial["image_sha256"], width=width, height=height,
                  revision=uuid.uuid4().hex, created_utc=datetime.now(timezone.utc).isoformat(),
                  reviewed=True, note=note, objects=boxes)
    path = annotation_path(database, trial["image_sha256"])
    if path.exists():
        old = json.loads(path.read_text())
        revision = old.get("revision", uuid.uuid4().hex)
        atomic_json(path.parent.parent / "history" / f"{trial['image_sha256']}-{revision}.json", old)
    atomic_json(Path(database).parent / "annotations/classes.json", catalog)
    atomic_json(path, record)
    return record


def manual_analysis(database, trial, samples):
    record = load_annotation(database, trial)
    if record is None or not record.get("reviewed"):
        return None
    automatic = load_analysis(database, trial, samples)
    parameters = automatic.get("fixation_parameters", DEFAULT_FIXATIONS) if automatic else DEFAULT_FIXATIONS
    fixes = associate(extract_fixations(samples, record["width"], record["height"], **parameters), record["objects"])
    transitions, stats = summarize(fixes, record["objects"])
    return dict(analysis_version=ANALYSIS_VERSION, annotation_source="manual", annotation_revision=record["revision"],
                trial_id=trial["id"], image_id=trial["image_id"], image_path=trial["path"],
                image_sha256=trial["image_sha256"], source_digest=source_digest(trial, samples),
                trial_status=trial["status"], fixation_parameters=parameters,
                objects=record["objects"], fixations=fixes, transitions=transitions, object_stats=stats,
                comparison=dict(automatic_assigned=sum(f["assignment"] == "assigned" for f in automatic["fixations"]) if automatic else None,
                                automatic_transitions=len(automatic["transitions"]) if automatic else None))


def effective_analysis(database, trial, samples, prefer_manual=True):
    if prefer_manual:
        manual = manual_analysis(database, trial, samples)
        if manual is not None:
            return manual
    result = load_analysis(database, trial, samples)
    if result:
        result = dict(result, annotation_source="yolo")
    return result
