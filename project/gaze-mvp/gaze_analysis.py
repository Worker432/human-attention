"""Pure fixation extraction, object association, and versioned sidecar storage."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile

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
