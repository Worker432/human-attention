#!/usr/bin/env python3
"""Run in .venv-detector. YOLO detections + provisional I-DT object scanpaths."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from gaze_analysis import (ANALYSIS_VERSION, DEFAULT_FIXATIONS, analysis_path, associate,
                           atomic_json, extract_fixations, load_analysis, read_recordings,
                           source_digest, summarize)

BASE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".cache/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))


def export_csv(database, recordings):
    folder = Path(database).parent / "analysis"
    tables = {"objects": [], "fixations": [], "transitions": [], "object_stats": []}
    defaults = {
        "objects": ["object_id", "class_id", "class_name", "confidence", "xyxy"],
        "fixations": ["id", "segment_id", "start", "end", "duration", "x", "y", "sample_count", "dispersion_px", "candidate_ids", "object_id", "class_id", "class_name", "assignment"],
        "transitions": ["from_fixation", "to_fixation", "from_object", "to_object", "from_class", "to_class", "elapsed", "gap_seconds"],
        "object_stats": ["object_id", "class_id", "class_name", "fixation_count", "dwell_seconds"],
    }
    for trial, samples in recordings:
        result = load_analysis(database, trial, samples)
        if result is None:
            continue
        for key in tables:
            for row in result[key]:
                tables[key].append(dict(trial_id=trial["id"], image_id=trial["image_id"],
                    image_path=trial["path"], trial_status=trial["status"], **row))
    import tempfile
    for key, rows in tables.items():
        with tempfile.NamedTemporaryFile("w", dir=folder, suffix=".csv.tmp", delete=False,
                                         newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["trial_id", "image_id", "image_path", "trial_status", *defaults[key]])
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                              for k, v in row.items()} for row in rows)
            temp = Path(file.name)
        temp.replace(folder / f"{key}.csv")
    # Descriptive counts include objects that were present but never fixated.
    # Only complete trials enter this aggregate; this is not a preference score.
    classes = {}
    for row in tables["objects"]:
        if row["trial_status"] != "completed":
            continue
        group = classes.setdefault(row["class_id"], dict(class_id=row["class_id"], class_name=row["class_name"],
                                  images=set(), object_count=0, fixated_objects=0, fixation_count=0, dwell_seconds=0.))
        group["images"].add(row["image_id"])
        group["object_count"] += 1
    for row in tables["object_stats"]:
        if row["trial_status"] != "completed":
            continue
        group = classes[row["class_id"]]
        group["fixated_objects"] += int(row["fixation_count"] > 0)
        group["fixation_count"] += row["fixation_count"]
        group["dwell_seconds"] += row["dwell_seconds"]
    with tempfile.NamedTemporaryFile("w", dir=folder, suffix=".csv.tmp", delete=False,
                                     newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_id", "class_name", "images_with_detection", "object_count",
                                                "fixated_objects", "fixation_count", "dwell_seconds"])
        writer.writeheader()
        for group in sorted(classes.values(), key=lambda row: row["class_id"]):
            row = dict(group)
            row["images_with_detection"] = len(row.pop("images"))
            writer.writerow(row)
        temp = Path(file.name)
    temp.replace(folder / "class_summary.csv")


def run(args):
    from ultralytics import YOLO, settings
    from PIL import Image
    import torch
    import math
    settings.update({"sync": False})
    protocol, recordings = read_recordings(args.database, args.trial)
    if not recordings:
        print("Нет завершённых или прерванных записей для обработки.", flush=True)
        return 0
    model_path = BASE / "models/yolo11m.pt"
    model_path.parent.mkdir(exist_ok=True)
    print("Загрузка YOLO11m…", flush=True)
    model = YOLO(str(model_path))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    detector = dict(model="yolo11m", weights_sha256=model_sha,
                    ultralytics_version=importlib.metadata.version("ultralytics"),
                    torch_version=torch.__version__, confidence=.25, iou=.7, imgsz=640,
                    classes="COCO-80", augment=False)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Устройство: {device}. Записей: {len(recordings)}", flush=True)
    failures = 0
    for index, (trial, samples) in enumerate(recordings, 1):
        try:
            existing = load_analysis(args.database, trial, samples)
            if existing and existing["detector"] == detector and existing["fixation_parameters"] == DEFAULT_FIXATIONS:
                print(f"{index}/{len(recordings)} · {trial['path']} · уже обработано", flush=True)
                continue
            path = Path(protocol["image_root"]) / trial["path"]
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != trial["image_sha256"]:
                raise ValueError("Исходное изображение изменилось; сопоставление запрещено")
            geometry = json.loads(trial["geometry"])
            # QImage used by the collector does not auto-rotate EXIF; strip EXIF
            # before passing PIL pixels to YOLO to preserve the same coordinate frame.
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                image.info.clear()
            width, height = image.size
            if (width, height) != (geometry["image_width"], geometry["image_height"]):
                raise ValueError("Размер изображения отличается от сохранённой геометрии")
            key = hashlib.sha256(json.dumps(dict(image_sha=trial["image_sha256"], detector=detector), sort_keys=True).encode()).hexdigest()
            cache_path = Path(args.database).parent / "analysis/detections" / f"{key}.json"
            cached = None
            try:
                cached = json.loads(cache_path.read_text())
            except (OSError, ValueError):
                pass
            if cached is not None:
                boxes, used_device = cached["objects"], cached["device"]
            else:
                try:
                    prediction = model.predict(image, device=device, conf=.25, iou=.7, imgsz=640, verbose=False)[0]
                except RuntimeError:
                    if device != "mps":
                        raise
                    print("MPS не выполнил операцию; повтор на CPU.", flush=True)
                    device = "cpu"
                    prediction = model.predict(image, device=device, conf=.25, iou=.7, imgsz=640, verbose=False)[0]
                used_device = device
                boxes = []
                for xyxy, confidence, class_id in zip(prediction.boxes.xyxy.cpu().tolist(),
                        prediction.boxes.conf.cpu().tolist(), prediction.boxes.cls.cpu().tolist()):
                    if not all(math.isfinite(v) for v in [*xyxy, confidence, class_id]):
                        continue
                    x1, y1, x2, y2 = xyxy
                    coords = [max(0., min(width, x1)), max(0., min(height, y1)),
                              max(0., min(width, x2)), max(0., min(height, y2))]
                    if coords[2] <= coords[0] or coords[3] <= coords[1]:
                        continue
                    boxes.append(dict(class_id=int(class_id), class_name=prediction.names[int(class_id)],
                                      confidence=confidence, xyxy=coords))
                boxes.sort(key=lambda b: (b["class_id"], *b["xyxy"], -b["confidence"]))
                for number, box in enumerate(boxes, 1):
                    box["object_id"] = number
                atomic_json(cache_path, dict(objects=boxes, device=used_device, detector=detector,
                                             image_sha256=trial["image_sha256"]))
            fixations = associate(extract_fixations(samples, width, height, **DEFAULT_FIXATIONS), boxes)
            transitions, stats = summarize(fixations, boxes)
            output = dict(analysis_version=ANALYSIS_VERSION, trial_id=trial["id"], image_id=trial["image_id"],
                image_path=trial["path"], image_sha256=trial["image_sha256"], source_digest=source_digest(trial, samples),
                created_utc=time.time(), trial_status=trial["status"], detector=detector, device=used_device,
                image_width=width, image_height=height, fixation_parameters=DEFAULT_FIXATIONS,
                assignment_method="unique bbox containing centroid; multiple=ambiguous; none=outside",
                objects=boxes, fixations=fixations, transitions=transitions, object_stats=stats)
            atomic_json(analysis_path(args.database, trial["id"]), output)
            print(f"{index}/{len(recordings)} · {trial['path']} · объектов {len(boxes)}, фиксаций {len(fixations)}, переходов {len(transitions)}", flush=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures += 1
            print(f"Ошибка {trial['path']}: {exc}", flush=True)
    # Export every available trial, even when --trial only updated one record.
    _, all_recordings = read_recordings(args.database)
    (Path(args.database).parent / "analysis").mkdir(exist_ok=True)
    export_csv(args.database, all_recordings)
    print(f"Готово. Ошибок: {failures}. Результаты: {Path(args.database).parent / 'analysis'}", flush=True)
    return 2 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=BASE / "collected/p001/dataset.sqlite3")
    parser.add_argument("--trial", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    try:
        if not args.database.is_file():
            raise ValueError("База записей не найдена")
        lock_path = args.database.parent / "analysis/worker.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Обработка этой базы уже запущена. Дождитесь её завершения.")
            return run(args)
    except Exception as exc:
        print(f"Не удалось запустить обработку: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
