"""Pretrained DeepGaze III inference. Generated paths are never human recordings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from gaze_analysis import atomic_json, inside

BASE = Path(__file__).resolve().parent
DEEPGAZE_COMMIT = "c7db17e2d1d7ea6468ffdee2cfaddf141095dcff"
SCHEMA_VERSION = 1
os.environ.setdefault("TORCH_HOME", str(BASE / "models/torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".cache/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def prediction_key(image_sha, steps=8, seed=42, max_side=1024):
    config = dict(schema_version=SCHEMA_VERSION, image_sha256=image_sha, model="DeepGazeIII",
                  source_commit=DEEPGAZE_COMMIT, steps=steps, seed=seed, max_side=max_side,
                  centerbias="MIT1003", initial_fixation="image center", sampling="categorical temperature=1")
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prediction_file(folder, image_sha, steps=8, seed=42, max_side=1024):
    return Path(folder) / (prediction_key(image_sha, steps, seed, max_side) + ".json")


def map_pixel(x, y, model_size, original_size):
    return (min(original_size[0]-1, max(0., (x+.5)*original_size[0]/model_size[0]-.5)),
            min(original_size[1]-1, max(0., (y+.5)*original_size[1]/model_size[1]-.5)))


def assign_steps(points, boxes):
    result = []
    for index, (x, y) in enumerate(points, 1):
        candidates = [b for b in boxes if inside(b, x, y)]
        chosen = candidates[0] if len(candidates) == 1 else None
        result.append(dict(step=index, x=x, y=y,
            object_id=chosen["object_id"] if chosen else None,
            class_name=chosen["class_name"] if chosen else None,
            assignment="assigned" if chosen else ("ambiguous" if candidates else "outside"),
            candidate_ids=[b["object_id"] for b in candidates]))
    return result


def load_prediction(path, image_sha=None, require_heatmaps=True):
    try:
        value = json.loads(Path(path).read_text())
        if value["schema_version"] != SCHEMA_VERSION or value["data_source"] != "model_prediction":
            return None
        if image_sha is not None and value["image_sha256"] != image_sha:
            return None
        if len(value["fixations"]) != value["parameters"]["steps"]:
            return None
        if len(value["heatmaps"]) != len(value["fixations"]):
            if require_heatmaps or value['heatmaps'] or value['parameters'].get('step_heatmaps') is not False:
                return None
        root = Path(path).parent.resolve()
        for name in value["heatmaps"]:
            file = (root / name).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                return None
        for point in value["fixations"]:
            if not all(math.isfinite(point[k]) for k in ("x", "y")):
                return None
            if not (0 <= point["x"] < value["image_width"] and 0 <= point["y"] < value["image_height"]):
                return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_model(device="cpu"):
    import torch
    import torchvision
    import deepgaze_pytorch
    from unittest.mock import patch
    original = torch.hub.load

    def compatible_hub(repo, name, *args, **kwargs):
        if repo == "pytorch/vision:v0.6.0" and name == "densenet201":
            # Same DenseNet201 architecture and ImageNet weights; the old hub
            # entry point is incompatible with current torch/torchvision.
            return torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1)
        return original(repo, name, *args, **kwargs)

    with patch("torch.hub.load", side_effect=compatible_hub):
        model = deepgaze_pytorch.DeepGazeIII(pretrained=True)
    return model.eval().float().to(device)


def center_bias(width, height):
    import numpy as np
    import torch
    from PIL import Image
    file = BASE / "models/centerbias_mit1003.npy"
    if not file.exists():
        file.parent.mkdir(exist_ok=True)
        temp = file.with_suffix(".npy.part")
        torch.hub.download_url_to_file("https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy", str(temp), progress=False)
        temp.replace(file)
    template = np.load(file, allow_pickle=False)
    if template.ndim != 2 or not np.isfinite(template).all():
        raise ValueError("Некорректный файл center bias")
    values = np.asarray(Image.fromarray(template.astype("float32")).resize((width, height), Image.Resampling.NEAREST), dtype="float32").copy()
    maximum = float(values.max())
    values -= maximum + math.log(float(np.exp(values.astype("float64")-maximum).sum()))
    return values, hashlib.sha256(file.read_bytes()).hexdigest()


def detect_objects(image, device):
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    file = BASE / "models/yolo11m.pt"
    detector = YOLO(str(file))
    result = detector.predict(image, device=device, imgsz=640, conf=.25, iou=.7, verbose=False)[0]
    boxes = []
    for xyxy, label, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        boxes.append(dict(xyxy=xyxy, class_id=int(label), class_name=result.names[int(label)], confidence=confidence))
    boxes.sort(key=lambda b: (b["class_id"], *b["xyxy"], -b["confidence"]))
    for index, box in enumerate(boxes, 1):
        box["object_id"] = index
    return boxes, hashlib.sha256(file.read_bytes()).hexdigest()


def run(args, runtime=None):
    import numpy as np
    from PIL import Image
    import torch
    if runtime is None:
        runtime = {}
    if not 1 <= args.steps <= 20 or not 256 <= args.max_side <= 1536:
        raise ValueError("Число фиксаций: 1–20; длинная сторона: 256–1536 px")
    args.output.mkdir(parents=True, exist_ok=True)
    image_sha = hashlib.sha256(args.image.read_bytes()).hexdigest()
    output = prediction_file(args.output, image_sha, args.steps, args.seed, args.max_side)
    save_step_heatmaps = not getattr(args, 'aggregate_only', False)
    if not args.refresh and load_prediction(output, image_sha, require_heatmaps=save_step_heatmaps):
        print(f"RESULT {output}", flush=True)
        return output
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
        image.info.clear()  # Preserve the QImage coordinate frame, without EXIF rotation.
    original_size = image.size
    factor = args.max_side / max(image.size)
    model_size = (max(32, round(image.width*factor)), max(32, round(image.height*factor)))
    image_model = image.resize(model_size, Image.Resampling.LANCZOS)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if runtime.get('device') != device:
        print(f"DeepGaze III · {device} · Загрузка модели…", flush=True)
        runtime.clear()
        runtime.update(device=device, model=load_model(device))
    model = runtime['model']
    context_key = (image_sha, model_size)
    if runtime.get('context_key') != context_key:
        bias, bias_sha = center_bias(*model_size)
        runtime.update(context_key=context_key, bias_sha=bias_sha,
            image_tensor=torch.from_numpy(np.asarray(image_model).copy()).permute(2, 0, 1).unsqueeze(0).float().to(device),
            bias_tensor=torch.from_numpy(bias).unsqueeze(0).to(device))
        from prepared_scanpath import PreparedScanpath
        runtime['prepared'] = PreparedScanpath(model, runtime['image_tensor'])
    bias_sha = runtime['bias_sha']
    image_tensor, bias_tensor = runtime['image_tensor'], runtime['bias_tensor']
    rng = np.random.default_rng(args.seed)
    history_x, history_y = [(model_size[0]-1)/2], [(model_size[1]-1)/2]
    points, heatmaps = [], []
    heat_folder = args.output / (output.stem + "-heatmaps")
    if save_step_heatmaps:
        heat_folder.mkdir(exist_ok=True)
    with torch.inference_mode():
        for step in range(args.steps):
            hx = [history_x[i] if len(history_x) >= -i else float("nan") for i in model.included_fixations]
            hy = [history_y[i] if len(history_y) >= -i else float("nan") for i in model.included_fixations]
            xt = torch.tensor([hx], dtype=torch.float32, device=device)
            yt = torch.tensor([hy], dtype=torch.float32, device=device)
            logp = runtime['prepared'](bias_tensor, xt, yt).detach().cpu().numpy()[0, 0].astype("float64")
            if logp.shape != (model_size[1], model_size[0]) or np.isnan(logp).any():
                raise ValueError("Модель вернула некорректную карту вероятностей")
            probs = np.exp(logp-np.max(logp))
            total = probs.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError("Невозможно нормировать прогноз")
            probs /= total
            flat = int(rng.choice(probs.size, p=probs.ravel()))
            y, x = divmod(flat, model_size[0])
            history_x.append(x)
            history_y.append(y)
            points.append(map_pixel(x, y, model_size, original_size))
            # Relative density visualization only; not an object-confidence score.
            if save_step_heatmaps:
                from matplotlib import colormaps
                display = np.sqrt(probs / probs.max())
                rgba = (colormaps["inferno"](display)*255).astype("uint8")
                rgba[:, :, 3] = np.round(display*155).astype("uint8")
                heat = Image.fromarray(rgba)
                heat.thumbnail((512, 512), Image.Resampling.BILINEAR)
                heat_path = heat_folder / f"step-{step+1:02}.png"
                heat.save(heat_path)
                heatmaps.append(str(heat_path.relative_to(args.output)))
            print(f"Прогноз: {step+1}/{args.steps}", flush=True)
    print("Поиск объектов YOLO…", flush=True)
    if runtime.get('objects_key') != image_sha:
        boxes, yolo_sha = detect_objects(image, device)
        runtime.update(objects_key=image_sha, boxes=boxes, yolo_sha=yolo_sha)
    boxes, yolo_sha = runtime['boxes'], runtime['yolo_sha']
    fixes = assign_steps(points, boxes)
    import importlib.metadata
    weight_files = BASE / "models/torch/hub/checkpoints"
    if 'checkpoints' not in runtime:
        runtime['checkpoints'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in weight_files.glob("*.pth")}
    checkpoints = runtime['checkpoints']
    result = dict(schema_version=SCHEMA_VERSION, data_source="model_prediction", model="DeepGaze III",
        source_commit=DEEPGAZE_COMMIT, created_utc=datetime.now(timezone.utc).isoformat(),
        image_path=str(args.image.resolve()), image_sha256=image_sha, image_width=image.width, image_height=image.height,
        model_size=list(model_size), device=device,
        parameters=dict(steps=args.steps, seed=args.seed, max_side=args.max_side,
                        centerbias="MIT1003, nearest resize, log-normalized", centerbias_sha256=bias_sha,
                        initial_fixation="image center (conditioning only, not a predicted fixation)",
                        sampling="categorical temperature=1", predicted_durations=False,
                        step_heatmaps=save_step_heatmaps),
        versions={name: importlib.metadata.version(name) for name in ("torch", "torchvision", "deepgaze_pytorch", "ultralytics")},
        checkpoint_sha256=checkpoints, yolo_weights_sha256=yolo_sha,
        objects=boxes, fixations=fixes, heatmaps=heatmaps,
        note="Synthetic sampled scanpath, not a measured human gaze trajectory. Playback timing is illustrative; no fixation durations are predicted.")
    atomic_json(output, result)
    print(f"RESULT {output}", flush=True)
    return output


def run_variants(args):
    count = getattr(args, 'variants', 1)
    if count not in (1, 10, 30, 100):
        raise ValueError('Число вариантов: 1, 10, 30 или 100')
    start_seed = args.seed
    runtime = {}
    outputs = []
    try:
        for index in range(count):
            args.seed = (start_seed+index) % 1000000
            print(f'VARIANT_BEGIN {index+1}/{count}', flush=True)
            outputs.append(run(args, runtime))
            print(f'VARIANT {index+1}/{count}', flush=True)
    finally:
        args.seed = start_seed
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "collected/predictions")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--variants', type=int, choices=[1, 10, 30, 100], default=1)
    parser.add_argument('--aggregate-only', action='store_true', help='Skip per-step PNGs for aggregate heatmaps')
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        run_variants(args)
        return 0
    except Exception as exc:
        print(f"Ошибка прогноза: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
