#!/usr/bin/env python3
"""Run with: python app.py. All camera processing stays local."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import logging
import faulthandler
from datetime import datetime, timezone

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    faulthandler.enable()
    if "--diagnose" in sys.argv:
        faulthandler.dump_traceback_later(15, repeat=True)
    logging.info("Запуск Python; загружаются NumPy, Qt и Apple Vision…")

import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QFont, QImage
from PySide6.QtWidgets import QApplication, QWidget
import Quartz
from core import Ridge, Smoother, to_pixels, select_model, FEATURE_NAMES
from vision_camera import Camera

CALIBRATION = [(x, y) for y in (.05, .275, .5, .725, .95)
               for x in (.05, .275, .5, .725, .95)]
TUNING = [(.18, .18), (.82, .82), (.82, .18), (.18, .82), (.42, .58)]
VALIDATION = [(x, y) for y in (.08, .5, .92) for x in (.08, .5, .92)]
TARGETS = {"calibration": CALIBRATION, "tuning": TUNING, "validation": VALIDATION}


class Window(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Vision Gaze MVP")
        self.screen_ref = QApplication.primaryScreen()
        self.setGeometry(self.screen_ref.geometry())
        display = Quartz.CGMainDisplayID()
        self.pixels = (int(Quartz.CGDisplayPixelsWide(display)),
                       int(Quartz.CGDisplayPixelsHigh(display)))
        self.original_geometry = self.screen_ref.geometry()
        self.screen_ref.geometryChanged.connect(self.display_changed)
        self.mode = "calibration"
        self.index = 0
        self.collect_since = None
        self.samples = []
        self.train_x, self.train_y = [], []
        self.errors = []
        self.tune_x, self.tune_y, self.region_errors = [], [], []
        self.model_scores = []
        self.show_debug = False
        self.debug_image = None
        self.model = None
        self.smoother = Smoother(args.smoothing)
        self.point = None
        self.status = "Ожидание камеры…"
        self.summary = ""
        self.sequence = 0
        self.last_good = 0
        self.round_number = 0
        self.folder = Path(args.output).expanduser().resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.folder / "trajectory.csv").open("x", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["timestamp", "x", "y"])
        self.csv_file.flush()
        self.metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": "primary display physical pixels, top-left origin, x right, y down",
            "display_id": int(display), "pixel_size": self.pixels,
            "logical_size": [self.width(), self.height()],
            "timestamp": "Unix seconds UTC, host time immediately after camera read",
            "feature_schema": "eyes_only_v1", "feature_names": FEATURE_NAMES,
            "camera_index": args.camera, "smoothing_tau_seconds": args.smoothing,
            "model_selection": "linear/quadratic pupil terms; alpha 0.1,1,10,30,100; separate tuning targets", "calibration_rounds": [],
        }
        self.save_metadata()
        self.camera = Camera(args.camera)
        self.ui_only = getattr(args, "ui_only", False)
        if self.ui_only:
            self.status = "Проверка интерфейса: камера отключена, калибровка недоступна"
        self.started_camera = False
        self.last_logged_status = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(25)

    def start_camera(self):
        if not self.ui_only:
            self.started_camera = True
            logging.info("Запуск потока камеры")
            self.camera.start()

    def save_metadata(self):
        (self.folder / "session.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def display_changed(self, geometry):
        if geometry != self.original_geometry:
            self.mode = "display_changed"
            self.point = None
            self.status = "Настройки экрана изменились. Закройте приложение и запустите заново."

    def target(self):
        return TARGETS[self.mode][self.index]

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif event.key() == Qt.Key.Key_D:
            self.show_debug = not self.show_debug
        elif event.key() == Qt.Key.Key_C and self.mode != "display_changed":
            self.mode, self.index = "calibration", 0
            self.collect_since = None
            self.samples, self.train_x, self.train_y, self.errors = [], [], [], []
            self.model, self.point = None, None
            self.smoother.reset()
            self.summary = ""
            self.tune_x, self.tune_y, self.region_errors = [], [], []
            self.model_scores = []
        elif (event.key() == Qt.Key.Key_Space and not self.ui_only
              and self.mode in TARGETS):
            if self.collect_since is None:
                self.samples = []
                self.collect_since = time.monotonic()

    def tick(self):
        now = time.monotonic()
        packet = self.camera.snapshot()
        if self.show_debug and hasattr(self.camera, "debug_snapshot"):
            preview = self.camera.debug_snapshot()
            if preview is not None:
                frame, captured = preview
                if now - captured < 0.5:
                    h, w, _ = frame.shape
                    self.debug_image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888).copy()
                else:
                    self.debug_image = None
            else:
                self.debug_image = None
        if packet is not None and packet[0] != self.sequence:
            self.sequence, features, status, stamp, mono = packet
            if status != self.last_logged_status:
                logging.info("Камера/Vision: %s", status)
                self.last_logged_status = status
            if self.mode != "display_changed":
                self.status = status
                if now - mono > 0.5:
                    features = None
                    self.status = "Кадр устарел — ждём свежий"
                if features is not None:
                    self.last_good = now
                    if self.mode in TARGETS:
                        self.collect(features, mono)
                    elif self.mode == "tracking":
                        prediction = self.model.predict(features)
                        self.point = self.smoother.update(prediction, mono)
                        x, y = to_pixels(self.point, *self.pixels)
                        self.writer.writerow([f"{stamp:.6f}", f"{x:.2f}", f"{y:.2f}"])
                        self.csv_file.flush()
                else:
                    self.point = None
                    self.smoother.reset()
        if now - self.last_good > 0.5:
            self.point = None
            self.smoother.reset()
        if self.collect_since is not None and now - self.collect_since > 20:
            self.collect_since = None
            self.samples = []
            self.status = "Мало валидных кадров. Поправьте свет/положение и нажмите Space."
        self.update()

    def collect(self, features, mono):
        # Reject pre-target frames and allow the eyes to settle after Space.
        if self.collect_since is None or mono - self.collect_since < 1.0:
            return
        self.samples.append(features.copy())
        if len(self.samples) < self.args.samples:
            return
        target = np.asarray(self.target())
        if self.mode == "calibration":
            self.train_x.extend(self.samples)
            self.train_y.extend([target.copy() for _ in self.samples])
        elif self.mode == "tuning":
            self.tune_x.extend(self.samples)
            self.tune_y.extend([target.copy() for _ in self.samples])
        else:
            residuals = (self.model.predict(self.samples) - target) * (np.array(self.pixels) - 1)
            point_errors = np.linalg.norm(residuals, axis=1)
            self.errors.extend(point_errors.tolist())
            self.region_errors.append({"target": target.tolist(),
                                       "median_px": float(np.median(point_errors)),
                                       "p90_px": float(np.percentile(point_errors, 90))})
        self.collect_since = None
        self.samples = []
        self.index += 1
        if self.mode == "calibration" and self.index == len(CALIBRATION):
            self.mode, self.index = "tuning", 0
        elif self.mode == "tuning" and self.index == len(TUNING):
            self.model, self.model_scores = select_model(
                self.train_x, self.train_y, self.tune_x, self.tune_y, self.pixels)
            self.round_number += 1
            logging.info("Selected degree=%s alpha=%s", self.model.degree, self.model.alpha)
            np.savez(self.folder / f"calibration-{self.round_number}.npz",
                     feature_names=np.array(FEATURE_NAMES),
                     x=self.train_x, y=self.train_y, tune_x=self.tune_x, tune_y=self.tune_y,
                     mean=self.model.mean, scale=self.model.scale, weights=self.model.weights,
                     center=self.model.center, degree=self.model.degree, alpha=self.model.alpha)
            self.mode, self.index = "validation", 0
        elif self.mode == "validation" and self.index == len(VALIDATION):
            median, p90 = np.percentile(self.errors, [50, 90])
            self.summary = f"Модель {self.model.degree} · α={self.model.alpha:g} · Проверка: медиана {median:.0f} px · P90 {p90:.0f} px"
            self.metadata["calibration_rounds"].append({
                "round": self.round_number, "completed_timestamp": time.time(),
                "degree": self.model.degree, "alpha": self.model.alpha,
                "candidate_scores": self.model_scores, "regions": self.region_errors,
                "validation_median_px": float(median), "validation_p90_px": float(p90)})
            self.save_metadata()
            self.mode = "tracking"
            self.smoother.reset()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#111827"))
        p.setPen(QColor("#e5e7eb"))
        p.setFont(QFont("Helvetica", 16))
        if self.mode in TARGETS:
            count = len(TARGETS[self.mode])
            title = {"calibration": "Калибровка", "tuning": "Подбор модели", "validation": "Независимая проверка"}[self.mode]
            progress = (f"Сбор: {len(self.samples)}/{self.args.samples} — смотрите на точку"
                        if self.collect_since is not None else "Смотрите на точку и нажмите Space")
            # Keep text between target rows, clear of the new near-edge targets.
            p.drawText(28, int(self.height()*.60), f"{title} {self.index + 1}/{count} · {progress}")
            x, y = self.target()
            center = QPointF(x * self.width(), y * self.height())
            p.setBrush(QColor("#fbbf24"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(center, 13, 13)
            p.setBrush(QColor("#111827"))
            p.drawEllipse(center, 3, 3)
        elif self.mode == "tracking":
            p.drawText(28, 35, self.summary)
            p.drawText(28, 65, "Точка взгляда · CSV записывается")
            p.setFont(QFont("Helvetica", 11))
            for row in range(3):
                values = self.region_errors[row*3:row*3+3]
                p.drawText(28, 92 + row*20, "   |   ".join(f"{r['median_px']:.0f} px" for r in values))
            if self.point is not None:
                clipped = np.clip(self.point, 0, 1)
                p.setBrush(QColor("#34d399"))
                p.drawEllipse(QPointF(float(clipped[0]) * (self.width()-1),
                                     float(clipped[1]) * (self.height()-1)), 12, 12)
        if self.show_debug and self.debug_image is not None and self.collect_since is None:
            preview = self.debug_image.scaled(420, 150, Qt.AspectRatioMode.KeepAspectRatio)
            p.drawImage(self.width()-preview.width()-20, self.height()-preview.height()-85, preview)
        p.setPen(QColor("#d1d5db"))
        p.setFont(QFont("Helvetica", 12))
        status_y = int(self.height()*.64) if self.mode in TARGETS else self.height()-48
        help_y = int(self.height()*.68) if self.mode in TARGETS else self.height()-24
        p.drawText(28, status_y, self.status)
        p.drawText(28, help_y, "D — глаза   ·   C — заново   ·   Esc/Q — выход   ·   Не двигайте голову и экран")
        p.end()

    def closeEvent(self, event):
        self.timer.stop()
        if self.started_camera:
            self.camera.requestInterruption()
            self.camera.wait()  # Release camera before destroying its thread.
        self.csv_file.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Apple Vision gaze MVP for the primary display")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--windowed", action="store_true", help="Open a normal window for troubleshooting")
    parser.add_argument("--ui-only", action="store_true", help="Test drawing without opening the camera")
    parser.add_argument("--diagnose", action="store_true", help="Print thread stacks every 15 seconds")
    parser.add_argument("--samples", type=int, default=35, help="Valid frames per target (default 35)")
    parser.add_argument("--smoothing", type=float, default=0.15, help="EMA time constant in seconds")
    parser.add_argument("--output", default="sessions/" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args = parser.parse_args()
    if args.samples < 10 or not 0.01 <= args.smoothing <= 2:
        parser.error("--samples must be >=10; --smoothing must be 0.01..2 seconds")
    logging.info("Импорты завершены; создаётся приложение Qt")
    app = QApplication(sys.argv[:1])
    window = Window(args)
    if args.windowed:
        window.show()
    else:
        window.showFullScreen()
    window.raise_()
    window.activateWindow()
    logging.info("Окно показано; результаты: %s", window.folder)
    # Let Qt paint and process native window events before opening AVFoundation.
    QTimer.singleShot(500, window.start_camera)
    QTimer.singleShot(0, lambda: logging.info("Цикл интерфейса работает"))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
