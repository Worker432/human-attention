#!/usr/bin/env python3
"""Visualize and record gaze coordinates supplied by Beam Eye Tracker."""
import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from beam_source import BeamSource
from core import Smoother


CONFIDENCE_LEVELS = {"low": 1, "medium": 2, "high": 3}


class Window(QWidget):
    def __init__(self, args, screen):
        super().__init__()
        self.args = args
        self.screen_ref = screen
        self.setWindowTitle("Beam Gaze Recorder")
        geometry = screen.geometry()
        self.viewport = (geometry.x(), geometry.y(), geometry.width(), geometry.height())
        self.setGeometry(geometry)

        if args.overlay:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
            )

        self.source = BeamSource(self.viewport, CONFIDENCE_LEVELS[args.min_confidence])
        self.started_source = False
        self.smoother = Smoother(args.smoothing)
        self.point = None
        self.sequence = 0
        self.last_good = 0.0
        self.status = "Ожидание Beam…"
        self.confidence = "LOST_TRACKING"
        self.last_logged_status = None

        self.folder = Path(args.output).expanduser().resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.folder / "trajectory.csv").open("x", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["timestamp", "x", "y", "confidence"])
        self.csv_file.flush()
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Beam Eye Tracker SDK",
            "screen_index": args.screen,
            "screen_name": screen.name(),
            "viewport_global": list(self.viewport),
            "coordinate_system": "selected-display logical pixels, top-left origin, x right, y down",
            "timestamp": "Unix seconds UTC, host time immediately after Beam SDK read",
            "smoothing_tau_seconds": args.smoothing,
            "minimum_confidence": args.min_confidence,
        }
        (self.folder / "session.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if args.ui_only:
            self.status = "Проверка интерфейса: Beam SDK не подключён"
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(25)

    def start_source(self):
        if not self.args.ui_only:
            self.started_source = True
            logging.info("Подключение к Beam Eye Tracker")
            self.source.start()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()

    def tick(self):
        now = time.monotonic()
        packet = self.source.snapshot()
        if packet is not None and packet[0] != self.sequence:
            self.sequence, point, confidence, status, stamp, mono = packet
            self.status = status
            self.confidence = confidence
            if status != self.last_logged_status:
                logging.info("%s", status)
                self.last_logged_status = status

            if now - mono <= 0.5 and point is not None:
                self.last_good = now
                self.point = self.smoother.update(point, mono)
                clipped = np.clip(self.point, 0.0, 1.0)
                width, height = self.viewport[2:]
                x = float(clipped[0]) * (width - 1)
                y = float(clipped[1]) * (height - 1)
                self.writer.writerow([f"{stamp:.6f}", f"{x:.2f}", f"{y:.2f}", confidence])
                self.csv_file.flush()
            else:
                self.point = None
                self.smoother.reset()

        if now - self.last_good > 0.5:
            self.point = None
            self.smoother.reset()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.args.overlay:
            painter.fillRect(self.rect(), QColor("#111827"))

        if self.point is not None:
            clipped = np.clip(self.point, 0.0, 1.0)
            center = QPointF(
                float(clipped[0]) * (self.width() - 1),
                float(clipped[1]) * (self.height() - 1),
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(52, 211, 153, 190))
            painter.drawEllipse(center, self.args.dot_radius, self.args.dot_radius)
            painter.setBrush(QColor(255, 255, 255, 210))
            painter.drawEllipse(center, 3, 3)

        painter.setFont(QFont("Helvetica", 12))
        painter.setPen(QColor(239, 68, 68, 230))
        painter.setBrush(QColor(239, 68, 68, 230))
        painter.drawEllipse(QPointF(18, 18), 6, 6)
        painter.drawText(30, 23, "REC gaze")
        painter.setPen(QColor(229, 231, 235, 230))
        painter.drawText(12, 48, self.status)
        if not self.args.overlay:
            painter.drawText(12, self.height() - 18, "Esc/Q — выход · CSV записывается")
        painter.end()

    def closeEvent(self, event):
        self.timer.stop()
        if self.started_source:
            self.source.requestInterruption()
            self.source.wait(1000)
        self.csv_file.close()
        event.accept()


def build_parser():
    parser = argparse.ArgumentParser(description="Beam gaze visualization and CSV recorder")
    parser.add_argument("--screen", type=int, default=0, help="Qt screen index; use --list-screens")
    parser.add_argument("--list-screens", action="store_true", help="Print available displays and exit")
    parser.add_argument("--overlay", action="store_true", help="Transparent click-through overlay")
    parser.add_argument("--windowed", action="store_true", help="Normal window for initial testing")
    parser.add_argument("--ui-only", action="store_true", help="Test UI without loading Beam SDK")
    parser.add_argument("--min-confidence", choices=CONFIDENCE_LEVELS, default="medium")
    parser.add_argument("--smoothing", type=float, default=0.12, help="EMA time constant in seconds")
    parser.add_argument("--dot-radius", type=int, default=18)
    parser.add_argument("--output", default="sessions/beam-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if not 0.01 <= args.smoothing <= 2.0 or not 3 <= args.dot_radius <= 100:
        parser.error("--smoothing must be 0.01..2; --dot-radius must be 3..100")
    if args.overlay and args.windowed:
        parser.error("choose either --overlay or --windowed")

    app = QApplication(sys.argv[:1])
    screens = app.screens()
    if args.list_screens:
        for index, screen in enumerate(screens):
            geometry = screen.geometry()
            print(
                f"{index}: {screen.name()} "
                f"{geometry.width()}x{geometry.height()} at ({geometry.x()},{geometry.y()})"
            )
        return 0
    if not 0 <= args.screen < len(screens):
        parser.error(f"--screen must be between 0 and {len(screens) - 1}")

    screen = screens[args.screen]
    window = Window(args, screen)
    if args.windowed:
        window.resize(min(1000, screen.geometry().width()), min(700, screen.geometry().height()))
        window.show()
    else:
        window.show()
        window.setGeometry(screen.geometry())
    window.raise_()
    logging.info("Экран %s: %s; результаты: %s", args.screen, screen.name(), window.folder)
    QTimer.singleShot(400, window.start_source)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
