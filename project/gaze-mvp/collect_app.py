#!/usr/bin/env python3
"""Fixed-duration image viewing with durable Beam gaze collection."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from beam_source import BeamSource
from collection_store import CollectionStore, image_coordinates

BASE = Path(__file__).resolve().parent


class Collector(QWidget):
    def __init__(self, store, screen, args):
        super().__init__()
        self.store, self.args = store, args
        self.setWindowTitle("Сбор датасета взгляда")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(screen.geometry())
        self.screen_ref = screen
        self.original_geometry = screen.geometry()
        self.viewport = (self.original_geometry.x(), self.original_geometry.y(),
                         self.original_geometry.width(), self.original_geometry.height())
        self.source = BeamSource(self.viewport, min_confidence=0, capture_queue=True)
        self.started_source = False
        self.source_status = "Ожидание подключения Beam…"
        self.last_good = -math.inf
        self.preview = None
        self.mode = "ready"
        self.message = ""
        self.trial = None
        self.image = None
        self.onset = None
        self.block_count = 0
        self.closed = False
        self.review_button = QPushButton("Просмотреть записи (R)", self)
        self.review_button.setStyleSheet("padding:10px; background:#364456; color:white; border-radius:6px;")
        self.review_button.clicked.connect(self.open_review)
        self.store.begin_run(dict(screen_name=screen.name(), viewport=self.viewport,
                                  device_pixel_ratio=screen.devicePixelRatio(),
                                  minimum_confidence="MEDIUM", smoothing="none",
                                  source="Beam Eye Tracker SDK", ui_only=args.ui_only,
                                  break_every=args.break_every,
                                  onset_clock="host monotonic at first QPainter draw; not photodiode verified",
                                  timestamp="host read time + Beam SDK timestamp; not camera exposure time"))
        screen.geometryChanged.connect(self.display_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def start_source(self):
        if not self.args.ui_only:
            self.started_source = True
            self.source.start()

    def display_changed(self, geometry):
        if geometry != self.original_geometry:
            self.pause("Настройки экрана изменились. Закройте и снова запустите приложение.", "display_changed")
            self.mode = "display_changed"

    def healthy(self, now):
        return now - self.last_good <= 0.5

    def start(self):
        if self.mode in {"done", "display_changed"}:
            return
        if not self.healthy(time.monotonic()):
            self.message = "Включите Beam и проверьте калибровку на этом экране. Нужен устойчивый сигнал."
            self.update()
            return
        self.block_count = 0
        self.message = ""
        self.next_image()

    def next_image(self):
        entry = self.store.next_image()
        if entry is None:
            self.mode = "done"
            self.update()
            return
        self.entry = entry
        path = self.store.root / entry["path"]
        try:
            data = path.read_bytes()
        except OSError as exc:
            data = b""
            logging.warning("Не удалось прочитать %s: %s", path, exc)
        self.image = QImage.fromData(data)
        self.sha = hashlib.sha256(data).hexdigest()
        if self.image.isNull():
            trial = self.store.begin_trial(entry["id"], self.sha, {}, time.time(), time.monotonic())
            self.store.finish_trial(trial, "unreadable", time.time(), 0.)
            self.mode = "paused"
            self.message = f"Не удалось открыть {entry['path']}. Файл отмечен. Space — следующий."
            self.update()
            return
        self.mode = "interval"
        self.interval_start = None
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.update()

    def finish(self, reason):
        if self.trial:
            # Record endpoint only after draining queued updates in tick().
            self.store.finish_trial(self.trial, reason, time.time(), time.monotonic() - self.onset)
            self.trial = None

    def pause(self, message, reason="interrupted_pause"):
        self.finish(reason)
        self.mode = "paused"
        self.message = message
        self.unsetCursor()
        self.update()

    def record_packets(self, now):
        rows = []
        for sequence, point, confidence, status, stamp, mono, sdk_stamp in self.source.drain():
            valid = (point is not None and confidence in {"MEDIUM", "HIGH"}
                     and 0 <= now - mono <= 0.5)
            self.source_status = status
            if valid:
                self.last_good = max(self.last_good, mono)
                self.preview = point
            else:
                self.preview = None
            if self.trial is None or not self.onset <= mono < self.onset + self.args.duration:
                continue
            if point is None:
                nx = ny = sx = sy = ix = iy = None
                inside = False
            else:
                nx, ny = point
                sx, sy, ix, iy, inside = image_coordinates(
                    point, self.viewport[2:], self.rect_values, (self.image.width(), self.image.height()))
            rows.append((self.trial, sequence, stamp, mono, sdk_stamp, mono - self.onset,
                         confidence, int(valid), int(inside), nx, ny, sx, sy, ix, iy))
        self.store.add_samples(rows)

    def tick(self):
        now = time.monotonic()
        self.record_packets(now)
        if self.mode == "viewing":
            if not self.healthy(now):
                self.pause("Сигнал взгляда потерян. Изображение отмечено как прерванное.\nПроверьте Beam. Space — продолжить.", "interrupted_tracking")
            elif now - self.onset >= self.args.duration:
                self.finish("completed")
                self.block_count += 1
                if self.block_count >= self.args.break_every:
                    self.pause("Пора отдохнуть. Прогресс сохранён.\nSpace — продолжить после перерыва.")
                else:
                    self.next_image()
        elif self.mode == "interval":
            if not self.healthy(now):
                self.pause("Нет устойчивого сигнала Beam. Space — продолжить после проверки.")
            elif self.interval_start is not None and now - self.interval_start >= 1.:
                self.mode = "pending"
        self.update()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() in {Qt.Key.Key_Escape, Qt.Key.Key_Q}:
            self.close()
        elif event.key() == Qt.Key.Key_R:
            self.open_review()
        elif event.key() == Qt.Key.Key_Space:
            if self.mode in {"viewing", "interval", "pending"}:
                self.record_packets(time.monotonic())
                self.pause("Пауза. Прогресс сохранён. Space — продолжить.")
            else:
                self.start()
        elif event.key() == Qt.Key.Key_Right and self.mode == "viewing":
            self.record_packets(time.monotonic())
            self.finish("skipped")
            self.next_image()

    def open_review(self):
        from review_app import ReviewPanel
        if self.mode in {"viewing", "interval", "pending"}:
            self.record_packets(time.monotonic())
            self.pause("Просмотр записей. Space — продолжить сбор.", "interrupted_pause")
        panel = ReviewPanel(self.store.folder / "dataset.sqlite3", self)
        panel.exec()
        panel.deleteLater()

    def event(self, event):
        if (event.type() == QEvent.Type.WindowDeactivate and hasattr(self, "mode")
                and self.mode in {"viewing", "interval", "pending"}):
            self.record_packets(time.monotonic())
            self.pause("Окно было скрыто. Space — продолжить.", "interrupted_focus")
        return super().event(event)

    def paintEvent(self, event):
        self.review_button.setGeometry(max(10, self.width() - 240), 14, 225, 38)
        self.review_button.setVisible(self.mode not in {"viewing", "interval", "pending"})
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191d"))
        if self.mode in {"pending", "viewing"}:
            if self.mode == "pending":
                scale = min(self.width() / self.image.width(), self.height() / self.image.height())
                width, height = self.image.width() * scale, self.image.height() * scale
                self.rect_values = ((self.width() - width) / 2, (self.height() - height) / 2, width, height)
                self.onset, stamp = time.monotonic(), time.time()
                geometry = dict(image_width=self.image.width(), image_height=self.image.height(),
                                image_rect=self.rect_values, viewport=self.viewport,
                                coordinate_system="screen local logical pixels; image original pixels; top-left origin")
                self.trial = self.store.begin_trial(self.entry["id"], self.sha, geometry, stamp, self.onset)
                # Timestamp after durable reservation, immediately before drawing.
                self.onset, stamp = time.monotonic(), time.time()
                with self.store.db:
                    self.store.db.execute("UPDATE trials SET onset_utc=?,onset_mono=? WHERE id=?",
                                          (stamp, self.onset, self.trial))
                self.mode = "viewing"
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(QRectF(*self.rect_values), self.image)
        elif self.mode == "interval":
            self.draw_cross(painter)
            if self.interval_start is None:
                self.interval_start = time.monotonic()
        else:
            self.unsetCursor()
            total, seen, completed = self.store.progress()
            painter.setPen(QColor("#f5f5f5"))
            painter.setFont(QFont("Helvetica", 24))
            title = "Все изображения просмотрены" if self.mode == "done" else "Свободный просмотр изображений"
            painter.drawText(QRectF(40, 55, self.width() - 80, 65), Qt.AlignmentFlag.AlignCenter, title)
            painter.setFont(QFont("Helvetica", 16))
            text = (f"Просмотрено: {seen:,} из {total:,} · Полных записей: {completed:,}\n"
                    f"Показ: {self.args.duration:g} с · Перерыв каждые {self.args.break_every} изображений\n\n"
                    "Смотрите на изображения естественно, как вам интересно.\n"
                    "Space — начать / пауза · → — пропустить · Esc — сохранить и выйти")
            status = f"{self.message}\n\n{self.source_status}"
            if self.args.ui_only:
                status += "\nПроверка интерфейса: запись через Beam отключена."
            flags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
            painter.drawText(QRectF(45, 135, self.width() - 90, self.height() / 2 - 160), flags, text)
            painter.setFont(QFont("Helvetica", 14))
            painter.drawText(QRectF(45, self.height() - 170, self.width() - 90, 155), flags, status)
            if self.mode in {"ready", "paused"}:
                self.draw_cross(painter)
                if self.preview is not None and self.healthy(time.monotonic()):
                    painter.setBrush(QColor("#34d399"))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(QPointF(self.preview[0] * (self.width() - 1),
                                               self.preview[1] * (self.height() - 1)), 9, 9)
        painter.end()

    def draw_cross(self, painter):
        painter.setPen(QColor("#dddddd"))
        x, y = self.width() // 2, self.height() // 2
        painter.drawLine(x - 10, y, x + 10, y)
        painter.drawLine(x, y - 10, x, y + 10)

    def closeEvent(self, event):
        if self.closed:
            event.accept()
            return
        self.timer.stop()
        if self.started_source:
            self.source.requestInterruption()
            if not self.source.wait(3000):
                self.timer.start(16)
                self.message = "Ожидание завершения Beam. Попробуйте закрыть окно ещё раз."
                event.ignore()
                return
        self.record_packets(time.monotonic())
        self.finish("interrupted_exit")
        with self.store.db:
            row = self.store.db.execute("SELECT settings FROM runs WHERE id=?", (self.store.run_id,)).fetchone()
            settings = json.loads(row[0])
            settings["queue_dropped_packets"] = self.source.dropped_packets
            self.store.db.execute("UPDATE runs SET settings=? WHERE id=?",
                                  (json.dumps(settings), self.store.run_id))
        self.store.close()
        self.closed = True
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Collect gaze on all local advertising images using Beam")
    parser.add_argument("--images", type=Path, default=BASE.parent.parent / "dataset/PittImageVideoAdsDataset/pitt_ads_images")
    parser.add_argument("--output", type=Path, default=BASE / "collected/p001")
    parser.add_argument("--participant", default="p001")
    parser.add_argument("--duration", type=float, default=5.)
    parser.add_argument("--screen", type=int, default=None)
    parser.add_argument("--break-every", type=int, default=50)
    parser.add_argument("--ui-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 60 or args.break_every < 1:
        parser.error("Показ: от 1 до 60 секунд; интервал перерывов: от 1 изображения.")
    if args.ui_only and args.output == BASE / "collected/p001":
        args.output = BASE / "collected/ui-preview"
    qt = QApplication(sys.argv[:1])
    screens = qt.screens()
    if args.screen is not None and not 0 <= args.screen < len(screens):
        parser.error("Неверный номер экрана. Список: python beam_app.py --list-screens")
    screen = screens[args.screen] if args.screen is not None else qt.primaryScreen()
    try:
        store = CollectionStore(args.output, args.images, args.duration, args.participant)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    window = Collector(store, screen, args)
    window.showFullScreen()
    QTimer.singleShot(400, window.start_source)
    return qt.exec()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
