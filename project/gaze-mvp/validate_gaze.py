"""Nine-point measurement of Beam screen error; never calibrates or alters gaze."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QDialog
from beam_source import BeamSource
from gaze_analysis import atomic_json

BASE = Path(__file__).resolve().parent
TARGETS = [(x, y) for y in (.1, .5, .9) for x in (.1, .5, .9)]


def summarize_target(rows, target_global, viewport):
    """Errors are in Qt logical screen pixels, no clipping or smoothing."""
    points = []
    for row in rows:
        if row["valid"]:
            points.append((viewport[0] + row["x"] * (viewport[2]-1),
                           viewport[1] + row["y"] * (viewport[3]-1)))
    result = dict(target_global=list(target_global), sample_count=len(rows),
                  valid_count=len(points), sufficient=len(points) >= 15)
    if not points:
        return dict(result, median_error_px=None, p90_error_px=None,
                    bias_x_px=None, bias_y_px=None, median_gaze_global=None)
    errors = sorted(math.hypot(x-target_global[0], y-target_global[1]) for x, y in points)
    # Explicit nearest-rank P90, rather than an unspecified percentile convention.
    center = [statistics.median(p[i] for p in points) for i in (0, 1)]
    return dict(result, median_error_px=statistics.median(errors),
                p90_error_px=errors[max(0, math.ceil(.9*len(errors))-1)],
                bias_x_px=center[0]-target_global[0], bias_y_px=center[1]-target_global[1],
                median_gaze_global=center)


class ValidationDialog(QDialog):
    def __init__(self, source, screen, folder, parent=None, owns_source=False):
        super().__init__(parent)
        self.source, self.screen_ref = source, screen
        self.folder, self.owns_source = Path(folder), owns_source
        self.viewport = tuple(source.viewport)
        self.original_geometry = screen.geometry()
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Проверка точности взгляда — 9 точек")
        self.setGeometry(screen.geometry())
        self.mode, self.index = "waiting", 0
        self.last_good = -math.inf
        self.rows, self.results, self.raw = [], [], []
        self.status = "Смотрите в центр жёлтой точки. Space — измерить. Esc — закрыть."
        self.report_path = None
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.screen_ref.geometryChanged.connect(self.display_changed)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def display_changed(self, geometry):
        if geometry != self.original_geometry:
            self.mode = "invalid"
            self.status = "Экран изменился. Закройте проверку и запустите её заново."

    def target_global(self):
        origin = self.mapToGlobal(QPoint(0, 0))
        x, y = TARGETS[self.index]
        return (origin.x()+x*(self.width()-1), origin.y()+y*(self.height()-1))

    def start_point(self):
        if self.mode != "waiting":
            return
        if time.monotonic()-self.last_good > .5:
            self.status = "Нет свежего сигнала MEDIUM/HIGH. Проверьте Beam и снова нажмите Space."
            self.update()
            return
        self.rows = []
        self.target = self.target_global()
        self.start_mono = time.monotonic()
        self.mode = "measuring"

    def tick(self):
        now = time.monotonic()
        for seq, point, confidence, status, stamp, mono, sdk_stamp in self.source.drain():
            valid = (point is not None and confidence in {"MEDIUM", "HIGH"}
                     and 0 <= now-mono <= .5 and all(math.isfinite(v) for v in point))
            if valid:
                self.last_good = mono
            if self.mode == "measuring" and self.start_mono+.75 <= mono < self.start_mono+2.25:
                self.rows.append(dict(sequence=seq, host_utc=stamp, monotonic=mono,
                    sdk_timestamp=sdk_stamp, confidence=confidence, valid=valid,
                    x=point[0] if point else None, y=point[1] if point else None))
        if self.mode == "measuring" and now-self.start_mono >= 2.25:
            summary = summarize_target(self.rows, self.target, self.viewport)
            summary["target_index"] = self.index
            self.results.append(summary)
            self.raw.append(dict(target_index=self.index, samples=self.rows))
            self.index += 1
            if self.index == len(TARGETS):
                self.mode = "results"
                self.save_report()
            else:
                self.mode = "waiting"
                self.status = "Смотрите в центр новой точки. Space — измерить. Esc — закрыть."
        self.update()

    def save_report(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.report_path = self.folder / (datetime.now().strftime("%Y%m%d-%H%M%S-")+uuid.uuid4().hex[:8]+".json")
        origin = self.mapToGlobal(QPoint(0, 0))
        measured = [r["median_error_px"] for r in self.results if r["sufficient"]]
        self.report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
            source="Beam Eye Tracker SDK", screen_name=self.screen_ref.name(), viewport=self.viewport,
            window_origin_global=[origin.x(), origin.y()], window_size=[self.width(), self.height()],
            device_pixel_ratio=self.screen_ref.devicePixelRatio(), unit="Qt logical pixels",
            settle_seconds=.75, measurement_seconds=1.5, minimum_valid_samples=15,
            sufficient_targets=len(measured), median_of_target_medians_px=statistics.median(measured) if measured else None,
            results=self.results, raw=self.raw,
            note="Validation only. No correction applied. Results depend on following the target; not visual-angle accuracy.")
        try:
            atomic_json(self.report_path, self.report)
            self.status = f"Отчёт сохранён: {self.report_path.name}. Esc — вернуться."
        except OSError as exc:
            self.status = f"Не удалось сохранить отчёт: {exc}. Esc — вернуться."

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() in {Qt.Key.Key_Escape, Qt.Key.Key_Q}:
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.start_point()

    def event(self, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.WindowDeactivate and getattr(self, "mode", None) == "measuring":
            self.mode = "waiting"
            self.rows = []
            self.status = "Окно было скрыто. Эта точка не засчитана. Space — повторить её."
        return super().event(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191d"))
        painter.setFont(QFont("Helvetica", 16))
        painter.setPen(QColor("#eeeeee"))
        if self.mode == "results":
            enough = self.report["sufficient_targets"]
            median = self.report["median_of_target_medians_px"]
            text = f"Измерено точек: {enough}/9"
            if median is not None:
                text += f" · Медиана ошибок по мишеням: {median:.0f} логических px"
            painter.drawText(30, 35, text)
            for row in self.results:
                tx, ty = TARGETS[row["target_index"]]
                target = QPointF(tx*(self.width()-1), ty*(self.height()-1))
                gaze = row["median_gaze_global"]
                if gaze:
                    origin = self.mapToGlobal(QPoint(0, 0))
                    estimated = QPointF(gaze[0]-origin.x(), gaze[1]-origin.y())
                    painter.setPen(QPen(QColor("#ffb56b"), 2))
                    painter.drawLine(target, estimated)
                    painter.setBrush(QColor("#ffb56b"))
                    painter.drawEllipse(estimated, 5, 5)
                painter.setPen(QColor("#eeeeee"))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(target, 9, 9)
                label = f"{row['median_error_px']:.0f} px" if row["median_error_px"] is not None else "Нет сигнала"
                if not row["sufficient"]:
                    label += " · мало точек"
                painter.drawText(target+QPointF(-35, 35), label)
        elif self.mode != "invalid":
            painter.drawText(30, 35, f"Проверка точности {self.index+1}/9 · Это измерение, калибровка Beam не меняется")
            x, y = TARGETS[self.index]
            target = QPointF(x*(self.width()-1), y*(self.height()-1))
            painter.setBrush(QColor("#ffd166"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(target, 12, 12)
            painter.setBrush(QColor("#17212a"))
            painter.drawEllipse(target, 2, 2)
            if self.mode == "measuring":
                self.status = "Продолжайте смотреть на точку…"
        painter.setPen(QColor("#dddddd"))
        painter.drawText(30, self.height()-25, self.status)
        painter.end()

    def closeEvent(self, event):
        if self.owns_source:
            self.source.requestInterruption()
            if not self.source.wait(3000):
                self.status = "Ожидание завершения Beam. Повторите выход."
                event.ignore()
                return
        self.timer.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=int, default=None)
    parser.add_argument("--output", type=Path, default=BASE / "collected/p001/validation")
    args = parser.parse_args()
    qt = QApplication(sys.argv[:1])
    screens = qt.screens()
    if args.screen is not None and not 0 <= args.screen < len(screens):
        parser.error("Неверный номер экрана")
    screen = screens[args.screen] if args.screen is not None else qt.primaryScreen()
    g = screen.geometry()
    source = BeamSource((g.x(), g.y(), g.width(), g.height()), min_confidence=0, capture_queue=True)
    window = ValidationDialog(source, screen, args.output, owns_source=True)
    window.showFullScreen()
    source.start()
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
