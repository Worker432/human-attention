"""Read-only viewer for recorded gaze, usable inside the collector or alone."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

from PySide6.QtCore import QPointF, QProcess, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout,
                              QLabel, QListWidget, QListWidgetItem, QPushButton,
                              QSlider, QSplitter, QTableWidget, QTableWidgetItem,
                              QHeaderView, QVBoxLayout, QWidget)
from gaze_analysis import load_analysis

BASE = Path(__file__).resolve().parent
STATUSES = {
    "completed": "Полный просмотр", "skipped": "Пропущено",
    "interrupted_pause": "Пауза", "interrupted_exit": "Выход",
    "interrupted_focus": "Окно скрыто", "interrupted_tracking": "Потерян сигнал",
    "interrupted_crash": "Сбой приложения", "display_changed": "Изменён экран",
    "unreadable": "Файл не прочитан", "presenting": "Запись идёт",
}


class RecordingReader:
    def __init__(self, database):
        self.database = Path(database).resolve()

    def connect(self):
        db = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def trials(self):
        db = self.connect()
        try:
            return [dict(r) for r in db.execute('''SELECT t.*,i.path FROM trials t
                JOIN images i ON i.id=t.image_id ORDER BY t.onset_utc DESC''')]
        finally:
            db.close()

    def load(self, trial_id):
        db = self.connect()
        try:
            db.execute("BEGIN")
            trial = db.execute('''SELECT t.*,i.path FROM trials t JOIN images i
                ON i.id=t.image_id WHERE t.id=?''', (trial_id,)).fetchone()
            if trial is None:
                raise ValueError("Запись не найдена. Обновите список.")
            protocol = json.loads(db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0])
            samples = [dict(r) for r in db.execute('''SELECT * FROM samples WHERE trial_id=?
                ORDER BY elapsed,sequence''', (trial_id,))]
            return dict(trial), samples, protocol
        finally:
            db.close()


def drawable(sample):
    return (sample["valid"] and sample["inside_image"]
            and all(sample[k] is not None and math.isfinite(sample[k]) for k in ("image_x", "image_y", "elapsed")))


class GazeCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.samples = []
        self.position = 0.
        self.analysis = None
        self.show_boxes = True
        self.empty_message = "Выберите сохранённый просмотр слева"
        self.setMinimumSize(280, 240)

    def set_recording(self, image, samples, position):
        self.image, self.samples, self.position = image, samples, position
        self.update()

    def visible_points(self):
        return [s for s in self.samples if s["elapsed"] <= self.position and drawable(s)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191d"))
        if self.image.isNull():
            painter.setPen(QColor("#d3d8df"))
            painter.drawText(self.rect().adjusted(20, 20, -20, -20),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_message)
            return
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        rect = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        painter.drawImage(rect, self.image)

        def point(row):
            return QPointF(rect.x() + row["image_x"] * scale, rect.y() + row["image_y"] * scale)

        painter.setClipRect(rect)
        if self.analysis and self.show_boxes:
            labels = []
            for box in self.analysis["objects"]:
                x1, y1, x2, y2 = box["xyxy"]
                bounds = QRectF(rect.x()+x1*scale, rect.y()+y1*scale, (x2-x1)*scale, (y2-y1)*scale)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#ffb56b"), 1.5))
                painter.drawRect(bounds)
                text = f"#{box['object_id']} {box['class_name']} {box['confidence']:.0%}"
                metrics = painter.fontMetrics()
                tw = min(metrics.horizontalAdvance(text) + 10, rect.width())
                tx = min(max(rect.x(), bounds.x()), rect.right()-tw)
                ty = max(rect.y(), bounds.y()-metrics.height()-4)
                label = QRectF(tx, ty, tw, metrics.height()+4)
                # Keep labels readable when neighbouring objects share a top edge.
                for offset in range(len(self.analysis["objects"]) + 1):
                    candidate_y = min(rect.bottom()-label.height(), ty + offset*(label.height()+2))
                    candidate = QRectF(tx, candidate_y, tw, label.height())
                    if not any(candidate.adjusted(-2, -1, 2, 1).intersects(other[0]) for other in labels):
                        label = candidate
                        break
                labels.append((label, metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(tw-6))))
            for label, text in labels:
                painter.fillRect(label, QColor("#26201b"))
                painter.setPen(QColor("#ffcf98"))
                painter.drawText(label.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, text)
        painter.setPen(QPen(QColor("#55dcff"), 2))
        painter.setBrush(QColor("#55dcff"))
        previous, first, last = None, None, None
        for row in self.samples:
            if row["elapsed"] > self.position:
                break
            if not drawable(row):
                previous = None
                continue
            current = point(row)
            if previous is not None and row["elapsed"] - previous["elapsed"] <= .1:
                painter.drawLine(point(previous), current)
            painter.drawEllipse(current, 2.5, 2.5)
            if first is None:
                first = current
            last, previous = current, row
        painter.setPen(QPen(QColor("#17212a"), 2))
        if first is not None:
            painter.drawPolygon(QPolygonF([first + QPointF(0, -10), first + QPointF(9, 7), first + QPointF(-9, 7)]))
        if last is not None:
            painter.setBrush(QColor("#ffc857"))
            painter.drawRect(QRectF(last.x() - 6, last.y() - 6, 12, 12))
        painter.end()


class ReviewPanel(QDialog):
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.reader = RecordingReader(database)
        self.trial = None
        self.playing = False
        self.duration = 0.
        self.analysis = None
        self.worker = None
        self.worker_output = ""
        self.setWindowTitle("Записи взгляда")
        self.resize(1180, 820)
        self.setStyleSheet('''QDialog {background:#20242b; color:#edf0f5;}
            QLabel {color:#edf0f5;} QListWidget {background:#171b21; color:#edf0f5; border:0;}
            QListWidget::item {padding:10px;} QListWidget::item:selected {background:#36506d;}
            QPushButton {padding:8px 14px; border-radius:6px; background:#3a4655; color:#ffffff;}
            QPushButton:disabled {color:#9ba4b0; background:#2a3039;}
            QPushButton:hover {background:#4b5f76;}''')
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("Сохранённые просмотры")
        top.addWidget(title)
        top.addStretch()
        self.analyze_button = QPushButton("Обработать новые · YOLO")
        self.analyze_button.clicked.connect(self.start_analysis)
        top.addWidget(self.analyze_button)
        self.refresh_button = QPushButton("Обновить список")
        self.refresh_button.clicked.connect(self.refresh)
        top.addWidget(self.refresh_button)
        close_button = QPushButton("Вернуться к сбору" if parent else "Закрыть")
        close_button.clicked.connect(self.close)
        top.addWidget(close_button)
        layout.addLayout(top)
        split = QSplitter()
        self.list = QListWidget()
        self.list.setMinimumWidth(210)
        self.list.currentItemChanged.connect(self.select_trial)
        split.addWidget(self.list)
        right = QWidget()
        content = QVBoxLayout(right)
        content.setContentsMargins(8, 0, 0, 0)
        self.summary = QLabel("Пока нет записей")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self.summary)
        self.canvas = GazeCanvas()
        content.addWidget(self.canvas, 1)
        self.legend = QLabel("△ Начало   →   □ Текущая точка / конец · Пропуски сигнала разрывают линию")
        self.legend.setWordWrap(True)
        content.addWidget(self.legend)
        playback = QHBoxLayout()
        self.play_button = QPushButton("Воспроизвести")
        self.play_button.clicked.connect(self.toggle_playback)
        playback.addWidget(self.play_button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 5000)
        self.slider.setAccessibleName("Время от начала показа, миллисекунды")
        self.slider.sliderPressed.connect(self.stop)
        self.slider.valueChanged.connect(self.seek)
        playback.addWidget(self.slider, 1)
        self.time_label = QLabel("0,00 / 0,00 с")
        playback.addWidget(self.time_label)
        content.addLayout(playback)
        options = QHBoxLayout()
        self.box_toggle = QCheckBox("Рамки объектов и классы")
        self.box_toggle.setChecked(True)
        self.box_toggle.setStyleSheet("color:#edf0f5;")
        self.box_toggle.toggled.connect(self.toggle_boxes)
        options.addWidget(self.box_toggle)
        self.analysis_label = QLabel("Объекты ещё не обработаны")
        self.analysis_label.setWordWrap(True)
        options.addWidget(self.analysis_label, 1)
        content.addLayout(options)
        self.fixation_table = QTableWidget(0, 5)
        self.fixation_table.setHorizontalHeaderLabels(["Фиксация", "Начало, с", "Длительность, с", "Объект", "Сопоставление"])
        self.fixation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.fixation_table.horizontalHeader().setStretchLastSection(True)
        self.fixation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.fixation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.fixation_table.setMaximumHeight(160)
        self.fixation_table.setStyleSheet("QTableWidget {background:#171b21; color:#edf0f5;} QHeaderView::section {background:#364456; color:white; padding:4px;}")
        self.fixation_table.cellClicked.connect(self.select_fixation)
        content.addWidget(self.fixation_table)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self.progress_label)
        self.notice = QLabel("Исходные точки Beam без сглаживания. Доля уверенных точек не измеряет точность калибровки.")
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        content.addWidget(self.notice)
        self.save_button = QPushButton("Сохранить картинку с траекторией…")
        self.save_button.clicked.connect(self.save_image)
        content.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignLeft)
        split.addWidget(right)
        split.setSizes([240, 900])
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance)
        self.refresh()

    def refresh(self):
        self.stop()
        selected = self.trial["id"] if self.trial else None
        self.list.blockSignals(True)
        self.list.clear()
        try:
            trials = self.reader.trials()
        except (sqlite3.Error, OSError, ValueError) as exc:
            self.summary.setText("Не удалось открыть записи")
            self.canvas.empty_message = str(exc)
            trials = []
        selected_row = 0
        for i, trial in enumerate(trials):
            item = QListWidgetItem(f"{trial['path']}\n{STATUSES.get(trial['status'], trial['status'])}")
            item.setData(Qt.ItemDataRole.UserRole, trial["id"])
            self.list.addItem(item)
            if trial["id"] == selected:
                selected_row = i
        self.list.blockSignals(False)
        if trials:
            self.list.setCurrentRow(selected_row)
        else:
            self.trial = None
            self.canvas.set_recording(QImage(), [], 0.)
            self.show_analysis(None)
            self.set_controls(False)

    def set_controls(self, enabled):
        self.play_button.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.save_button.setEnabled(enabled and not self.canvas.image.isNull())

    def select_trial(self, item, previous=None):
        self.stop()
        if item is None:
            return
        try:
            trial, samples, protocol = self.reader.load(item.data(Qt.ItemDataRole.UserRole))
            path = Path(protocol["image_root"]) / trial["path"]
            image = QImage()
            warning = ""
            try:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != trial["image_sha256"]:
                    warning = "Файл изображения изменён. Траектория скрыта, чтобы не показывать неверное наложение."
                else:
                    image = QImage.fromData(data)
                    if image.isNull():
                        warning = "Изображение не удалось прочитать."
            except OSError:
                warning = f"Исходное изображение не найдено: {path}"
            geometry = json.loads(trial["geometry"])
            if not image.isNull() and (image.width(), image.height()) != (geometry.get("image_width"), geometry.get("image_height")):
                image = QImage()
                warning = "Размер изображения отличается от сохранённого. Наложение скрыто."
            self.trial = trial
            last_sample = max((s["elapsed"] for s in samples), default=0.)
            recorded_end = min(float(protocol["duration"]), trial["duration"]) if trial["duration"] is not None else last_sample
            self.duration = max(recorded_end, last_sample, .001)
            self.canvas.empty_message = warning or "Изображение недоступно"
            self.canvas.set_recording(image, samples, self.duration)
            self.show_analysis(load_analysis(self.reader.database, trial, samples) if not image.isNull() else None)
            total = len(samples)
            valid = sum(bool(s["valid"]) for s in samples)
            inside = sum(bool(drawable(s)) for s in samples)
            actual = f"{trial['duration']:.2f} с" if trial["duration"] is not None else "не зафиксирована"
            percentage = 100 * valid / total if total else 0
            self.summary.setText(f"{trial['path']} · {STATUSES.get(trial['status'], trial['status'])}\n"
                                 f"Точек: {total} · Уверенных: {valid} ({percentage:.0f}%) · Внутри изображения: {inside}\n"
                                 f"Длительность в журнале: {actual} · Окно записи: {protocol['duration']:g} с")
            self.notice.setText(warning or "Исходные точки Beam без сглаживания. Фиксации отдельно не выделены; уверенность не измеряет точность калибровки.")
            self.slider.setRange(0, round(self.duration * 1000))
            self.slider.setValue(self.slider.maximum())
            self.seek(self.slider.value())
            self.set_controls(bool(samples))
        except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            self.trial = None
            self.summary.setText(f"Не удалось открыть просмотр: {exc}")
            self.canvas.set_recording(QImage(), [], 0.)
            self.show_analysis(None)
            self.set_controls(False)

    def toggle_boxes(self, enabled):
        self.canvas.show_boxes = enabled
        self.canvas.update()

    def show_analysis(self, result):
        self.analysis = self.canvas.analysis = result
        self.fixation_table.setRowCount(0)
        if result is None:
            self.analysis_label.setText("Нажмите «Обработать новые · YOLO»")
            self.canvas.update()
            return
        objects = len(result["objects"])
        fixations = result["fixations"]
        assigned = sum(f["assignment"] == "assigned" for f in fixations)
        outside = sum(f["assignment"] == "outside" for f in fixations)
        ambiguous = len(fixations) - assigned - outside
        self.analysis_label.setText(f"Объектов: {objects} · Фиксаций: {len(fixations)} · В объектах: {assigned} · Вне: {outside} · Спорно: {ambiguous} · Переходов: {len(result['transitions'])}")
        self.fixation_table.setRowCount(len(fixations))
        for row, fixation in enumerate(fixations):
            state = {"assigned": "Однозначно", "ambiguous": "Несколько рамок", "outside": "Вне объектов"}[fixation["assignment"]]
            target = f"#{fixation['object_id']} {fixation['class_name']}" if fixation["object_id"] else "—"
            if fixation["assignment"] == "ambiguous":
                target = ", ".join(f"#{n}" for n in fixation["candidate_ids"])
            for col, value in enumerate([str(fixation["id"]), f"{fixation['start']:.2f}", f"{fixation['duration']:.2f}", target, state]):
                self.fixation_table.setItem(row, col, QTableWidgetItem(value))
        self.canvas.update()

    def select_fixation(self, row, column):
        if self.analysis and row < len(self.analysis["fixations"]):
            self.stop()
            self.slider.setValue(round(self.analysis["fixations"][row]["end"] * 1000))

    def start_analysis(self):
        if self.worker is not None:
            return
        python = BASE / ".venv-detector/bin/python"
        if not python.exists():
            self.progress_label.setText("Сначала установите детектор: откройте «Настроить детектор.command».")
            return
        if not self.reader.database.exists():
            self.progress_label.setText("Сначала соберите хотя бы один просмотр.")
            return
        self.stop()
        self.worker_output = ""
        self.analyze_button.setEnabled(False)
        self.progress_label.setText("Запускается YOLO11m…")
        self.worker = QProcess(self)
        self.worker.setWorkingDirectory(str(BASE))
        self.worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.worker.readyReadStandardOutput.connect(self.analysis_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.errorOccurred.connect(self.analysis_error)
        self.worker.start(str(python), ["-u", str(BASE / "analyze_gaze.py"), "--database", str(self.reader.database)])

    def analysis_progress(self):
        if self.worker:
            self.worker_output += bytes(self.worker.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.worker_output = self.worker_output[-8000:]
            lines = [line for line in self.worker_output.splitlines() if line.strip()]
            if lines:
                self.progress_label.setText(lines[-1][-500:])

    def analysis_finished(self, code, status):
        self.analysis_progress()
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        self.analyze_button.setEnabled(True)
        if code != 0:
            self.progress_label.setText("Обработка завершилась с ошибкой. " + self.worker_output[-600:])
        self.refresh()

    def analysis_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.progress_label.setText("Не удалось запустить окружение детектора. Откройте «Настроить детектор.command».")
            self.worker.deleteLater()
            self.worker = None
            self.analyze_button.setEnabled(True)

    def seek(self, value):
        self.canvas.position = value / 1000
        self.time_label.setText(f"{value/1000:.2f} / {self.duration:.2f} с".replace(".", ","))
        self.canvas.update()
        if self.playing:
            self.epoch = time.monotonic() - value / 1000

    def toggle_playback(self):
        if self.playing:
            self.stop()
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self.epoch = time.monotonic() - self.slider.value() / 1000
        self.playing = True
        self.play_button.setText("Пауза")
        self.timer.start()

    def advance(self):
        position = min(self.slider.maximum(), round((time.monotonic() - self.epoch) * 1000))
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        self.canvas.position = position / 1000
        self.time_label.setText(f"{position/1000:.2f} / {self.duration:.2f} с".replace(".", ","))
        self.canvas.update()
        if position >= self.slider.maximum():
            self.stop()

    def stop(self):
        self.playing = False
        if hasattr(self, "timer"):
            self.timer.stop()
        self.play_button.setText("Воспроизвести")

    def save_image(self):
        self.stop()
        name, _ = QFileDialog.getSaveFileName(self, "Сохранить траекторию",
                     str(Path.home() / f"gaze-{Path(self.trial['path']).stem}.png"), "PNG (*.png)")
        if name:
            if not name.lower().endswith(".png"):
                name += ".png"
            if self.canvas.grab().save(name, "PNG"):
                self.notice.setText(f"Сохранено: {name}")
            else:
                self.notice.setText("Не удалось сохранить файл. Выберите другую папку.")

    def closeEvent(self, event):
        if self.worker is not None:
            self.progress_label.setText("Дождитесь завершения обработки YOLO, затем закройте панель.")
            event.ignore()
            return
        self.stop()
        event.accept()

    def reject(self):
        if self.worker is not None:
            self.progress_label.setText("Дождитесь завершения обработки YOLO.")
            return
        self.stop()
        super().reject()


def main():
    parser = argparse.ArgumentParser(description="View saved gaze trajectories without Beam")
    parser.add_argument("--database", type=Path, default=BASE / "collected/p001/dataset.sqlite3")
    args = parser.parse_args()
    qt = QApplication(sys.argv[:1])
    window = ReviewPanel(args.database)
    window.show()
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
