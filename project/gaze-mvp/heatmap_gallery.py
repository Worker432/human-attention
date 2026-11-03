"""One heatmap from pooled model predictions, with progress and cancellation."""
from pathlib import Path
import time

from PySide6.QtCore import QProcess, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from attention_heatmap import HeatmapDensity
from scanpath_prediction import BASE, load_prediction, prediction_file


def composite(image, heat, opacity):
    output = image.convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(output)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.setOpacity(opacity)
    painter.drawImage(QRectF(0, 0, output.width(), output.height()), heat)
    painter.end()
    return output


class HeatmapTile(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.message = "Ожидает расчёта"
        self.setMinimumSize(150, 160)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#e9e8e2'))
        if self.image.isNull():
            painter.setPen(QColor('#72776f'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        painter.drawImage(QRectF((self.width()-w)/2, (self.height()-h)/2, w, h), self.image)


class HeatmapGallery(QDialog):
    heat_updated = Signal(QImage, int)
    generation_failed = Signal(str)
    generation_finished = Signal()
    progress_updated = Signal(str)
    def __init__(self, image_path, image_sha, image, folder, steps=8, seed=42,
                 opacity=.65, parent=None, variants=10):
        super().__init__(parent)
        self.image_path, self.image_sha = Path(image_path), image_sha
        self.image = image.copy()
        self.folder, self.steps = Path(folder), steps
        if variants not in (10, 30, 100):
            raise ValueError('Число вариантов: 10, 30 или 100')
        self.count = variants
        self.seeds = [(seed+i) % 1000000 for i in range(variants)]
        self.results = {}
        self.density = HeatmapDensity(image.width(), image.height())
        self.heat = QImage()
        self.worker, self.active_index = None, None
        self.closing = False
        self.log = ''
        self.started_at = None
        self.current_variant = '1'
        self.current_step = '0'
        self.poll = QTimer(self)
        self.poll.setInterval(1000)
        self.poll.timeout.connect(self.tick)
        self.setWindowTitle(f'Общая heatmap · {variants} вариантов')
        self.setObjectName('predictionWindow')
        self.resize(1380, 850)
        self.setMinimumSize(940, 620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        header = QHBoxLayout()
        title = QLabel('Общая карта внимания')
        title.setObjectName('pageTitle')
        header.addWidget(title)
        header.addStretch()
        self.save_button = QPushButton('Сохранить heatmap…')
        self.save_button.clicked.connect(self.save_heatmap)
        self.save_button.setEnabled(False)
        header.addWidget(self.save_button)
        layout.addLayout(header)
        subtitle = QLabel(f'{self.image_path.name} · Общая карта по {variants} прогнозам, по {steps} точек в каждом.')
        subtitle.setWordWrap(True)
        subtitle.setObjectName('subtitle')
        layout.addWidget(subtitle)
        self.preview = HeatmapTile()
        self.preview.image = self.image.copy()
        layout.addWidget(self.preview, 1)
        footer = QHBoxLayout()
        self.status = QLabel('Проверяем сохранённые варианты…')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        footer.addWidget(QLabel('Интенсивность'))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(round(opacity*100))
        self.opacity.setFixedWidth(130)
        self.opacity.setAccessibleName('Интенсивность общей тепловой карты')
        self.opacity.valueChanged.connect(self.refresh_images)
        footer.addWidget(self.opacity)
        close_button = QPushButton('Закрыть')
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        layout.addLayout(footer)
        QTimer.singleShot(0, self.start_next)

    def accept_result(self, index, result):
        if (result['image_width'], result['image_height']) != (self.image.width(), self.image.height()):
            return False
        if index in self.results:
            return True
        self.results[index] = result
        self.density.add(result['fixations'])
        return True

    def publish_heat(self):
        # Pool coordinates before smoothing and normalizing once. Combining
        # already colorized maps would hide relative density across variants.
        self.heat = self.density.image()
        self.refresh_images()
        self.heat_updated.emit(self.heat.copy(), len(self.results))

    def refresh_images(self):
        self.preview.image = composite(self.image, self.heat, self.opacity.value()/100)
        self.preview.update()

    def start_next(self):
        if self.closing or self.worker:
            return
        self.scan_cached()
        if len(self.results) == self.count:
            self.status.setText(f'Объединены {self.count} вариантов')
            self.save_button.setEnabled(True)
            self.generation_finished.emit()
            return
        executable = BASE / '.venv-detector/bin/python'
        if not executable.exists():
            self.generation_failed.emit('Откройте «Настроить прогноз.command» для установки моделей.')
            return
        self.log = ''
        self.status.setText(f'Готово {len(self.results)} из {self.count} · строим прогнозы…')
        self.started_at = time.monotonic()
        self.poll.start()
        self.worker = QProcess(self)
        self.worker.setWorkingDirectory(str(BASE))
        self.worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.worker.readyReadStandardOutput.connect(self.read_progress)
        self.worker.finished.connect(self.process_finished)
        self.worker.errorOccurred.connect(self.process_error)
        self.worker.start(str(executable), ['-u', str(BASE/'scanpath_prediction.py'),
            str(self.image_path), '--steps', str(self.steps), '--seed', str(self.seeds[0]),
            '--variants', str(self.count), '--aggregate-only', '--output', str(self.folder)])

    def scan_cached(self):
        changed = False
        for index, seed in enumerate(self.seeds):
            if index in self.results:
                continue
            file = prediction_file(self.folder, self.image_sha, self.steps, seed)
            result = load_prediction(file, self.image_sha, require_heatmaps=False)
            if (result and result['parameters']['steps'] == self.steps
                    and result['parameters']['seed'] == seed and self.accept_result(index, result)):
                changed = True
        if changed:
            self.publish_heat()

    def read_progress(self):
        if self.worker:
            chunk = bytes(self.worker.readAllStandardOutput()).decode('utf-8', errors='replace')
            self.log = (self.log + chunk)[-16000:]
            for line in chunk.splitlines():
                if line.startswith('VARIANT_BEGIN '):
                    self.current_variant = line.split()[-1]
                    self.current_step = '0'
                elif line.startswith('Прогноз: '):
                    self.current_step = line.removeprefix('Прогноз: ')
            if not self.closing:
                if 'RESULT ' in chunk:
                    self.scan_cached()
                self.emit_progress()

    def emit_progress(self):
        elapsed = int(time.monotonic()-self.started_at) if self.started_at else 0
        message = (f'Готово {len(self.results)} из {self.count} · вариант {self.current_variant}'
                   f' · точка {self.current_step} · {elapsed} с')
        self.status.setText(message)
        self.progress_updated.emit(message)

    def tick(self):
        if self.worker and not self.closing:
            self.scan_cached()
            self.emit_progress()

    def process_finished(self, code, exit_status):
        self.poll.stop()
        self.read_progress()
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        if self.closing:
            return
        self.scan_cached()
        if code == 0 and len(self.results) == self.count:
            QTimer.singleShot(0, self.start_next)
        else:
            self.status.setText(f'Расчёт остановился: готово {len(self.results)} из {self.count}. Можно повторить — готовые варианты сохраняются.')
            self.status.setToolTip(self.log[-2000:])
            self.generation_failed.emit(self.status.text())

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.poll.stop()
            worker, self.worker = self.worker, None
            if worker:
                worker.deleteLater()
            self.status.setText('Не удалось запустить модель. Откройте «Настроить прогноз.command».')
            self.generation_failed.emit(self.status.text())

    def combined_image(self):
        if len(self.results) != self.count:
            return QImage()
        return composite(self.image, self.heat, self.opacity.value()/100)

    def save_heatmap(self):
        if len(self.results) != self.count:
            return
        name, _ = QFileDialog.getSaveFileName(self, 'Сохранить общую heatmap',
            str(Path.home()/f'heatmap-{self.count}-{self.image_path.stem}.png'), 'PNG (*.png)')
        if name:
            if not name.lower().endswith('.png'):
                name += '.png'
            self.status.setText(f'Сохранено: {name}' if self.combined_image().save(name, 'PNG') else 'Не удалось сохранить')

    def cancel(self):
        self.closing = True
        self.poll.stop()
        if self.worker:
            self.worker.kill()
            self.worker.waitForFinished(3000)

    def closeEvent(self, event):
        self.cancel()
        event.accept()

    def reject(self):
        self.cancel()
        super().reject()
