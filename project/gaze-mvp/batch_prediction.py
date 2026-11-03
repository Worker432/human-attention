"""Sequential folder prediction, resumable through the shared prediction cache."""
import csv
from datetime import datetime
import hashlib
from pathlib import Path

from PySide6.QtCore import QTimer, Signal, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QListWidget, QProgressBar, QPushButton, QVBoxLayout)

from gaze_analysis import atomic_json
from heatmap_gallery import HeatmapGallery
from prediction_metrics import object_attention, HEATMAP_VARIANTS


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def image_files(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    excluded = destination if destination != source and destination.is_relative_to(source) else None
    return sorted((p for p in source.rglob('*') if p.is_file()
                   and p.suffix.lower() in EXTENSIONS
                   and not p.name.endswith('.heatmap.png')
                   and (excluded is None or not p.resolve().is_relative_to(excluded))), key=lambda p: str(p).lower())


class BatchPredictionPanel(QDialog):
    completed = Signal()

    def __init__(self, cache_folder, steps=8, seed=42, opacity=.65, parent=None):
        super().__init__(parent)
        self.cache_folder = Path(cache_folder)
        self.steps, self.seed, self.intensity = steps, seed, opacity
        self.source = self.destination = self.run_folder = None
        self.files, self.records, self.csv_rows = [], [], []
        self.index = 0
        self.running = False
        self.job = None
        self.setWindowTitle('Обработать папку')
        self.setObjectName('predictionWindow')
        self.resize(760, 580)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        title = QLabel('Heatmap для папки')
        title.setObjectName('pageTitle')
        layout.addWidget(title)
        note = QLabel('Для каждого баннера: общая heatmap и доли предсказанных точек по объектам.\nВложенные папки тоже обрабатываются. Исходные изображения сохраняются.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.source_button = QPushButton('Выбрать папку с изображениями…')
        self.source_button.clicked.connect(self.choose_source)
        form.addRow('Изображения', self.source_button)
        self.dest_button = QPushButton('Куда сохранить результаты…')
        self.dest_button.clicked.connect(self.choose_destination)
        form.addRow('Результаты', self.dest_button)
        form.addRow('Вариантов на баннер', QLabel(str(HEATMAP_VARIANTS)))
        form.addRow('Точек на вариант', QLabel(str(steps)))
        layout.addLayout(form)
        self.status = QLabel('Выберите исходную папку и место сохранения.')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)
        self.log = QListWidget()
        layout.addWidget(self.log, 1)
        footer = QHBoxLayout()
        self.start_button = QPushButton('Начать обработку')
        self.start_button.setProperty('primary', True)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        footer.addWidget(self.start_button)
        self.stop_button = QPushButton('Остановить')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        footer.addWidget(self.stop_button)
        footer.addStretch()
        close_button = QPushButton('Закрыть')
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        layout.addLayout(footer)

    def choose_source(self):
        name = QFileDialog.getExistingDirectory(self, 'Папка с рекламными изображениями')
        if name:
            self.source = Path(name)
            self.configure()

    def choose_destination(self):
        name = QFileDialog.getExistingDirectory(self, 'Папка для результатов')
        if name:
            self.destination = Path(name)
            self.configure()

    def configure(self):
        self.run_folder = None
        self.files, self.records, self.csv_rows = [], [], []
        self.index = 0
        self.log.clear()
        if self.source:
            self.source = self.source.resolve()
            self.source_button.setText(self.source.name)
            self.source_button.setToolTip(str(self.source))
        if self.destination:
            self.destination = self.destination.resolve()
            self.dest_button.setText(self.destination.name)
            self.dest_button.setToolTip(str(self.destination))
        if self.source and self.destination:
            self.files = image_files(self.source, self.destination)
            self.status.setText(f'Найдено изображений: {len(self.files)}. Результаты появятся в отдельной новой папке.')
            self.progress.setRange(0, max(1, len(self.files)))
            self.progress.setValue(0)
        self.start_button.setEnabled(bool(self.files))

    def controls(self):
        self.source_button.setEnabled(not self.running)
        self.dest_button.setEnabled(not self.running)
        self.start_button.setEnabled(not self.running and self.index < len(self.files))
        self.stop_button.setEnabled(self.running)

    def start(self):
        if self.running or not self.files or self.index >= len(self.files):
            return
        try:
            if self.run_folder is None:
                self.run_folder = self.destination / ('attention-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
                self.run_folder.mkdir(parents=True, exist_ok=False)
            self.running = True
            self.persist('running')
        except OSError as exc:
            self.running = False
            self.status.setText(f'Не удалось создать результаты: {exc}')
        self.controls()
        if self.running:
            QTimer.singleShot(0, self.process_next)

    def process_next(self):
        if not self.running or self.job:
            return
        if self.index == len(self.files):
            self.running = False
            try:
                self.persist('completed')
            except OSError as exc:
                self.status.setText(f'Расчёт завершён, но отчёт не сохранён: {exc}')
                self.controls()
                return
            failed = sum(r['status'] == 'failed' for r in self.records)
            self.status.setText(f'Готово: {self.index-failed}, ошибок: {failed}.\nРезультаты: {self.run_folder}')
            self.controls()
            self.completed.emit()
            return
        path = self.files[self.index]
        self.status.setText(f'{self.index+1} из {len(self.files)} · {path.name}')
        try:
            data = path.read_bytes()
            image = QImage.fromData(data)
            if image.isNull():
                raise ValueError('Не удалось прочитать изображение')
            sha = hashlib.sha256(data).hexdigest()
            self.job = HeatmapGallery(path, sha, image, self.cache_folder,
                self.steps, self.seed, self.intensity, self, variants=HEATMAP_VARIANTS)
            self.job.heat_updated.connect(self.show_progress)
            self.job.generation_finished.connect(self.image_finished)
            self.job.generation_failed.connect(self.image_failed)
        except (OSError, ValueError) as exc:
            self.image_failed(str(exc))

    def show_progress(self, heat, count):
        self.status.setText(f'{self.index+1} из {len(self.files)} · {self.files[self.index].name}\n'
                            f'Варианты: {count} из {HEATMAP_VARIANTS}')

    def image_finished(self):
        if not self.running or not self.job:
            return
        try:
            path = self.files[self.index]
            relative = path.relative_to(self.source)
            target = self.run_folder / relative.parent / (relative.name+'.heatmap.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not self.job.combined_image().save(str(target), 'PNG'):
                raise OSError('Не удалось сохранить heatmap')
            predictions = [r for _, r in sorted(self.job.results.items())]
            metrics = object_attention(predictions)
            atomic_json(target.with_suffix('.json'), metrics)
            for row in metrics['rows']:
                self.csv_rows.append(dict(image=str(relative), data_source='model_prediction',
                    variants=metrics['variants'], total_points=metrics['total_points'],
                    object_id=row['object_id'], class_name=row['class_name'], assignment=row['assignment'],
                    points=row['count'], share=row['share']))
            self.records.append(dict(image=str(relative), status='completed', heatmap=str(target.relative_to(self.run_folder))))
            self.advance_image(f'Готово · {relative}')
        except (OSError, ValueError) as exc:
            self.image_failed(str(exc))

    def image_failed(self, message):
        if not self.running:
            return
        relative = self.files[self.index].relative_to(self.source)
        self.records.append(dict(image=str(relative), status='failed', error=message))
        self.advance_image(f'Ошибка · {relative}: {message}')

    def advance_image(self, message):
        job, self.job = self.job, None
        if job:
            job.deleteLater()
        self.log.addItem(message)
        self.log.scrollToBottom()
        self.index += 1
        self.progress.setValue(self.index)
        try:
            self.persist('running')
        except OSError as exc:
            self.running = False
            self.status.setText(f'Не удалось сохранить отчёт: {exc}')
            self.controls()
            return
        QTimer.singleShot(0, self.process_next)

    def persist(self, state):
        atomic_json(self.run_folder/'manifest.json', dict(schema_version=1,
            data_source='model_prediction', state=state, source=str(self.source),
            variants=HEATMAP_VARIANTS, steps=self.steps, seed=self.seed,
            images_total=len(self.files), processed=self.index, records=self.records))
        fields = ['image', 'data_source', 'variants', 'total_points', 'object_id', 'class_name', 'assignment', 'points', 'share']
        temp = self.run_folder/'attention.csv.part'
        with temp.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.csv_rows)
        temp.replace(self.run_folder/'attention.csv')

    def stop(self):
        self.running = False
        job, self.job = self.job, None
        if job:
            job.cancel()
            job.deleteLater()
        if self.run_folder:
            try:
                self.persist('stopped')
            except OSError as exc:
                self.status.setText(f'Ошибка сохранения отчёта: {exc}')
                self.controls()
                return
        self.status.setText('Обработка остановлена. Готовые прогнозы сохранены; можно продолжить.')
        self.start_button.setText('Продолжить')
        self.controls()

    def closeEvent(self, event):
        if self.running:
            self.stop()
        event.accept()

    def reject(self):
        if self.running:
            self.stop()
        super().reject()
