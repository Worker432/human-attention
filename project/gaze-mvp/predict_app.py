"""Standalone demo: an image in, a pretrained synthetic scanpath out. No Beam."""
import argparse
import hashlib
import math
import csv
from pathlib import Path
import sys

from PySide6.QtCore import (QAbstractAnimation, QEasingCurve, QPointF, QProcess,
                           QPropertyAnimation, QRectF, Qt, QTimer, Signal, Property)
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QTableWidgetItem, QWidget
from scanpath_prediction import BASE, load_prediction, prediction_file
from attention_heatmap import route_heatmap
from prediction_metrics import object_attention, HEATMAP_VARIANTS

CLASS_LABELS = {"person":"Человек", "bicycle":"Велосипед", "car":"Автомобиль", "motorcycle":"Мотоцикл",
    "airplane":"Самолёт", "bus":"Автобус", "train":"Поезд", "truck":"Грузовик", "boat":"Лодка",
    "traffic light":"Светофор", "fire hydrant":"Гидрант", "stop sign":"Знак стоп", "parking meter":"Паркомат",
    "bench":"Скамейка", "bird":"Птица", "cat":"Кошка", "dog":"Собака", "horse":"Лошадь", "sheep":"Овца",
    "cow":"Корова", "elephant":"Слон", "bear":"Медведь", "zebra":"Зебра", "giraffe":"Жираф",
    "backpack":"Рюкзак", "umbrella":"Зонт", "handbag":"Сумка", "tie":"Галстук", "suitcase":"Чемодан",
    "frisbee":"Фрисби", "skis":"Лыжи", "snowboard":"Сноуборд", "sports ball":"Мяч", "kite":"Воздушный змей",
    "baseball bat":"Бита", "baseball glove":"Бейсбольная перчатка", "skateboard":"Скейтборд", "surfboard":"Доска для сёрфинга",
    "tennis racket":"Ракетка", "bottle":"Бутылка", "wine glass":"Бокал", "cup":"Чашка", "fork":"Вилка",
    "knife":"Нож", "spoon":"Ложка", "bowl":"Миска", "banana":"Банан", "apple":"Яблоко", "sandwich":"Сэндвич",
    "orange":"Апельсин", "broccoli":"Брокколи", "carrot":"Морковь", "hot dog":"Хот-дог", "pizza":"Пицца",
    "donut":"Пончик", "cake":"Торт", "chair":"Стул", "couch":"Диван", "potted plant":"Растение",
    "bed":"Кровать", "dining table":"Стол", "toilet":"Унитаз", "tv":"Экран", "laptop":"Ноутбук",
    "mouse":"Мышь", "remote":"Пульт", "keyboard":"Клавиатура", "cell phone":"Телефон", "microwave":"Микроволновка",
    "oven":"Духовка", "toaster":"Тостер", "sink":"Раковина", "refrigerator":"Холодильник", "book":"Книга",
    "clock":"Часы", "vase":"Ваза", "scissors":"Ножницы", "teddy bear":"Игрушка", "hair drier":"Фен", "toothbrush":"Зубная щётка"}


def object_label(point):
    if point["assignment"] == "outside":
        return "Область без рамки"
    if point["assignment"] == "ambiguous":
        return "Пересечение объектов"
    return f"{CLASS_LABELS.get(point['class_name'], point['class_name'])} · {point['object_id']}"


class PredictionCanvas(QWidget):
    image_dropped = Signal(str)
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.result = None
        self.folder = None
        self.step = 0
        self.show_boxes, self.show_heatmap = False, False
        self.show_route = True
        self.heat_opacity = .65
        self.heat_mode = 0
        self.heat_cache = None
        self.aggregate_heat = QImage()
        self._motion_progress = 1.
        self.motion = QPropertyAnimation(self, b'motionProgress', self)
        self.motion.setStartValue(0.)
        self.motion.setEndValue(1.)
        self.motion.setEasingCurve(QEasingCurve.Type.Linear)
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 260)

    def get_motion_progress(self):
        return self._motion_progress

    def set_motion_progress(self, value):
        self._motion_progress = value
        self.update()

    motionProgress = Property(float, get_motion_progress, set_motion_progress)

    def current_point(self):
        if not self.result or self.step < 1:
            return None
        target = self.result['fixations'][self.step-1]
        if self.step == 1:
            return QPointF(target['x'], target['y'])
        origin = self.result['fixations'][self.step-2]
        t = self._motion_progress
        return QPointF(origin['x']+(target['x']-origin['x'])*t,
                       origin['y']+(target['y']-origin['y'])*t)

    def animate_current(self):
        if self.step < 2:
            return False
        origin, target = self.result['fixations'][self.step-2:self.step]
        distance = math.hypot(target['x']-origin['x'], target['y']-origin['y'])
        # Constant image-relative speed; segment duration follows its length.
        speed = min(self.image.width(), self.image.height())*.85
        self.motion.setDuration(max(1, round(distance/speed*1000)))
        self.motion.start()
        return True

    def finish_motion(self):
        self.motion.stop()
        self.set_motion_progress(1.)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#e9e8e2"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(self.rect()), 12, 12)
        if self.image.isNull():
            painter.setPen(QColor("#687160"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Перетащите сюда баннер\nили выберите изображение сверху")
            return
        scale = min((self.width()-40)/self.image.width(), (self.height()-40)/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        rect = QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(rect.adjusted(-1, -1, 1, 1), QColor("#d6d7cf"))
        painter.drawImage(rect, self.image)
        if not self.result:
            return
        painter.setClipRect(rect)
        if self.show_heatmap and (self.step > 0 or self.heat_mode == 2):
            key = (id(self.result), self.step, self.heat_mode)
            if self.heat_mode == 2:
                heat = self.aggregate_heat
            elif not self.heat_cache or self.heat_cache[0] != key:
                heat = (route_heatmap(self.result['fixations'][:self.step], self.image.width(), self.image.height())
                        if self.heat_mode == 0 else (QImage(str(self.folder / self.result['heatmaps'][self.step-1]))
                            if len(self.result['heatmaps']) >= self.step else QImage()))
                self.heat_cache = (key, heat)
            if self.heat_mode != 2:
                heat = self.heat_cache[1]
            if not heat.isNull():
                painter.setOpacity(self.heat_opacity)
                painter.drawImage(rect, heat)
                painter.setOpacity(1.)
        if self.show_boxes:
            labels = []
            for box in self.result["objects"]:
                x1, y1, x2, y2 = box["xyxy"]
                box_rect = QRectF(rect.x()+x1*scale, rect.y()+y1*scale, (x2-x1)*scale, (y2-y1)*scale)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#537762"), 1.3))
                painter.drawRect(box_rect)
                text = f"{CLASS_LABELS.get(box['class_name'],box['class_name'])} · {box['object_id']}"
                metrics = painter.fontMetrics()
                tw = min(rect.width(), metrics.horizontalAdvance(text)+8)
                tx = min(rect.right()-tw, max(rect.x(), box_rect.x()))
                ty = max(rect.y(), box_rect.y()-metrics.height()-4)
                label = QRectF(tx, ty, tw, metrics.height()+4)
                for offset in range(len(self.result["objects"])+1):
                    candidate = QRectF(tx, min(rect.bottom()-label.height(), ty+offset*(label.height()+2)), tw, label.height())
                    if not any(candidate.intersects(other[0]) for other in labels):
                        label = candidate
                        break
                labels.append((label, metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(tw-6))))
            for bounds, text in labels:
                painter.fillRect(bounds, QColor("#f6f8ef"))
                painter.setPen(QColor("#365e3e"))
                painter.drawText(bounds.adjusted(3, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, text)
        points = self.result["fixations"][:self.step] if self.show_route else []
        previous = None
        for point in points:
            current = point is points[-1]
            xy = self.current_point() if current else QPointF(point['x'], point['y'])
            center = QPointF(rect.x()+xy.x()*scale, rect.y()+xy.y()*scale)
            painter.setPen(QPen(QColor("#d87846"), 2.2))
            if previous is not None:
                painter.drawLine(previous, center)
            painter.setBrush(QColor("#d87846" if current else "#fff9ec"))
            painter.setPen(QPen(QColor("#fcf9f0" if current else "#d87846"), 2))
            painter.drawEllipse(center, 11, 11)
            painter.setPen(QColor("#ffffff" if current else "#8f4e2d"))
            painter.drawText(QRectF(center.x()-10, center.y()-10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(point["step"]))
            previous = center
        painter.end()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                self.image_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                break


class PredictionPanel(QDialog):
    def __init__(self, image=None, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.image_sha = None
        self.result = None
        self.worker = None
        self.log = ""
        self.aggregate_job = None
        self.aggregate_key = None
        self.aggregate_complete = False
        self.aggregate_results = []
        self.object_metrics = None
        self.output_folder = BASE / "collected/predictions"
        from prediction_layout import build_interface
        build_interface(self, PredictionCanvas)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(0)  # Start playback without an artificial dwell.
        self.timer.timeout.connect(self.advance)
        self.canvas.motion.finished.connect(self.motion_finished)
        self.set_enabled()
        if image:
            self.set_image(Path(image))

    def set_enabled(self):
        idle = self.worker is None and self.aggregate_job is None
        for widget in (self.open_button, self.steps, self.seed, self.batch_button):
            widget.setEnabled(idle)
        self.predict_button.setEnabled(idle and self.image_path is not None)
        self.variant_button.setEnabled(idle and self.image_path is not None)
        self.gallery_button.setEnabled(idle and self.image_path is not None)
        for widget in (self.play_button, self.slider, self.export_button):
            widget.setEnabled(self.result is not None and idle)
        self.canvas.setAcceptDrops(idle)
        self.heat_mode.setEnabled(idle)
        self.metrics_button.setEnabled(idle and self.object_metrics is not None)
        self.cancel_button.setVisible(not idle)
        self.sequence.setEnabled(idle)
        self.predict_button.setVisible(self.result is None or not idle
            or (self.canvas.heat_mode == 2 and not self.aggregate_complete)
            or (self.canvas.heat_mode == 1 and self.result is not None and not self.result['heatmaps']))
        self.variant_button.setVisible(self.result is not None)
        for widget in (self.boxes, self.heatmap):
            widget.setEnabled(self.result is not None)

    def open_image(self):
        if self.worker or self.aggregate_job:
            return
        name, _ = QFileDialog.getOpenFileName(self, "Выбрать рекламное изображение",
            str(BASE.parent.parent / "dataset/PittImageVideoAdsDataset/pitt_ads_images"), "Изображения (*.jpg *.jpeg *.png *.webp)")
        if name:
            self.set_image(Path(name))

    def open_gallery(self):
        if self.worker or not self.image_path:
            return
        self.stop()
        self.route.setChecked(False)
        self.heat_mode.setCurrentIndex(2)
        self.heatmap.setChecked(True)
        self.ensure_aggregate()

    def ensure_aggregate(self):
        if not self.image_path or self.worker or self.aggregate_job:
            return
        count = HEATMAP_VARIANTS
        key = (self.image_sha, self.steps.value(), self.seed.value(), count)
        if self.aggregate_key == key and self.aggregate_complete:
            return
        from heatmap_gallery import HeatmapGallery
        self.aggregate_key = key
        self.aggregate_complete = False
        self.aggregate_results = []
        self.canvas.aggregate_heat = QImage()
        job = HeatmapGallery(self.image_path, self.image_sha, self.canvas.image,
            self.output_folder, self.steps.value(), self.seed.value(),
            self.opacity.value()/100, self, variants=count)
        self.aggregate_job = job
        job.heat_updated.connect(self.aggregate_updated)
        job.generation_failed.connect(self.aggregate_failed)
        job.generation_finished.connect(self.aggregate_finished)
        job.progress_updated.connect(self.status.setText)
        self.status.setText(f'Строим общую heatmap: 0 из {count} вариантов…')
        self.set_enabled()

    def load_aggregate_cache(self):
        """Display complete saved results only; this method never starts inference."""
        if not self.image_path or self.aggregate_job:
            return
        key = (self.image_sha, self.steps.value(), self.seed.value(), HEATMAP_VARIANTS)
        if self.aggregate_key == key and self.aggregate_complete:
            return
        results = []
        for index in range(HEATMAP_VARIANTS):
            seed = (self.seed.value()+index) % 1000000
            result = load_prediction(prediction_file(self.output_folder, self.image_sha,
                self.steps.value(), seed), self.image_sha, require_heatmaps=False)
            if (not result or result['parameters']['steps'] != self.steps.value()
                    or result['parameters']['seed'] != seed
                    or (result['image_width'], result['image_height']) !=
                       (self.canvas.image.width(), self.canvas.image.height())):
                break
            results.append(result)
        self.aggregate_key = key
        self.aggregate_complete = len(results) == HEATMAP_VARIANTS
        self.aggregate_results = results if self.aggregate_complete else []
        self.canvas.aggregate_heat = QImage()
        if self.aggregate_complete:
            if self.result is None:
                self.show_result(results[0], self.output_folder)
            points = [p for result in results for p in result['fixations']]
            self.canvas.aggregate_heat = route_heatmap(points, self.canvas.image.width(), self.canvas.image.height())
            self.result_summary.setText(f'Общая карта: {HEATMAP_VARIANTS} вариантов\nТочек прогноза: {len(points)}')
            self.status.setText('Загружена сохранённая общая heatmap.')
        else:
            self.result_summary.setText('Для этих параметров нужна новая общая карта')
            self.status.setText('Нажмите «Построить прогноз», чтобы рассчитать общую heatmap.')
        self.update_object_metrics()
        self.canvas.update()

    def aggregate_updated(self, heat, count):
        if self.result is None and self.aggregate_job and 0 in self.aggregate_job.results:
            self.show_result(self.aggregate_job.results[0], self.output_folder)
        self.canvas.aggregate_heat = heat
        self.canvas.update()
        self.status.setText(f'Общая heatmap · готово {count} из {HEATMAP_VARIANTS} вариантов')
        points = sum(len(r['fixations']) for r in self.aggregate_job.results.values()) if self.aggregate_job else 0
        self.result_summary.setText(f'Общая карта: {count} вариантов\nТочек прогноза: {points}')
        if self.aggregate_job:
            self.aggregate_results = [r for _, r in sorted(self.aggregate_job.results.items())]
            self.update_object_metrics()

    def aggregate_finished(self):
        self.aggregate_complete = True
        job, self.aggregate_job = self.aggregate_job, None
        if job:
            job.deleteLater()
        self.set_enabled()

    def update_object_metrics(self):
        results = self.aggregate_results if self.canvas.heat_mode == 2 else ([self.result] if self.result else [])
        self.object_metrics = object_attention(results) if results else None
        rows = self.object_metrics['rows'] if self.object_metrics else []
        self.object_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            name = object_label(row)
            for col, value in enumerate((name, str(row['count']), f"{row['share']*100:.1f}")):
                item = QTableWidgetItem(value)
                item.setToolTip(name)
                self.object_table.setItem(index, col, item)
        self.metrics_button.setEnabled(self.object_metrics is not None and self.aggregate_job is None)

    def export_metrics(self):
        if not self.object_metrics or self.aggregate_job or self.worker:
            return
        name, _ = QFileDialog.getSaveFileName(self, 'Сохранить внимание по объектам',
            str(Path.home()/f'attention-{self.image_path.stem}.csv'), 'CSV (*.csv)')
        if name:
            if not name.lower().endswith('.csv'):
                name += '.csv'
            try:
                with open(name, 'w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['image', 'data_source', 'variants', 'total_points', 'object_id', 'class_name', 'assignment', 'points', 'share'])
                    for row in self.object_metrics['rows']:
                        writer.writerow([self.image_path.name, 'model_prediction', self.object_metrics['variants'],
                            self.object_metrics['total_points'], row['object_id'], row['class_name'],
                            row['assignment'], row['count'], row['share']])
                self.status.setText(f'Сохранено: {name}')
            except OSError as exc:
                self.status.setText(f'Не удалось сохранить: {exc}')

    def open_batch(self):
        if self.worker or self.aggregate_job:
            return
        self.stop()
        from batch_prediction import BatchPredictionPanel
        dialog = BatchPredictionPanel(self.output_folder, self.steps.value(), self.seed.value(),
            self.opacity.value()/100, self)
        dialog.setStyleSheet(self.styleSheet())
        dialog.exec()
        dialog.deleteLater()

    def aggregate_failed(self, message):
        self.status.setText(message)
        job, self.aggregate_job = self.aggregate_job, None
        if job:
            job.deleteLater()
        self.set_enabled()

    def cancel_prediction(self):
        job, self.aggregate_job = self.aggregate_job, None
        if job:
            job.cancel()
            job.deleteLater()
            self.aggregate_complete = False
        if self.worker:
            self.worker.kill()
            self.worker.waitForFinished(3000)
        self.status.setText('Расчёт остановлен. Готовые варианты сохранены; можно продолжить кнопкой «Построить прогноз».')
        self.set_enabled()

    def set_image(self, path):
        if self.worker or self.aggregate_job:
            return
        self.stop()
        try:
            data = path.read_bytes()
        except OSError as exc:
            self.status.setText(str(exc))
            return
        image = QImage.fromData(data)
        if image.isNull():
            self.status.setText("Не удалось прочитать изображение")
            return
        self.image_path, self.image_sha = path.resolve(), hashlib.sha256(data).hexdigest()
        self.canvas.image, self.canvas.result = image, None
        self.canvas.step = 0
        self.result = None
        self.aggregate_results = []
        self.aggregate_key = None
        self.aggregate_complete = False
        self.canvas.aggregate_heat = QImage()
        self.update_object_metrics()
        self.slider.setRange(0, self.steps.value())
        self.slider.setValue(0)
        self.sequence.setRowCount(0)
        self.result_summary.setText("Изображение готово к просмотру")
        self.step_detail.setText("Выберите точку маршрута")
        self.filename.setText(f"{path.name} · {image.width()} × {image.height()} px")
        cached = prediction_file(self.output_folder, self.image_sha, self.steps.value(), self.seed.value())
        result = load_prediction(cached, self.image_sha, require_heatmaps=False)
        if result:
            self.show_result(result, cached.parent)
        else:
            self.status.setText("Постройте маршрут, чтобы посмотреть прогноз.")
        self.canvas.update()
        self.set_enabled()
        if self.heat_mode.currentIndex() == 2 and self.heatmap.isChecked():
            self.load_aggregate_cache()
            self.set_enabled()

    def parameters_changed(self):
        if not self.image_path or self.worker or self.aggregate_job:
            return
        self.stop()
        self.aggregate_results = []
        self.aggregate_complete = False
        self.aggregate_key = None
        self.canvas.aggregate_heat = QImage()
        file = prediction_file(self.output_folder, self.image_sha, self.steps.value(), self.seed.value())
        result = load_prediction(file, self.image_sha, require_heatmaps=False)
        if result:
            self.show_result(result, file.parent)
        else:
            self.result = self.canvas.result = None
            self.aggregate_results = []
            self.update_object_metrics()
            self.sequence.setRowCount(0)
            self.result_summary.setText("Для этих параметров нужен новый прогноз")
            self.slider.setRange(0, self.steps.value())
            self.slider.setValue(0)
            self.seek(0)
            self.status.setText("Параметры изменены. Нажмите «Построить прогноз».")
        self.canvas.update()
        self.set_enabled()
        if self.heat_mode.currentIndex() == 2 and self.heatmap.isChecked():
            self.load_aggregate_cache()
            self.set_enabled()

    def predict(self):
        if self.worker or self.aggregate_job or not self.image_path:
            return
        self.stop()
        if self.canvas.heat_mode == 2:
            self.ensure_aggregate()
            return
        file = prediction_file(self.output_folder, self.image_sha, self.steps.value(), self.seed.value())
        cached = load_prediction(file, self.image_sha)
        if cached:
            self.show_result(cached, file.parent)
            self.set_enabled()
            return
        executable = BASE / ".venv-detector/bin/python"
        if not executable.exists():
            self.status.setText("Откройте «Настроить прогноз.command» для установки моделей.")
            return
        self.log = ""
        self.worker = QProcess(self)
        self.worker.setWorkingDirectory(str(BASE))
        self.worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.worker.readyReadStandardOutput.connect(self.read_progress)
        self.worker.finished.connect(self.prediction_finished)
        self.worker.errorOccurred.connect(self.process_error)
        self.worker.start(str(executable), ["-u", str(BASE / "scanpath_prediction.py"), str(self.image_path),
            "--steps", str(self.steps.value()), "--seed", str(self.seed.value()), "--output", str(self.output_folder)])
        self.status.setText("Строится прогноз… Первая загрузка весов может занять несколько минут.")
        self.set_enabled()

    def another_variant(self):
        if self.worker or self.aggregate_job:
            return
        self.seed.setValue((self.seed.value()+1) % 1000000)
        if self.result is None or (self.canvas.heat_mode == 2 and not self.aggregate_complete):
            self.predict()

    def read_progress(self):
        if self.worker:
            self.log = (self.log+bytes(self.worker.readAllStandardOutput()).decode("utf-8", errors="replace"))[-16000:]
            lines = [line for line in self.log.splitlines() if line.strip()]
            if lines:
                last = lines[-1]
                if last.startswith("Прогноз: "):
                    self.status.setText("Строим маршрут: " + last.removeprefix("Прогноз: "))
                elif "YOLO" in last:
                    self.status.setText("Находим объекты на изображении…")

    def prediction_finished(self, code, status):
        self.read_progress()
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        file = prediction_file(self.output_folder, self.image_sha, self.steps.value(), self.seed.value())
        result = load_prediction(file, self.image_sha) if code == 0 else None
        if result:
            self.show_result(result, file.parent)
        else:
            self.status.setText("Прогноз не построен. " + self.log[-500:])
        self.set_enabled()

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.status.setText("Не удалось запустить модель. Откройте «Настроить прогноз.command».")
            self.worker.deleteLater()
            self.worker = None
            self.set_enabled()

    def show_result(self, result, folder):
        if (result["image_width"], result["image_height"]) != (self.canvas.image.width(), self.canvas.image.height()):
            self.status.setText("Размер изображения не совпал с прогнозом")
            return
        self.result = self.canvas.result = result
        self.canvas.folder = Path(folder)
        self.slider.setRange(0, len(result["fixations"]))
        self.sequence.blockSignals(True)
        self.sequence.setRowCount(len(result["fixations"]))
        for row, point in enumerate(result["fixations"]):
            assignment = {"assigned": "Одна рамка", "outside": "Вне объектов", "ambiguous": "Пересечение рамок"}[point["assignment"]]
            name = object_label(point)
            for col, text in enumerate([str(point["step"]), f"{point['x']:.0f}, {point['y']:.0f}", name, assignment]):
                item = QTableWidgetItem(text)
                item.setToolTip(f"{name}\n{point['x']:.0f}, {point['y']:.0f} px · {assignment}")
                self.sequence.setItem(row, col, item)
        self.sequence.blockSignals(False)
        self.slider.setValue(self.slider.maximum())
        self.seek(self.slider.value())
        self.result_summary.setText(f"Точек маршрута: {len(result['fixations'])}\nНайдено объектов: {len(result['objects'])}")
        self.status.setText("Выберите точку в списке или включите воспроизведение.")
        self.update_object_metrics()

    def seek(self, value):
        self.canvas.finish_motion()
        self.canvas.step = value
        self.step_label.setText(f"{value} из {self.slider.maximum()}")
        self.sequence.blockSignals(True)
        if self.result and value > 0:
            self.sequence.setCurrentCell(value-1, 2)
            self.sequence.scrollToItem(self.sequence.item(value-1, 2))
            point = self.result["fixations"][value-1]
            self.step_detail.setText(f"Точка {value}: {object_label(point)}\n{point['x']:.0f}, {point['y']:.0f} px")
        else:
            self.sequence.clearSelection()
            self.sequence.setCurrentCell(-1, -1)
            self.step_detail.setText("Начальная позиция — центр изображения")
        self.sequence.blockSignals(False)
        self.canvas.update()

    def sequence_selected(self, row, column, previous_row, previous_column):
        if row >= 0 and self.result:
            self.stop()
            self.slider.setValue(row+1)

    def save_shortcut(self):
        if self.result and not self.worker:
            self.export_image()

    def toggle_layers(self):
        entering_aggregate = (self.heat_mode.currentIndex() == 2 and self.heatmap.isChecked()
            and (self.canvas.heat_mode != 2 or not self.canvas.show_heatmap))
        self.canvas.show_boxes, self.canvas.show_heatmap = self.boxes.isChecked(), self.heatmap.isChecked()
        self.canvas.show_route = self.route.isChecked()
        self.canvas.heat_opacity = self.opacity.value()/100
        self.canvas.heat_mode = self.heat_mode.currentIndex()
        self.heat_controls.setVisible(self.heatmap.isChecked())
        if self.result:
            if self.canvas.heat_mode == 2 and self.aggregate_complete:
                self.result_summary.setText(f"Общая карта: {HEATMAP_VARIANTS} вариантов\nТочек прогноза: {sum(len(r['fixations']) for r in self.aggregate_results)}")
            elif self.canvas.heat_mode != 2:
                self.result_summary.setText(f"Точек маршрута: {len(self.result['fixations'])}\nНайдено объектов: {len(self.result['objects'])}")
        if entering_aggregate:
            self.load_aggregate_cache()
        if self.canvas.heat_mode == 1 and self.result and not self.result['heatmaps']:
            self.status.setText('Для карты отдельного шага нажмите «Построить прогноз». Общая heatmap уже сохранена.')
        self.update_object_metrics()
        self.canvas.update()
        self.set_enabled()

    def play(self):
        if not self.result or self.worker or self.aggregate_job:
            return
        state = self.canvas.motion.state()
        if self.timer.isActive() or state == QAbstractAnimation.State.Running:
            self.stop()
        else:
            if state == QAbstractAnimation.State.Paused:
                self.canvas.motion.resume()
            elif self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self.play_button.setText("Пауза")
            if state != QAbstractAnimation.State.Paused and self.slider.value() < self.slider.maximum():
                self.timer.start()

    def motion_finished(self):
        if self.slider.value() == self.slider.maximum():
            self.stop()
        else:
            self.advance()

    def advance(self):
        self.timer.stop()
        self.slider.setValue(min(self.slider.maximum(), self.slider.value()+1))
        moving = self.canvas.animate_current()
        if self.slider.value() == self.slider.maximum():
            self.timer.stop()
            if not moving:
                self.stop()
        elif not moving:
            self.timer.start()

    def stop(self):
        if hasattr(self, "timer"):
            self.timer.stop()
        if self.canvas.motion.state() == QAbstractAnimation.State.Running:
            self.canvas.motion.pause()
        self.play_button.setText("Смотреть маршрут")

    def export_image(self):
        if not self.result or self.worker:
            return
        self.stop()
        name, _ = QFileDialog.getSaveFileName(self, "Сохранить прогноз",
            str(Path.home()/f"prediction-{self.image_path.stem}.png"), "PNG (*.png)")
        if not name:
            return
        if not name.lower().endswith(".png"):
            name += ".png"
        self.canvas.finish_motion()
        screenshot = self.canvas.grab().toImage()
        output = QImage(screenshot.width(), screenshot.height()+50, QImage.Format.Format_ARGB32)
        output.fill(QColor("#f4f3ef"))
        painter = QPainter(output)
        painter.drawImage(0, 0, screenshot)
        painter.setPen(QColor("#282d2a"))
        painter.drawText(12, screenshot.height()+28, f"Прогноз DeepGaze III · вариант {self.result['parameters']['seed']} · точка {self.slider.value()}")
        painter.end()
        self.status.setText(f"Сохранено: {name}" if output.save(name, "PNG") else "Не удалось сохранить изображение")

    def heatmap_image(self):
        """Export at source resolution, without canvas margins or UI annotations."""
        if not self.result:
            return QImage()
        output = self.canvas.image.convertToFormat(QImage.Format.Format_ARGB32)
        if self.canvas.heat_mode == 2:
            heat = self.canvas.aggregate_heat
        elif self.canvas.heat_mode == 0:
            heat = route_heatmap(self.result['fixations'], output.width(), output.height())
        else:
            step = max(1, self.slider.value())
            if len(self.result['heatmaps']) < step:
                return QImage()
            heat = QImage(str(self.canvas.folder / self.result['heatmaps'][step-1]))
        painter = QPainter(output)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(self.canvas.heat_opacity)
        painter.drawImage(QRectF(0, 0, output.width(), output.height()), heat)
        painter.end()
        return output

    def export_heatmap(self):
        if not self.result or self.worker:
            return
        if self.canvas.heat_mode == 2 and not self.aggregate_complete:
            self.status.setText(f'Дождитесь всех {HEATMAP_VARIANTS} вариантов перед сохранением общей heatmap.')
            return
        if self.canvas.heat_mode == 1 and not self.result['heatmaps']:
            self.status.setText('Сначала постройте карту отдельного шага кнопкой «Построить прогноз».')
            return
        name, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение с heatmap",
            str(Path.home()/f"heatmap-{self.image_path.stem}.png"), "PNG (*.png)")
        if name:
            if not name.lower().endswith('.png'):
                name += '.png'
            self.status.setText(f"Сохранено: {name}" if self.heatmap_image().save(name, 'PNG')
                                else "Не удалось сохранить изображение")

    def closeEvent(self, event):
        if self.worker:
            self.status.setText("Дождитесь завершения прогноза, затем закройте окно.")
            event.ignore()
            return
        self.stop()
        if self.aggregate_job:
            self.aggregate_job.cancel()
            self.aggregate_job = None
        event.accept()

    def reject(self):
        if self.worker:
            self.status.setText("Дождитесь завершения прогноза.")
            return
        self.stop()
        if self.aggregate_job:
            self.aggregate_job.cancel()
            self.aggregate_job = None
        super().reject()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--gallery", action="store_true", help="Open one heatmap pooled from 30 variants")
    args = parser.parse_args()
    qt = QApplication(sys.argv[:1])
    default = BASE.parent.parent / "dataset/PittImageVideoAdsDataset/pitt_ads_images/059738.jpg"
    image = args.image or (default if default.exists() else None)
    window = PredictionPanel(image)
    window.show()
    if args.gallery and image:
        QTimer.singleShot(0, window.open_gallery)
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
