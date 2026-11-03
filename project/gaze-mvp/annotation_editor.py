"""Small native bounding-box editor. Gaze is deliberately hidden while annotating."""
import copy
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)
from gaze_analysis import class_catalog, load_annotation, save_annotation


class BoxCanvas(QWidget):
    changed = Signal()
    selected = Signal(int)

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.boxes = []
        self.selection = -1
        self.mode = "select"
        self.label = "person"
        self.drag = None
        self.undo_stack = []
        self.setMinimumSize(400, 350)

    def image_rect(self):
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h), scale

    def to_image(self, pos, clip=False):
        rect, scale = self.image_rect()
        if not clip and not rect.contains(pos):
            return None
        return QPointF(min(self.image.width(), max(0., (pos.x()-rect.x())/scale)),
                       min(self.image.height(), max(0., (pos.y()-rect.y())/scale)))

    def checkpoint(self):
        self.undo_stack.append(copy.deepcopy(self.boxes))
        self.undo_stack = self.undo_stack[-50:]

    def undo(self):
        if self.undo_stack:
            self.boxes = self.undo_stack.pop()
            self.selection = -1
            self.changed.emit()
            self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.to_image(event.position())
        if point is None:
            return
        if self.mode == "draw":
            self.drag = dict(kind="draw", start=point, end=point)
            self.update()
            return
        if 0 <= self.selection < len(self.boxes):
            x1, y1, x2, y2 = self.boxes[self.selection]["xyxy"]
            _, scale = self.image_rect()
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for index, (x, y) in enumerate(corners):
                if abs(point.x()-x)*scale <= 10 and abs(point.y()-y)*scale <= 10:
                    self.checkpoint()
                    self.drag = dict(kind="resize", corner=index, original=list(self.boxes[self.selection]["xyxy"]))
                    return
        hits = [(i, b) for i, b in enumerate(self.boxes) if b["xyxy"][0] <= point.x() <= b["xyxy"][2]
                and b["xyxy"][1] <= point.y() <= b["xyxy"][3]]
        if hits:
            i, box = min(hits, key=lambda pair: (pair[1]["xyxy"][2]-pair[1]["xyxy"][0])*(pair[1]["xyxy"][3]-pair[1]["xyxy"][1]))
            self.selection = i
            self.selected.emit(i)
            self.checkpoint()
            self.drag = dict(kind="move", start=point, original=list(box["xyxy"]))
        else:
            self.selection = -1
            self.selected.emit(-1)
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag is None:
            return
        point = self.to_image(event.position(), clip=True)
        if self.drag["kind"] == "draw":
            self.drag["end"] = point
        elif self.drag["kind"] == "move":
            x1, y1, x2, y2 = self.drag["original"]
            dx = min(self.image.width()-x2, max(-x1, point.x()-self.drag["start"].x()))
            dy = min(self.image.height()-y2, max(-y1, point.y()-self.drag["start"].y()))
            self.boxes[self.selection]["xyxy"] = [x1+dx, y1+dy, x2+dx, y2+dy]
        else:
            coords = list(self.drag["original"])
            corner = self.drag["corner"]
            coords[0 if corner in (0, 3) else 2] = point.x()
            coords[1 if corner in (0, 1) else 3] = point.y()
            self.boxes[self.selection]["xyxy"] = [min(coords[0], coords[2]), min(coords[1], coords[3]),
                                                   max(coords[0], coords[2]), max(coords[1], coords[3])]
        self.update()

    def mouseReleaseEvent(self, event):
        if self.drag is None:
            return
        if self.drag["kind"] == "draw":
            a, b = self.drag["start"], self.drag["end"]
            if abs(a.x()-b.x()) >= 2 and abs(a.y()-b.y()) >= 2 and self.label.strip():
                self.checkpoint()
                self.boxes.append(dict(object_id=max((box["object_id"] for box in self.boxes), default=0)+1,
                    class_name=self.label.strip(), class_id=0, confidence=None,
                    xyxy=[min(a.x(), b.x()), min(a.y(), b.y()), max(a.x(), b.x()), max(a.y(), b.y())]))
                self.selection = len(self.boxes)-1
        self.drag = None
        self.changed.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15181d"))
        rect, scale = self.image_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(rect, self.image)
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box["xyxy"]
            bounds = QRectF(rect.x()+x1*scale, rect.y()+y1*scale, (x2-x1)*scale, (y2-y1)*scale)
            painter.setPen(QPen(QColor("#62e6a7" if i == self.selection else "#ffb56b"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(bounds)
            if i == self.selection:
                painter.setBrush(QColor("#62e6a7"))
                for point in (bounds.topLeft(), bounds.topRight(), bounds.bottomLeft(), bounds.bottomRight()):
                    painter.drawRect(QRectF(point.x()-4, point.y()-4, 8, 8))
            label = f"#{box['object_id']} {box['class_name']}"
            painter.drawText(bounds.topLeft()+QPointF(2, 16), label)
        if self.drag and self.drag["kind"] == "draw":
            a, b = self.drag["start"], self.drag["end"]
            painter.setPen(QPen(QColor("#62e6a7"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(QPointF(rect.x()+a.x()*scale, rect.y()+a.y()*scale),
                                    QPointF(rect.x()+b.x()*scale, rect.y()+b.y()*scale)).normalized())
        painter.end()


class AnnotationEditor(QDialog):
    def __init__(self, database, trial, image, automatic=None, parent=None):
        super().__init__(parent)
        self.database, self.trial = database, trial
        self.saved = False
        self.setWindowTitle(f"Разметка объектов · {trial['path']}")
        self.resize(1100, 800)
        self.setStyleSheet("QDialog {background:#20242b;} QLabel,QCheckBox {color:#edf0f5;} QPushButton {padding:8px;}")
        outer = QVBoxLayout(self)
        help_text = QLabel("Разметьте выбранные типы объектов целиком, включая те, на которые не смотрели.\n"
                           "Траектория скрыта. Вложенные/пересекающиеся рамки могут дать неоднозначное сопоставление.")
        help_text.setWordWrap(True)
        outer.addWidget(help_text)
        toolbar = QHBoxLayout()
        self.draw_button = QPushButton("Новая рамка: протянуть мышью")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self.set_draw)
        toolbar.addWidget(self.draw_button)
        self.classes = QComboBox()
        self.classes.setEditable(True)
        self.classes.addItems(sorted(set(class_catalog(database)) | {"face", "text", "logo"}))
        toolbar.addWidget(self.classes, 1)
        apply_class = QPushButton("Применить класс")
        apply_class.clicked.connect(self.apply_class)
        toolbar.addWidget(apply_class)
        delete = QPushButton("Удалить рамку")
        delete.clicked.connect(self.delete_box)
        toolbar.addWidget(delete)
        undo = QPushButton("Отменить правку")
        undo.clicked.connect(self.undo)
        toolbar.addWidget(undo)
        outer.addLayout(toolbar)
        body = QHBoxLayout()
        self.canvas = BoxCanvas(image)
        annotation = load_annotation(database, trial)
        self.canvas.boxes = copy.deepcopy(annotation["objects"] if annotation else (automatic or {}).get("objects", []))
        self.canvas.selected.connect(self.select_box)
        self.canvas.changed.connect(self.refresh_list)
        self.classes.currentTextChanged.connect(self.set_label)
        self.canvas.label = self.classes.currentText()
        body.addWidget(self.canvas, 1)
        self.list = QListWidget()
        self.list.setMaximumWidth(220)
        self.list.currentRowChanged.connect(self.select_box)
        body.addWidget(self.list)
        outer.addLayout(body, 1)
        self.reviewed = QCheckBox("Проверены все выбранные типы объектов на этом изображении")
        outer.addWidget(self.reviewed)
        self.notice = QLabel("Выбор: щёлкните по рамке или строке. Перенос: тяните рамку. Размер: тяните её угол.")
        self.notice.setWordWrap(True)
        outer.addWidget(self.notice)
        bottom = QHBoxLayout()
        save = QPushButton("Сохранить проверенную разметку")
        save.clicked.connect(self.save)
        bottom.addWidget(save)
        cancel = QPushButton("Закрыть без сохранения")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(cancel)
        outer.addLayout(bottom)
        self.refresh_list()

    def set_draw(self, enabled):
        self.canvas.mode = "draw" if enabled else "select"
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_label(self, text):
        self.canvas.label = text.strip()

    def select_box(self, index):
        if not 0 <= index < len(self.canvas.boxes):
            self.canvas.selection = -1
        else:
            self.canvas.selection = index
            self.classes.setCurrentText(self.canvas.boxes[index]["class_name"])
            self.list.blockSignals(True)
            self.list.setCurrentRow(index)
            self.list.blockSignals(False)
        self.canvas.update()

    def refresh_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for box in self.canvas.boxes:
            self.list.addItem(f"#{box['object_id']} {box['class_name']}")
        self.list.setCurrentRow(self.canvas.selection)
        self.list.blockSignals(False)
        self.canvas.update()

    def apply_class(self):
        i = self.canvas.selection
        if 0 <= i < len(self.canvas.boxes) and self.classes.currentText().strip():
            self.canvas.checkpoint()
            self.canvas.boxes[i]["class_name"] = self.classes.currentText().strip()
            self.refresh_list()

    def delete_box(self):
        i = self.canvas.selection
        if 0 <= i < len(self.canvas.boxes):
            self.canvas.checkpoint()
            self.canvas.boxes.pop(i)
            self.canvas.selection = -1
            self.refresh_list()

    def undo(self):
        self.canvas.undo()

    def save(self):
        try:
            save_annotation(self.database, self.trial, self.canvas.boxes, reviewed=self.reviewed.isChecked())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.notice.setText(str(exc))
            return
        self.saved = True
        self.accept()
