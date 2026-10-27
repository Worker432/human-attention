"""Camera acquisition and Vision run away from the GUI thread."""
import threading
import time
import logging
import cv2
import numpy as np
import objc
import Vision
from Foundation import NSData
from PySide6.QtCore import QThread
from core import eye_features


def points(region):
    if region is None or region.pointCount() == 0:
        raise ValueError("Missing eye/pupil landmarks")
    buffer = region.normalizedPoints()
    return np.array([(buffer[i].x, buffer[i].y) for i in range(region.pointCount())])


def eye_preview(frame, face, left_eye, left_pupil, right_eye, right_pupil):
    """Convert Vision's face-relative bottom-left coordinates to camera pixels."""
    box = face.boundingBox()
    h, w = frame.shape[:2]
    image = frame.copy()
    all_pixels = []
    for region, color in ((left_eye, (0, 255, 0)), (right_eye, (0, 255, 0)),
                          (left_pupil, (0, 0, 255)), (right_pupil, (0, 0, 255))):
        for x, y in region:
            px = int((box.origin.x + x * box.size.width) * w)
            py = int((1 - box.origin.y - y * box.size.height) * h)
            all_pixels.append((px, py))
            cv2.circle(image, (px, py), 2, color, -1)
    low = np.min(all_pixels, axis=0) - 25
    high = np.max(all_pixels, axis=0) + 25
    x0, y0 = np.maximum(low, 0)
    x1, y1 = np.minimum(high, [w, h])
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def extract(frame, preview_callback=None):
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode camera frame")
    data = NSData.dataWithBytes_length_(encoded.tobytes(), encoded.size)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    # Locate a single face before requesting its eye landmarks.
    detector = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    detector.setRevision_(Vision.VNDetectFaceRectanglesRequestRevision3)
    ok, error = handler.performRequests_error_([detector], None)
    if not ok:
        raise RuntimeError(f"Vision: {error}")
    faces = detector.results() or []
    if len(faces) != 1:
        return None, "Нужно ровно одно лицо в кадре"
    detected_face = faces[0]
    if detected_face.confidence() < 0.7:
        return None, "Низкая уверенность обнаружения лица"
    request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    request.setRevision_(Vision.VNDetectFaceLandmarksRequestRevision3)
    request.setInputFaceObservations_([detected_face])
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision landmarks: {error}")
    results = request.results() or []
    if len(results) != 1:
        return None, "Нет результата landmarks"
    face = results[0]
    landmarks = face.landmarks()
    if landmarks is None:
        return None, "Нет landmarks"
    try:
        left_eye, left_pupil = points(landmarks.leftEye()), points(landmarks.leftPupil())
        right_eye, right_pupil = points(landmarks.rightEye()), points(landmarks.rightPupil())
        if preview_callback is not None:
            preview_callback(eye_preview(frame, face, left_eye, left_pupil, right_eye, right_pupil))
        features = eye_features(left_eye, left_pupil)
        features += eye_features(right_eye, right_pupil)
    except ValueError:
        return None, "Не видны зрачки / глаза закрыты"
    features = np.asarray(features)
    if not np.isfinite(features).all():
        return None, "Некорректные landmarks"
    return features, "Лицо и зрачки видны"


class Camera(QThread):
    def __init__(self, index=0):
        super().__init__()
        self.index = index
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.preview = None

    def save_preview(self, frame):
        with self.lock:
            self.preview = None if frame is None else (frame, time.monotonic())

    def debug_snapshot(self):
        with self.lock:
            return self.preview

    def publish(self, features, status, stamp, monotonic):
        with self.lock:
            self.sequence += 1
            self.latest = (self.sequence, features, status, stamp, monotonic)

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        cap = None
        try:
            logging.info("Открытие камеры AVFoundation, индекс %s", self.index)
            cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                raise RuntimeError("Камера недоступна: проверьте разрешение Camera и --camera")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            logging.info("Камера открыта; ожидание первого кадра")
            first = True
            while not self.isInterruptionRequested():
                started = time.monotonic()
                ok, frame = cap.read()
                stamp, mono = time.time(), time.monotonic()
                if not ok:
                    raise RuntimeError("Не удалось прочитать кадр камеры")
                if first:
                    logging.info("Первый кадр получен: %s; вызов Vision", frame.shape)
                with objc.autorelease_pool():
                    features, status = extract(frame, self.save_preview)
                if first:
                    logging.info("Первый вызов Vision завершён: %s", status)
                    first = False
                self.publish(features, status, stamp, mono)
                self.msleep(max(1, int((1 / 20 - (time.monotonic() - started)) * 1000)))
        except Exception as exc:
            logging.exception("Ошибка камеры/Vision")
            self.publish(None, str(exc), time.time(), time.monotonic())
        finally:
            if cap is not None:
                cap.release()
