"""Beam SDK acquisition in a worker thread."""
import logging
import math
import threading
import time
from collections import deque

from PySide6.QtCore import QThread


CONFIDENCE_NAMES = {
    0: "LOST_TRACKING",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
}

RECEPTION_STATUS_NAMES = {
    0: "NOT_RECEIVING_TRACKING_DATA",
    1: "RECEIVING_TRACKING_DATA",
    2: "ATTEMPTING_TRACKING_AUTO_START",
}


def enum_value(value):
    """Accept both pybind enum objects and plain integers returned by some builds."""
    return int(getattr(value, "value", value))


def enum_name(value, names):
    name = getattr(value, "name", None)
    return str(name) if name is not None else names.get(enum_value(value), str(enum_value(value)))


class BeamSource(QThread):
    """Publish the latest normalized gaze point without blocking the Qt thread."""

    def __init__(self, viewport, min_confidence=2, capture_queue=False):
        super().__init__()
        self.viewport = tuple(int(value) for value in viewport)
        self.min_confidence = int(min_confidence)
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.capture_queue = capture_queue
        self.pending = deque(maxlen=10000)
        self.dropped_packets = 0
        self.sdk_timestamp = None

    def publish(self, point, confidence, status, stamp=None, mono=None):
        with self.lock:
            self.sequence += 1
            self.latest = (
                self.sequence,
                point,
                confidence,
                status,
                time.time() if stamp is None else stamp,
                time.monotonic() if mono is None else mono,
            )
            if self.capture_queue:
                if len(self.pending) == self.pending.maxlen:
                    self.dropped_packets += 1
                self.pending.append((*self.latest, self.sdk_timestamp))

    def drain(self):
        """Return all queued SDK updates, rather than only the last UI sample."""
        with self.lock:
            packets = list(self.pending)
            self.pending.clear()
            return packets

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        try:
            from eyeware import beam_eye_tracker as beam

            x, y, width, height = self.viewport
            geometry = beam.ViewportGeometry()
            geometry.point_00 = beam.Point(x, y)
            geometry.point_11 = beam.Point(x + width - 1, y + height - 1)
            api = beam.API("Diplom Gaze Tracker", geometry)
            version = api.get_version()
            logging.info(
                "Beam SDK %s.%s.%s.%s; viewport=(%s, %s, %s, %s)",
                version.major, version.minor, version.patch, version.build,
                x, y, width, height,
            )
            api.attempt_starting_the_beam_eye_tracker()
            last_timestamp = beam.NULL_DATA_TIMESTAMP()

            while not self.isInterruptionRequested():
                if not api.wait_for_new_tracking_state_set(last_timestamp, 250):
                    reception = api.get_tracking_data_reception_status()
                    reception_name = enum_name(reception, RECEPTION_STATUS_NAMES)
                    self.publish(None, "LOST_TRACKING", f"Beam: {reception_name}")
                    continue

                user = api.get_latest_tracking_state_set().user_state()
                # The SDK cursor must advance; otherwise the same state can be
                # returned repeatedly and look like fresh gaze samples.
                timestamp = user.timestamp_in_seconds
                if self.sdk_timestamp == float(timestamp.value):
                    self.msleep(2)
                    continue
                last_timestamp = timestamp
                self.sdk_timestamp = float(timestamp.value)
                gaze = user.viewport_gaze
                confidence = gaze.confidence
                confidence_name = enum_name(confidence, CONFIDENCE_NAMES)
                confidence_value = enum_value(confidence)
                stamp, mono = time.time(), time.monotonic()

                if confidence_value < self.min_confidence:
                    self.publish(
                        None,
                        confidence_name,
                        f"Beam: низкая уверенность ({confidence_name})",
                        stamp,
                        mono,
                    )
                    continue

                point = gaze.normalized_point_of_regard
                normalized = (float(point.x), float(point.y))
                if not all(math.isfinite(value) for value in normalized):
                    self.publish(None, confidence_name, "Beam вернул неверные координаты", stamp, mono)
                    continue
                self.publish(normalized, confidence_name, f"Beam: {confidence_name}", stamp, mono)
        except Exception as exc:
            logging.exception("Ошибка Beam SDK")
            self.publish(None, "ERROR", f"Beam SDK: {exc}")
