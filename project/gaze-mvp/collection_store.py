"""Durable single-participant collection; no Qt or Beam dependencies."""
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time
import uuid


def image_coordinates(point, viewport, rect, image_size):
    """Unclipped screen and original-image coordinates in a top-left system."""
    sx, sy = point[0] * (viewport[0] - 1), point[1] * (viewport[1] - 1)
    x, y, width, height = rect
    inside = x <= sx < x + width and y <= sy < y + height
    return sx, sy, (sx - x) * image_size[0] / width, (sy - y) * image_size[1] / height, inside


class CollectionStore:
    def __init__(self, folder, image_root, duration=5., participant="p001", seed=42):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder / "collector.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("Сбор уже запущен. Закройте другое окно приложения.")
        self.db = sqlite3.connect(self.folder / "dataset.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started REAL, ended REAL, settings TEXT);
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY, image_id INTEGER UNIQUE NOT NULL, run_id TEXT,
                status TEXT, onset_utc REAL, onset_mono REAL, ended_utc REAL,
                duration REAL, image_sha256 TEXT, geometry TEXT, sample_count INTEGER DEFAULT 0,
                valid_count INTEGER DEFAULT 0, inside_count INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS samples (
                trial_id TEXT, sequence INTEGER, host_utc REAL, host_mono REAL,
                sdk_timestamp REAL, elapsed REAL, confidence TEXT,
                valid INTEGER, inside_image INTEGER, normalized_x REAL, normalized_y REAL,
                screen_x REAL, screen_y REAL, image_x REAL, image_y REAL);
            CREATE INDEX IF NOT EXISTS samples_trial ON samples(trial_id);
        ''')
        root = str(Path(image_root).resolve())
        settings = dict(image_root=root, duration=float(duration), participant=participant,
                        seed=seed, protocol="free_viewing_fixed_v1", inter_image_seconds=1.)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
        if previous:
            if json.loads(previous[0]) != settings:
                self.close()
                raise ValueError("Параметры отличаются от сохранённого сбора. Используйте прежние параметры или другую --output папку.")
        else:
            paths = sorted(str(p.relative_to(root)) for p in Path(root).rglob("*")
                           if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            if not paths:
                self.close()
                raise ValueError("В папке не найдены изображения.")
            random.Random(seed).shuffle(paths)
            self.db.execute("INSERT INTO metadata VALUES ('protocol', ?)", (json.dumps(settings),))
            self.db.executemany("INSERT INTO images(path) VALUES (?)", ((p,) for p in paths))
            self.db.commit()
        self.root = Path(root)
        self.settings = settings
        # A crash must never cause a previously exposed image to be presented again.
        self.db.execute('''UPDATE trials SET status='interrupted_crash',
            sample_count=(SELECT count(*) FROM samples WHERE trial_id=trials.id),
            valid_count=(SELECT coalesce(sum(valid),0) FROM samples WHERE trial_id=trials.id),
            inside_count=(SELECT coalesce(sum(valid * inside_image),0) FROM samples WHERE trial_id=trials.id)
            WHERE status='presenting' ''')
        self.db.execute("UPDATE runs SET ended=? WHERE ended IS NULL", (time.time(),))
        self.db.commit()

    def begin_run(self, settings):
        self.run_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?, ?, NULL, ?)",
                            (self.run_id, time.time(), json.dumps(settings, ensure_ascii=False)))
        return self.run_id

    def progress(self):
        total = self.db.execute("SELECT count(*) FROM images").fetchone()[0]
        seen = self.db.execute("SELECT count(*) FROM trials").fetchone()[0]
        complete = self.db.execute("SELECT count(*) FROM trials WHERE status='completed'").fetchone()[0]
        return total, seen, complete

    def next_image(self):
        return self.db.execute('''SELECT images.* FROM images LEFT JOIN trials
            ON images.id=trials.image_id WHERE trials.id IS NULL ORDER BY images.id LIMIT 1''').fetchone()

    def begin_trial(self, image_id, sha, geometry, stamp, mono):
        trial = uuid.uuid4().hex
        with self.db:
            self.db.execute('''INSERT INTO trials
                (id,image_id,run_id,status,onset_utc,onset_mono,image_sha256,geometry)
                VALUES (?,?,?,'presenting',?,?,?,?)''',
                (trial, image_id, self.run_id, stamp, mono, sha, json.dumps(geometry)))
        return trial

    def add_samples(self, rows):
        if rows:
            with self.db:
                self.db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def finish_trial(self, trial, status, stamp, duration):
        with self.db:
            self.db.execute('''UPDATE trials SET status=?,ended_utc=?,duration=?,
                sample_count=(SELECT count(*) FROM samples WHERE trial_id=?),
                valid_count=(SELECT coalesce(sum(valid),0) FROM samples WHERE trial_id=?),
                inside_count=(SELECT coalesce(sum(valid * inside_image),0) FROM samples WHERE trial_id=?)
                WHERE id=?''', (status, stamp, duration, trial, trial, trial, trial))

    def close(self):
        if getattr(self, "run_id", None):
            with self.db:
                self.db.execute("UPDATE runs SET ended=? WHERE id=?", (time.time(), self.run_id))
        self.db.close()
        self.lock.close()


def export_dataset(database, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    try:
        db.execute("BEGIN")
        for table in ("metadata", "images", "runs", "trials", "samples"):
            cursor = db.execute(f"SELECT * FROM {table}")
            with (destination / f"{table}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([column[0] for column in cursor.description])
                writer.writerows(cursor)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export a consistent dataset snapshot to CSV")
    parser.add_argument("database")
    parser.add_argument("destination")
    args = parser.parse_args()
    export_dataset(args.database, args.destination)
