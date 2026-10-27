"""Pure numerical code; no camera or macOS imports."""
import math
import numpy as np

FEATURE_NAMES = ["left_pupil_x", "left_pupil_y", "left_eye_aspect",
                 "right_pupil_x", "right_pupil_y", "right_eye_aspect"]


def eye_features(eye, pupil):
    eye, pupil = np.asarray(eye, float), np.asarray(pupil, float)
    low, high = eye.min(axis=0), eye.max(axis=0)
    width, height = high - low
    if width < 1e-5 or height / width < 0.10:
        raise ValueError("Eye closed or degenerate")
    relative = (pupil.mean(axis=0) - (low + high) / 2) / width
    return [*relative, height / width]


class Ridge:
    def __init__(self, alpha=30.0, degree=1):
        self.alpha = alpha
        self.degree = degree

    def expand(self, x):
        x = np.asarray(x, float)
        if self.degree == 1:
            return x
        # Quadratic terms only for pupil positions: avoid a large head-pose polynomial.
        indices = [0, 1, 3, 4]
        terms = [x[..., i] * x[..., j] for n, i in enumerate(indices) for j in indices[n:]]
        return np.concatenate([x, np.stack(terms, axis=-1)], axis=-1)

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or y.shape != (len(x), 2) or len(x) < 9:
            raise ValueError("Invalid calibration shape")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite calibration")
        x = self.expand(x)
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-4)
        self.center = y.mean(axis=0)
        z = (x - self.mean) / self.scale
        self.weights = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]),
                                       z.T @ (y - self.center))
        return self

    def predict(self, x):
        return ((self.expand(x) - self.mean) / self.scale) @ self.weights + self.center


class Smoother:
    def __init__(self, tau=0.15):
        self.tau = tau
        self.reset()

    def reset(self):
        self.value = None
        self.last = None

    def update(self, point, now):
        point = np.asarray(point, float)
        if self.value is None or now - self.last > 0.5:
            self.value = point.copy()
        else:
            alpha = 1 - math.exp(-max(0, now - self.last) / self.tau)
            self.value += alpha * (point - self.value)
        self.last = now
        return self.value.copy()


def to_pixels(point, width, height):
    return np.clip(point, 0, 1) * np.array([width - 1, height - 1])


def select_model(train_x, train_y, tune_x, tune_y, pixels):
    """Tune on separate targets. Final evaluation must use a third dataset."""
    scores, models = [], []
    for degree in (1, 2):
        for alpha in (0.1, 1., 10., 30., 100.):
            model = Ridge(alpha=alpha, degree=degree).fit(train_x, train_y)
            residual = (model.predict(tune_x) - np.asarray(tune_y)) * (np.asarray(pixels)-1)
            error = float(np.mean(np.linalg.norm(residual, axis=1)))
            scores.append(dict(degree=degree, alpha=alpha, mean_error_px=error))
            models.append(model)
    best = int(np.argmin([r['mean_error_px'] for r in scores]))
    return models[best], scores
