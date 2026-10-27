import unittest
import numpy as np
from core import Ridge, Smoother, eye_features, to_pixels, select_model


class CoreTests(unittest.TestCase):
    def test_quadratic_selection_on_separate_data(self):
        rng = np.random.default_rng(65)
        def dataset(n):
            x = rng.uniform(-1, 1, size=(n, 6))
            y = np.column_stack([.5 + .2*x[:, 0] + .25*x[:, 1]**2,
                                 .5 + .2*x[:, 3]*x[:, 4]])
            return x, y
        x, y = dataset(500)
        tx, ty = dataset(100)
        vx, vy = dataset(100)
        model, scores = select_model(x, y, tx, ty, (3024, 1964))
        self.assertEqual(model.degree, 2)
        self.assertEqual(len(scores), 10)
        np.testing.assert_allclose(model.predict(vx), vy, atol=.002)
        np.testing.assert_allclose(model.predict(vx[0]), model.predict(vx)[0])

    def test_unseen_linear_mapping_and_constant_feature(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(400, 5))
        x[:, -1] = 1
        weights = rng.normal(size=(5, 2))
        model = Ridge(alpha=0.01).fit(x, x @ weights + 0.3)
        held_out = rng.normal(size=(30, 5))
        held_out[:, -1] = 1
        np.testing.assert_allclose(model.predict(held_out), held_out @ weights + .3, atol=.001)

    def test_missing_gap_resets_smoothing(self):
        s = Smoother(.15)
        s.update([0, 0], 1)
        self.assertTrue(0 < s.update([1, 1], 1.1)[0] < 1)
        np.testing.assert_array_equal(s.update([.2, .3], 2), [.2, .3])

    def test_smoothing_independent_of_frame_rate(self):
        a, b = Smoother(), Smoother()
        a.update([0, 0], 0)
        b.update([0, 0], 0)
        for t in [.1, .2, .3]:
            av = a.update([1, 1], t)
        bv = b.update([1, 1], .3)
        np.testing.assert_allclose(av, bv)

    def test_retina_bounds(self):
        np.testing.assert_array_equal(to_pixels([-1, 2], 3024, 1964), [0, 1963])
        np.testing.assert_array_equal(to_pixels([1, 0], 3024, 1964), [3023, 0])

    def test_eye_normalization(self):
        eye = np.array([[0, .1], [.5, 0], [1, .1], [.5, .2]])
        np.testing.assert_allclose(eye_features(eye, [[.5, .1]]), [0, 0, .2])
        with self.assertRaises(ValueError):
            eye_features([[0, 0], [1, 0]], [[.5, 0]])


if __name__ == "__main__":
    unittest.main()
