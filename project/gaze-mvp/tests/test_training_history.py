import unittest
from training.history import FixationHistory


class HistoryTests(unittest.TestCase):
    def test_only_last_points_survive(self):
        h = FixationHistory(6)
        for n in range(10000):
            h.append(n, n + 1)
        self.assertEqual(h.window(), ([(float(n), float(n+1)) for n in range(9994, 10000)], [False]*6))
        self.assertEqual(len(h._points), 6)

    def test_padding_and_reset(self):
        h = FixationHistory(2)
        h.append(3, 4)
        self.assertEqual(h.window(), ([(3., 4.), (0., 0.)], [False, True]))
        h.reset()
        self.assertEqual(h.window(), ([(0., 0.)]*2, [True]*2))

    def test_zero_history(self):
        h = FixationHistory(0)
        h.append(1, 2)
        self.assertEqual(h.window(), ([], []))

    def test_invalid_inputs(self):
        for k in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                FixationHistory(k)
        with self.assertRaises(ValueError):
            FixationHistory().append(float('nan'), 1)
