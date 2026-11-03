"""Run in .venv-detector to verify optimized inference against original weights."""
import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires model environment')
class PreparedScanpathTests(unittest.TestCase):
    def test_prepared_output_matches_original_for_different_histories(self):
        import torch
        from scanpath_prediction import BASE, load_model
        from prepared_scanpath import PreparedScanpath
        if not (BASE/'models/torch/hub/checkpoints/deepgaze3.pth').exists():
            self.skipTest('Local model weights are not installed')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = load_model('cpu')
        image = torch.rand(1, 3, 256, 320)*255
        bias = torch.zeros(1, 256, 320)
        calls = []
        hook = model.features.register_forward_hook(lambda *args: calls.append(1))
        try:
            prepared = PreparedScanpath(model, image)
            self.assertEqual(len(calls), 1)
            for xs, ys in (([160., float('nan'), float('nan'), float('nan')],
                            [128., float('nan'), float('nan'), float('nan')]),
                           ([200., 160., 100., 50.], [70., 128., 200., 30.])):
                hx, hy = torch.tensor([xs]), torch.tensor([ys])
                with torch.inference_mode():
                    reference = model(image, bias, hx, hy)
                    before = len(calls)
                    actual = prepared(bias, hx, hy)
                    self.assertEqual(len(calls), before)
                    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        finally:
            hook.remove()


if __name__ == '__main__':
    unittest.main()
