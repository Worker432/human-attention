"""Cache the image-only branch of the pinned DeepGaze III mixture in eval mode."""
import math


class PreparedScanpath:
    def __init__(self, model, image):
        import torch
        import torch.nn.functional as F
        if model.training:
            raise ValueError('Prepared scanpaths require an eval model')
        self.model = model
        self.size = image.shape[2:]
        self.device = image.device
        self.readout_shape = [math.ceil(n/model.downsample/model.readout_factor) for n in self.size]
        with torch.inference_mode():
            features = model.features(F.interpolate(image, scale_factor=1/model.downsample,
                                                     recompute_scale_factor=False))
            readout = torch.cat([F.interpolate(f, self.readout_shape) for f in features], dim=1)
            self.saliency = [network(readout) for network in model.saliency_networks]

    def __call__(self, centerbias, x_hist, y_hist):
        import torch
        import torch.nn.functional as F
        from deepgaze_pytorch.modules import encode_scanpath_features
        with torch.inference_mode():
            features = encode_scanpath_features(x_hist, y_hist, size=self.size, device=self.device)
            features = F.interpolate(features, self.readout_shape)
            predictions = []
            for static, scanpath, selection, finalizer in zip(self.saliency,
                    self.model.scanpath_networks, self.model.fixation_selection_networks,
                    self.model.finalizers):
                history = scanpath(features) if scanpath is not None else None
                density = finalizer(selection((static, history)), centerbias)
                predictions.append(density[:, None, :, :])
            return (torch.cat(predictions, dim=1)-math.log(len(predictions))).logsumexp(dim=1, keepdim=True)
