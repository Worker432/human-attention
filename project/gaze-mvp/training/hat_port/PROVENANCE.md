# HAT Mac benchmark port

Upstream: https://github.com/cvlab-stonybrook/HAT
Commit: 557205d3136a63120041c700ee717196a429e89b
HAT code license: MIT, retained in LICENSE. Component copyright notices retained.
Pixel decoder/transformer components originate from Meta Mask2Former / DETR;
deformable attention originates from SenseTime Deformable DETR (Apache 2.0).

This is a local adaptation for performance testing, not an official release,
not a newly proposed architecture, and not a trained gaze predictor.

Copied modules: common/position_encoding.py; transformer_decoder/transformer.py;
transformer_decoder/position_encoding.py; pixel_decoder/msdeformattn.py;
pixel_decoder/ops/modules/ms_deform_attn.py. models.py contains the upstream
HumanAttnTransformer and its attention/MLP helpers, omitting unused model classes.

Changes:
- Imports localized; Detectron2 Conv2d, ShapeSpec, GN and initialization helpers
  replaced by explicit equivalents in compat.py.
- ImageFeatureEncoder uses torchvision ResNet50 with FrozenBatchNorm and stride
  in 3x3, consistent with resnet50.yaml. It exposes res2 through res5 to the
  original six-layer MSD decoder. Classification head is unused.
- Original pretrained Mask2Former weight loading removed for this benchmark.
  All weights are random. No DeepGaze dependency or model predictions.
- CUDA extension import replaced by differentiable grid_sample implementation.
- Split sizes converted to Python integers for compatibility with MPS.
- Training uses spatial log_softmax + NLL as required by project specification;
  the original training-map logits are used directly, not sigmoid/clipped logs.
  The original inference sigmoid path has not yet been adapted or validated.
- Backbone frozen; pixel decoder and HAT history/query modules trainable.

Architecture benchmark: hidden dimension384, 4 attention heads, six prediction
layers, three working-memory encoder layers, one FV task, P1 dorsal source,
512x320 RGB, K=6. Benchmark input tensors are synthetic; targets are grid cells
on the 1/4-resolution map. No stopping/duration loss. End-to-end preprocessing,
real images, checkpoint loading, continuous generation, K=0 and partial-history
inference remain to be integrated and validated.

This benchmark does not establish numerical equivalence of the complete port
with the original pretrained Detectron2 model. It measures this specified port.
