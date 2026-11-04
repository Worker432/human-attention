"""Portable differentiable multi-scale deformable attention core.

Independent expression of the grid-sampling operation described by Deformable
DETR and used by HAT. Uses PyTorch autograd instead of the CUDA-only extension.
No pretrained weights or DeepGaze predictions are used. This core alone is not HAT.
Reference: https://github.com/cvlab-stonybrook/HAT/tree/main/hat/pixel_decoder/ops
"""
import torch
from torch.nn import functional as F


def deformable_attention(value, spatial_shapes, locations, weights):
    """value [B,S,H,C], locations [B,Q,H,L,P,2], weights [B,Q,H,L,P]."""
    shapes = [(int(h), int(w)) for h, w in spatial_shapes]
    batch, pixels, heads, channels = value.shape
    b, queries, nh, levels, samples, xy = locations.shape
    if b != batch or nh != heads or xy != 2 or levels != len(shapes):
        raise ValueError('incompatible attention dimensions')
    if any(h <= 0 or w <= 0 for h, w in shapes):
        raise ValueError('spatial shapes must be positive')
    if sum(h * w for h, w in shapes) != pixels:
        raise ValueError('spatial shapes do not match flattened features')
    if tuple(weights.shape) != tuple(locations.shape[:-1]):
        raise ValueError('weights must match sampling dimensions')
    pieces = value.split([h * w for h, w in shapes], dim=1)
    result = value.new_zeros((batch * heads, channels, queries))
    for level, ((h, w), features) in enumerate(zip(shapes, pieces)):
        features = features.permute(0, 2, 3, 1).reshape(batch * heads, channels, h, w)
        grid = locations[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = grid.reshape(batch * heads, queries, samples, 2) * 2 - 1
        sampled = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        mass = weights[:, :, :, level].permute(0, 2, 1, 3).reshape(batch * heads, 1, queries, samples)
        result = result + (sampled * mass).sum(dim=-1)
    return result.reshape(batch, heads * channels, queries).transpose(1, 2).contiguous()
