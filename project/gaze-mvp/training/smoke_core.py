"""Checks the portable attention core, not a full HAT training step."""
import argparse
import json
import time
import torch
from training.deformable_attention import deformable_attention


def run(device):
    torch.manual_seed(42)
    shapes = [(3, 4), (2, 2)]
    v = torch.randn(1, 16, 2, 3, dtype=torch.float64, requires_grad=True)
    loc = (torch.rand(1, 3, 2, 2, 2, 2, dtype=torch.float64) * .7 + .13).requires_grad_()
    w = torch.rand(1, 3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    checked = torch.autograd.gradcheck(lambda a,b,c: deformable_attention(a, shapes,b,c), (v,loc,w))
    cpu = deformable_attention(v,shapes,loc,w).detach().float()
    vd, ld, wd = [t.detach().float().to(device).requires_grad_() for t in (v,loc,w)]
    start = time.perf_counter()
    out = deformable_attention(vd,shapes,ld,wd)
    out.square().mean().backward()
    if device == 'mps': torch.mps.synchronize()
    seconds = time.perf_counter()-start
    error = (out.detach().cpu()-cpu).abs().max().item()
    assert error < 1e-5, error
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (vd,ld,wd))
    constant = deformable_attention(torch.ones(1,4,1,1), [(2,2)], torch.full((1,1,1,1,1,2),.5), torch.ones(1,1,1,1,1))
    assert constant.item() == 1.
    return dict(scope='attention core only; not full HAT', device=device, torch=torch.__version__, cpu_double_gradcheck=checked, max_error_float32=error, forward_backward_seconds=seconds)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--device', choices=['cpu','mps'],default='cpu');p.add_argument('--output')
    args=p.parse_args();result=run(args.device);text=json.dumps(result,indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text+'\n')
    print(text)
