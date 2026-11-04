"""Synthetic full-graph training-step timing for the documented HAT Mac port."""
import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from training.hat_port.models import HumanAttnTransformer


def run(args):
    torch.manual_seed(42);torch.set_num_threads(args.threads)
    pa=SimpleNamespace(im_h=args.height,im_w=args.width,backbone_config='',pixel_decoder='MSD',special_symbols=['<pad>'],max_traj_length=max(args.history,1)+1,pad_idx=0,eos_idx=-1)
    model=HumanAttnTransformer(pa,num_decoder_layers=6,hidden_dim=384,nhead=4,ntask=1,tgt_vocab_size=1,num_output_layers=3,train_encoder=False,train_pixel_decoder=True,dropout=0.,dim_feedforward=512,parallel_arch=True,dorsal_source=['P1'],num_encoder_layers=3,separate_fix_arch=True,combine_pos_emb=True)
    model.to(args.device);model.train();model.encoder.backbone.eval()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    image=torch.rand(args.batch,3,args.height,args.width,device=args.device)
    seq=torch.randint(1,(args.height//4)*(args.width//4)+1,(args.batch,args.history),device=args.device)
    mask=torch.zeros(args.batch,args.history,device=args.device,dtype=torch.bool)
    target=torch.randint((args.height//4)*(args.width//4),(args.batch,),device=args.device)
    tasks=torch.zeros(args.batch,device=args.device,dtype=torch.long)
    times=[];losses=[];alloc=[]
    def sync():
        if args.device=='mps':torch.mps.synchronize()
    for step in range(args.warmup+args.steps):
        target=torch.randint((args.height//4)*(args.width//4),(args.batch,),device=args.device)
        sync();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        pred=model(image,seq,mask,seq,tasks)['pred_fixation_map'].reshape(args.batch,-1)
        logp=torch.log_softmax(pred,dim=1)
        loss=-logp[torch.arange(args.batch,device=args.device),target].mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();optimizer.step();sync();elapsed=time.perf_counter()-start
        grad=model.fixation_embed.layers[-1].weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().max().item()==0:raise RuntimeError("missing, zero or nonfinite prediction gradients")
        if step>=args.warmup:times.append(elapsed);losses.append(loss.item())
        if args.device=='mps':alloc.append(torch.mps.driver_allocated_memory())
        print(json.dumps(dict(step=step,warmup=step<args.warmup,seconds=elapsed,nll=loss.item())),flush=True)
    report=dict(scope='full HAT Mac port synthetic forward/backward/AdamW; not real-data training',upstream_commit='557205d3136a63120041c700ee717196a429e89b',device=args.device,platform=platform.platform(),torch=torch.__version__,resolution=[args.width,args.height],batch=args.batch,history=args.history,threads=args.threads,weights='random; no DeepGaze',frozen_backbone=True,train_pixel_decoder=True,warmup=args.warmup,step_seconds=times,median_seconds=statistics.median(times),losses=losses,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mps_max_observed_driver_bytes=max(alloc,default=None))
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--height',type=int,default=320);p.add_argument('--width',type=int,default=512);p.add_argument('--batch',type=int,default=1);p.add_argument('--history',type=int,default=6);p.add_argument('--warmup',type=int,default=1);p.add_argument('--steps',type=int,default=5);p.add_argument('--threads',type=int,default=4);p.add_argument('--output');args=p.parse_args()
    if args.steps<1 or args.warmup<0 or args.history<1 or args.batch<1 or min(args.height,args.width)<64 or args.height%32 or args.width%32:p.error('positive steps/batch/history, nonnegative warmup, dimensions >=64 divisible by32 required')
    run(args)
