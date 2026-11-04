"""Resumable real-data K=6 HAT training on CPU, MPS or CUDA."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights
from training.freeview_data import FreeviewDataset
from training.hat_port.models import HumanAttnTransformer
from training.prepare_images import prepare
from training.process_lock import acquire_lock

STOP=False

def atomic_save(data,path):
    path=Path(path);temp=path.with_suffix('.partial');torch.save(data,temp);os.replace(temp,path)

def build_model(device,k=6,pretrained=True):
    pa=SimpleNamespace(im_h=320,im_w=512,backbone_config='',pixel_decoder='MSD',special_symbols=['<pad>'],max_traj_length=k+1,pad_idx=0,eos_idx=-1)
    model=HumanAttnTransformer(pa,num_decoder_layers=6,hidden_dim=384,nhead=4,ntask=1,tgt_vocab_size=1,num_output_layers=3,train_encoder=False,train_pixel_decoder=True,dropout=0.,dim_feedforward=512,parallel_arch=True,dorsal_source=['P1'],num_encoder_layers=3,separate_fix_arch=True,combine_pos_emb=True)
    if pretrained:
        print('Инициализация визуального энкодера ImageNet ResNet50 V2 (не DeepGaze).',flush=True)
        model.encoder.backbone.load_state_dict(ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True,check_hash=True))
    return model.to(device)

def logp_for(model,batch,device):
    image,seq,mask,target=[v.to(device) for v in batch]
    tasks=torch.zeros(len(image),device=device,dtype=torch.long)
    # Upstream HAT eval emits sigmoid maps. Use training logits for both
    # train and evaluation; dropout is zero and FrozenBN has no running state.
    pred=model(image,seq,mask,seq,tasks)['pred_fixation_map'].reshape(len(image),-1)
    return torch.log_softmax(pred,dim=1),target

def evaluate(model,dataset,device,batch_size,indices):
    model.train();model.encoder.backbone.eval()
    total=0.;nss_total=0.;count=0
    with torch.no_grad():
        for batch in DataLoader(Subset(dataset,indices),batch_size=batch_size,num_workers=0):
            logp,target=logp_for(model,batch,device);ix=torch.arange(len(target),device=device)
            total+=(-logp[ix,target]).sum().item();prob=logp.exp()
            z=(prob-prob.mean(1,keepdim=True))/prob.std(1,keepdim=True,unbiased=False).clamp_min(1e-8)
            nss_total+=z[ix,target].sum().item();count+=len(target)
            if STOP:break
    return {'nll_nats_grid':total/count,'nss_grid':nss_total/count,'count':count}

def validate_resume(previous, current, allow_transfer=False):
    core = ('schema','model','history','seed','batch','resolution','source_sha256',
            'backbone_init','frozen_backbone','loss','train_targets','valid_targets')
    environment = ('source_code_sha256','torch','device')
    for field in core + (() if allow_transfer else environment):
        if previous[field] != current[field]:
            raise ValueError(f'resume mismatch: {field}')
    return {field: {'from': previous[field], 'to': current[field]}
            for field in environment if previous[field] != current[field]}


def main(args):
    global STOP
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(output/'run.lock')
    def request_stop(sig,frame):
        global STOP
        STOP=True;print('\nОстановка запрошена. Завершаю текущий шаг и сохраняю веса…',flush=True)
    for name in ('SIGINT','SIGTERM','SIGHUP','SIGBREAK'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),request_stop)
    torch.manual_seed(42);torch.set_num_threads(4)
    root=Path(args.data).resolve();fix=root/'raw/COCOFreeView_fixations_trainval.json'
    try:images=prepare(root,stop_requested=lambda: STOP) if args.prepare else root/'images'
    except InterruptedError:return
    if STOP:return
    train=FreeviewDataset(fix,images,'train');valid=FreeviewDataset(fix,images,'valid')
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob('*.py')))).hexdigest()
    cfg=dict(schema=1,model='HAT-Mac-FV',history=6,seed=42,batch=args.batch,epochs=args.epochs,
             resolution=[512,320],source_sha256=hashlib.sha256(fix.read_bytes()).hexdigest(),
             source_code_sha256=code_hash,backbone_init='ImageNet ResNet50 V2',frozen_backbone=True,
             loss='spatial log_softmax NLL on 128x80 grid',torch=str(torch.__version__),device=args.device,
             train_targets=len(train),valid_targets=len(valid),duration_prediction=False,
             full_validation_each_epoch=True,health_validation_count=min(2048,len(valid)),
             early_stopping='two full-validation epochs without NLL improvement >=0.001')
    checkpoint=output/'latest.pt';resume=checkpoint.exists()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False) if resume else None
    if state:
        changes=validate_resume(state['config'],cfg,args.allow_transfer)
        if changes:
            transfer=dict(step=state['step'],changes=changes,note='Explicit environment transfer; not bitwise reproducible')
            with (output/'transfers.jsonl').open('a') as f:f.write(json.dumps(transfer)+'\n')
            print(f'Перенос окружения, продолжаем с шага {state["step"]}: {changes}',flush=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    model=build_model(args.device,pretrained=not resume)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    epoch=0;cursor=0;step=0;best=math.inf;bad=0;prior_seconds=0.
    if state:
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for values in opt.state.values():
            for key,v in values.items():
                if isinstance(v,torch.Tensor) and key!='step':values[key]=v.to(args.device)
        epoch=state['epoch'];cursor=state['cursor'];step=state['step'];best=state['best'];bad=state['bad_epochs'];prior_seconds=state['elapsed_seconds']
        torch.set_rng_state(state['cpu_rng'])
        if args.device=='mps' and state.get('mps_rng') is not None:torch.mps.set_rng_state(state['mps_rng'])
        if args.device=='cuda' and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.monotonic();metrics=(output/'metrics.jsonl').open('a',buffering=1)
    health=torch.randperm(len(valid),generator=torch.Generator().manual_seed(43))[:min(2048,len(valid))].tolist()
    def save(name='latest.pt'):
        atomic_save(dict(config=cfg,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,cursor=cursor,step=step,best=best,bad_epochs=bad,elapsed_seconds=prior_seconds+time.monotonic()-started,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if args.device=='mps' else None,cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),output/name)
    print(f'Обучение K=6 | {len(train):,} целей | {len(valid):,} целей валидации | {args.device} | пакет {args.batch}',flush=True)
    print(f'Журнал: {output}/metrics.jsonl\nВеса: {output}/latest.pt\nCtrl+C — сохранить и остановить. Повторный запуск — продолжить.',flush=True)
    if args.check_only:
        batch=next(iter(DataLoader(train,batch_size=args.batch)));model.train();model.encoder.backbone.eval();opt.zero_grad();lp,t=logp_for(model,batch,args.device);loss=-lp[torch.arange(len(t),device=args.device),t].mean();loss.backward()
        assert torch.isfinite(loss) and model.fixation_embed.layers[-1].weight.grad.abs().max()>0
        print(f'Проверка реального обучающего шага: NLL={loss.item():.4f}',flush=True)
        print(evaluate(model,valid,args.device,args.batch,health[:args.batch*2]),flush=True);return
    last_save=time.monotonic();running_seconds=0.;running_examples=0
    while epoch<args.epochs and not STOP:
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        loader=DataLoader(Subset(train,order[cursor:]),batch_size=args.batch,num_workers=0)
        model.train();model.encoder.backbone.eval()
        for batch in loader:
            before=time.monotonic();opt.zero_grad(set_to_none=True);lp,target=logp_for(model,batch,args.device)
            loss=-lp[torch.arange(len(target),device=args.device),target].mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; refusing checkpoint overwrite')
            loss.backward();grad=model.fixation_embed.layers[-1].weight.grad
            if grad is None or not torch.isfinite(grad).all():raise RuntimeError('Invalid prediction gradients')
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
            opt.step()
            if args.device=='mps':torch.mps.synchronize()
            if args.device=='cuda':torch.cuda.synchronize()
            cursor+=len(target);step+=1;running_seconds+=time.monotonic()-before;running_examples+=len(target)
            if step%25==0 or STOP:
                rate=running_examples/running_seconds;eta=(len(train)-cursor)/rate
                item=dict(event='train',epoch=epoch+1,step=step,examples=cursor,total=len(train),nll=loss.item(),examples_per_second=rate,epoch_eta_seconds=eta,elapsed_seconds=prior_seconds+time.monotonic()-started)
                metrics.write(json.dumps(item)+'\n')
                print(f'Эпоха {epoch+1}/{args.epochs} | {cursor/len(train):6.2%} | шаг {step:,} | NLL {loss.item():.4f} | {rate:.1f} прим./с | до конца эпохи {eta/3600:.2f} ч',flush=True)
                (output/'status.json').write_text(json.dumps(item,indent=2)+'\n')
            if step%1000==0 or time.monotonic()-last_save>args.save_seconds or STOP:
                save();last_save=time.monotonic();print(f'Контрольная точка сохранена, шаг {step}',flush=True)
            if STOP:break
            if step%2000==0:
                result=evaluate(model,valid,args.device,args.batch,health);metrics.write(json.dumps(dict(event='validation_health',step=step,**result))+'\n');print(f'Промежуточная валидация: {result}',flush=True)
                if STOP:save();break
        if STOP:break
        print('Полная валидация эпохи…',flush=True)
        result=evaluate(model,valid,args.device,args.batch,range(len(valid)));metrics.write(json.dumps(dict(event='validation_full',epoch=epoch+1,step=step,**result))+'\n')
        if STOP:save();break
        improved=result['nll_nats_grid']<best-.001
        if improved:best=result['nll_nats_grid'];bad=0
        else:bad+=1
        epoch+=1;cursor=0;save()
        if improved:save('best.pt')
        print(f'Эпоха завершена. Валидация: {result}. Лучшая NLL: {best:.4f}',flush=True)
        if bad>=2:print('Ранняя остановка по полной валидации.',flush=True);break
    save();print('Остановлено с сохранением.' if STOP else 'Обучение завершено.',flush=True)
    metrics.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');p.add_argument('--batch',type=int,default=4);p.add_argument('--epochs',type=int,default=5);p.add_argument('--prepare',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--allow-transfer',action='store_true',help='Explicitly allow code/PyTorch/device migration while preserving data/model/batch invariants');p.add_argument('--save-seconds',type=float,default=600);args=p.parse_args()
    if args.batch<1 or args.epochs<1 or args.save_seconds<=0:p.error('batch and epochs must be positive')
    main(args)
