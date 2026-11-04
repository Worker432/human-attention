"""Official padded-canvas data, chronological windows, no DeepGaze inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class FreeviewDataset(Dataset):
    def __init__(self,fixations,images,split,history=6,width=512,height=320):
        if history<0:raise ValueError('negative history')
        self.root=Path(images);self.k=history;self.width=width;self.height=height
        self.rows=[r for r in json.loads(Path(fixations).read_text()) if r['split']==split]
        self.examples=[(r,t) for r,row in enumerate(self.rows) for t in range(1,row['length'])]
        self.mean=torch.tensor([.485,.456,.406])[:,None,None];self.std=torch.tensor([.229,.224,.225])[:,None,None]
        if not self.examples:raise ValueError(f'empty split {split}')
        for name in {r['name'] for r in self.rows}:
            if Path(name).name!=name or not (self.root/name).is_file():raise ValueError(f'missing/unsafe image {name}')
    def __len__(self):return len(self.examples)
    def __getitem__(self,index):
        ri,t=self.examples[index];r=self.rows[ri]
        with Image.open(self.root/r['name']) as image:
            if image.size!=(1680,1050):raise ValueError('expected official 1680x1050 padded canvas')
            image=image.convert('RGB').resize((self.width,self.height),Image.Resampling.BILINEAR)
            rgb=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255.
        gw,gh=self.width//4,self.height//4
        def cell(j):
            x,y=r['X'][j],r['Y'][j]
            if not (0<=x<1680 and 0<=y<1050):raise ValueError('fixation outside presentation canvas')
            return min(gh-1,int(y/1050*gh))*gw+min(gw-1,int(x/1680*gw))
        previous=list(range(max(0,t-self.k),t)) if self.k else []
        seq=[cell(j)+1 for j in previous]+[0]*(self.k-len(previous))
        mask=[False]*len(previous)+[True]*(self.k-len(previous))
        return ( (rgb-self.mean)/self.std,torch.tensor(seq,dtype=torch.long),torch.tensor(mask,dtype=torch.bool),torch.tensor(cell(t),dtype=torch.long))
