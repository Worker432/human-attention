"""Small explicit replacements for the Detectron2 layer utilities HAT uses."""
from dataclasses import dataclass
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F

@dataclass
class ShapeSpec:
    channels: int
    stride: int

class Conv2d(nn.Conv2d):
    def __init__(self,*args,norm=None,activation=None,**kwargs):
        super().__init__(*args,**kwargs);self.norm=norm;self.activation=activation
    def forward(self,x):
        x=F.conv2d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)
        if self.norm is not None:x=self.norm(x)
        return self.activation(x) if self.activation is not None else x

def get_norm(norm,channels):
    if norm=='GN':return nn.GroupNorm(32,channels)
    if norm=='':return None
    raise ValueError(f'Unsupported norm {norm}')

def c2_xavier_fill(layer):
    nn.init.kaiming_uniform_(layer.weight,a=1)
    if layer.bias is not None:nn.init.zeros_(layer.bias)
weight_init=SimpleNamespace(c2_xavier_fill=c2_xavier_fill)
