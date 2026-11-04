"""R50 and official six-layer MSD decoder, with no weight downloads for timing.

Torchvision R50 uses stride in 3x3 as specified by upstream resnet50.yaml.
FrozenBatchNorm matches the upstream default normalization. Random initialization
is for performance measurements only, not a trained or validated gaze model.
"""
from torch import nn
from torchvision.models import resnet50
from torchvision.ops import FrozenBatchNorm2d
from .compat import ShapeSpec
from .pixel_decoder import MSDeformAttnPixelDecoder

class ImageFeatureEncoder(nn.Module):
    def __init__(self,cfg_path,dropout,pixel_decoder):
        super().__init__()
        if pixel_decoder!='MSD':raise ValueError('Only MSD supported')
        self.backbone=resnet50(weights=None,norm_layer=FrozenBatchNorm2d)
        self.pixel_decoder=MSDeformAttnPixelDecoder(
            input_shape={f'res{i}':ShapeSpec(c,s) for i,c,s in [(2,256,4),(3,512,8),(4,1024,16),(5,2048,32)]},
            conv_dim=256,mask_dim=256,norm='GN',transformer_dropout=dropout,
            transformer_nheads=8,transformer_dim_feedforward=1024,
            transformer_enc_layers=6,transformer_in_features=['res3','res4','res5'],common_stride=4)
    def forward(self,x):
        b=self.backbone;x=b.maxpool(b.relu(b.bn1(b.conv1(x))));features={}
        for i,layer in enumerate([b.layer1,b.layer2,b.layer3,b.layer4],2):
            x=layer(x);features[f'res{i}']=x
        high,_,multi=self.pixel_decoder.forward_features(features)
        return high,multi[0],multi[1]
