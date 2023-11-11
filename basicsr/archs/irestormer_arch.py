import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class IRestormer(nn.Module):
    def __init__(self, inp_ch=3, width=32, output_ch=3, enc_blks=[2, 4, 4, 8], dec_blks=[1, 1, 1, 1]):
        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
       pass
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size #检查宽高是否为padder_size的倍数
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

