import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, List, Sequence
from einops import rearrange
from basicsr.archs.arch_util import LayerNorm, DownSample, ConvLayer2d, make_divisible
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.local_arch import Local_Base
from mmcv.ops import ModulatedDeformConv2d
import math
import numpy as np
import time

 
class SimpleGate(nn.Module):
    def __init__(self):
        super(SimpleGate, self).__init__()
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
    
class SimpleSeModule(nn.Module):
    def __init__(self, inp_chan):
        super(SimpleSeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp_chan, inp_chan, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x * self.se(x)
    

class SeModule(nn.Module):
    def __init__(self, inp_chan, reduction=4):
        """
        Channel Attention aplied in MobileNetv3
        """
        super(SeModule, self).__init__()
        expand_size =  max(inp_chan // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp_chan, expand_size, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, inp_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
    

class HFBlock(nn.Module):
    def __init__(self):
        super(HFBlock, self).__init__()
        self.down = nn.AvgPool2d(2)
    def forward(self, x):
        y = self.down(x)
        high = x - F.interpolate(y, size = x.size()[-2:], mode='bilinear', align_corners=True)
        return high


class ModulatedDeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = ModulatedDeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

        self.init_offset()

    def init_offset(self):
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, input):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output


class InvertedResidualSE(nn.Module):
    def __init__(
        self,
        inp_chan: int,
        expand_ratio: Optional[int] = 2,
        ffn_ratio: Optional[int] = 2,
        bias: Optional[bool] = False,
        # skip_connection: Optional[bool] = True,
    ):
        super(InvertedResidualSE, self).__init__()
        dw_chan = inp_chan * expand_ratio
        ffn_chan = inp_chan * ffn_ratio

        self.norm1 = LayerNorm(inp_chan)
        self.norm2 = LayerNorm(inp_chan)
        
        self.pw1 = nn.Conv2d(inp_chan, dw_chan, kernel_size=1, bias=bias)
        self.dw = nn.Conv2d(dw_chan, dw_chan, kernel_size=3, padding=1, groups=dw_chan, bias=bias)

        self.sca = SeModule(dw_chan)

        self.pw2 = nn.Conv2d(dw_chan, inp_chan, kernel_size=1, bias=bias)

        self.ffn_conv1 = nn.Conv2d(inp_chan, ffn_chan, kernel_size=1, bias=bias)

        self.ffn_conv2 = nn.Conv2d(ffn_chan, inp_chan, kernel_size=1, bias=bias)

        self.beta = nn.Parameter(torch.zeros((1, inp_chan, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, inp_chan, 1, 1)), requires_grad=True)

        self.inp_chan = inp_chan
        self.ffn_chan = ffn_chan
        self.dw_chan = dw_chan

    def forward(self, inp: Tensor) -> Tensor:
        # LayerNorm
        x = self.norm1(inp)
        # Depthwise Separable Convolutions
        x = self.pw1(x)
        x = self.dw(x)
        x = F.gelu(x)
        # Squeeze Channel Attention
        x = self.sca(x)
        x = self.pw2(x)

        y = x * self.gamma + inp

        # ffn
        x = self.norm2(y)
        x = self.ffn_conv1(x)
        x = F.gelu(x)
        x = self.ffn_conv2(x)

        oup = x * self.beta + y

        return oup
    

class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_w: int = 2,
        patch_h: int = 2,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv = nn.Conv2d(embed_dim, 1 + (2 * embed_dim), kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(1 + (2 * embed_dim), 1 + (2 * embed_dim), kernel_size=3, stride=1, padding=1, groups=1 + (2 * embed_dim), bias=bias)
        self.project_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)

        self.embed_dim = embed_dim

        self.patch_w = patch_w
        self.patch_h = patch_h

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)
    
    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:

        _, _, H, W = x.shape

        x = self.resize_input_if_needed(x)
        # [B, C, P, N] --> [B, h + 2d, P, N]
        patches, output_size = self.unfolding_pytorch(x)

        qkv = self.qkv_dwconv(self.qkv(patches))

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        # context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)

        out = self.project_out(out)

        out = self.folding_pytorch(out, output_size)

        return out[:, :, :H, :W]

    def forward(
        self, x: Tensor, *args, **kwargs
    ) -> Tensor:
        return self._forward_self_attn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, patch_w, patch_h, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, patch_h=patch_h, patch_w=patch_w, bias=bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



@ARCH_REGISTRY.register()
class Reformer_v4(nn.Module):
    def __init__(self, 
        inp_ch: Optional[int] = 3, 
        width: Optional[int] = 32,
        patch_w: Optional[int] = 2,
        patch_h: Optional[int] = 2,
        middle_blk_num: Optional[int]=1,
        middle_use_attn: Optional[bool] = 0,
        enc_blk_nums: Optional[List[int]] = [1, 1, 1, 28],
        enc_use_attns: Optional[List[bool]] = [0, 0, 0, 0],
        dec_blk_nums: Optional[List[int]] = [1, 1, 1, 1],
        dec_use_attns: Optional[List[bool]] = [0, 0, 0, 0],
        dw_expand: Optional[int] = 2,
        ffn_expand: Optional[int] = 2,
        bias: Optional[bool] = False, 
    ):
        super(Reformer_v4, self).__init__()
        
        self.intro = nn.Conv2d(in_channels=inp_ch, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
        self.ending = nn.Conv2d(in_channels=width, out_channels=inp_ch, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num, enc_use_attn in zip(enc_blk_nums, enc_use_attns):
            self.downs.append(
                DownSample(chan, bias=bias)
            )
            if enc_use_attn:
                self.encoders.append(nn.Sequential(
                    *[TransformerBlock(chan, 
                                       patch_h=patch_h, 
                                       patch_w=patch_w,
                                       ffn_expansion_factor=ffn_expand,
                                       bias=bias) for _ in range(num)]
                    )
                )

            else:
                self.encoders.append(
                    nn.Sequential(
                        *[InvertedResidualSE(chan, 
                                             expand_ratio=dw_expand, 
                                             ffn_ratio=ffn_expand) for _ in range(num)]
                    )
                )
            chan = chan * 2

        if middle_use_attn:
            self.middle_blks = nn.Sequential(
                        *[TransformerBlock(chan, 
                                        patch_h=patch_h, 
                                        patch_w=patch_w,
                                        ffn_expansion_factor=ffn_expand,
                                        bias=bias) for _ in range(num)]
                        )
            
        else:
            self.middle_blks = nn.Sequential(
                        *[InvertedResidualSE(chan, 
                                             expand_ratio=dw_expand, 
                                             ffn_ratio=ffn_expand) for _ in range(middle_blk_num)]
                        )


        for num, dec_use_attn in zip(dec_blk_nums, dec_use_attns):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=bias),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            if dec_use_attn:
                self.decoders.append(nn.Sequential(
                    *[TransformerBlock(chan, 
                                       patch_h=patch_h, 
                                       patch_w=patch_w,
                                       ffn_expansion_factor=ffn_expand,
                                       bias=bias) for _ in range(num)]
                    )
                )

            else:
                self.decoders.append(
                    nn.Sequential(
                        *[InvertedResidualSE(chan, expand_ratio=dw_expand, ffn_ratio=ffn_expand) for _ in range(num)]
                    )
                )
        
        # self.dcn1 = ModulatedDeformConvWithOff(width * 4, width * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)
        # self.dcn2 = ModulatedDeformConvWithOff(width * 4, width * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp_img: torch.Tensor): #shape: 1 x 3 x 256 x 256

        _, _, H, W = inp_img.shape
        inp_img = self.check_image_size(inp_img)
        
        x = self.intro(inp_img)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x) + inp_img

        return x[:, :, :H, :W]
        
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# @ARCH_REGISTRY.register()
class ReformerLocal(Local_Base, Reformer_v4):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        Reformer_v4.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == "__main__":

    # inp_chan = 256
    # transformer_chan = 256
    # ffn_dim = transformer_chan * 2
    # n_attn_blocks = 4
    # patch_h = 2
    # patch_w = 2
    # conv_ksize = 3

    # net = MobileViTBlockv2(in_channels=inp_chan, transformer_dim=transformer_chan, 
    #                        ffn_dim=ffn_dim, n_attn_blocks=n_attn_blocks, attn_dropout=0.,
    #                        dropout=0., ffn_dropout=0., patch_h=patch_h, patch_w=patch_w, conv_ksize=conv_ksize)
    
    # net = MobileViTBlock(in_channels=inp_chan, transformer_dim=transformer_chan,
    #                      ffn_dim=ffn_dim, n_transformer_blocks=n_attn_blocks, head_dim=4, 
    #                      attn_dropout=0., dropout=0., ffn_dropout=0.,
    #                      patch_h=patch_h, patch_w=patch_w, conv_ksize=conv_ksize, no_fusion=False)

    net = Reformer_v4(inp_ch=3, width=16, patch_h=2, patch_w=2, 
                      middle_blk_num=8, middle_use_attn=True, 
                      enc_blk_nums=[2, 4, 4, 6], enc_use_attns=[0, 0, 0, True],
                      dec_blk_nums=[6, 4, 4, 2], dec_use_attns=[True, 0, 0, 0], 
                      dw_expand=2, ffn_expand=2, bias=False)

    inp_shape = (3, 224, 224)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)

    # inp = torch.randn((1, 3, 256, 256))
    
    # net(inp)
