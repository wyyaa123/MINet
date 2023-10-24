import torch
import torch.nn as nn
# from basicsr.archs.arch_util import LayerNorm2d
# from basicsr.models.archs.local_arch import Local_Base
from mmcv.ops import ModulatedDeformConv2d
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# mobileNet_v3
class BaseBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(BaseBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class BaseFPN(nn.Module):
    def __init__(self, img_channel, width=16, num_filters=128, act=nn.Hardswish):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, 
                               kernel_size=3, padding=1,bias=True)
        

        self.bneck = nn.Sequential(
            BaseBlock(kernel_size=3, in_size=16, expand_size=16, out_size=16, act=nn.ReLU, se=True, stride=2),
            #
            BaseBlock(kernel_size=3, in_size=16, expand_size=72, out_size=24, act=nn.ReLU, se=False, stride=2),
            BaseBlock(kernel_size=3, in_size=24, expand_size=88, out_size=24, act=nn.ReLU, se=False, stride=1),
            #
            BaseBlock(kernel_size=5, in_size=24, expand_size=96, out_size=40, act=act, se=True, stride=2),
            BaseBlock(kernel_size=5, in_size=40, expand_size=240, out_size=40, act=act, se=True, stride=1),
            BaseBlock(kernel_size=5, in_size=40, expand_size=240, out_size=40, act=act, se=True, stride=1),
            BaseBlock(kernel_size=5, in_size=40, expand_size=120, out_size=48, act=act, se=True, stride=1),
            BaseBlock(kernel_size=5, in_size=48, expand_size=144, out_size=48, act=act, se=True, stride=1),
            #
            BaseBlock(kernel_size=5, in_size=48, expand_size=288, out_size=96, act=act, se=True, stride=2),
            BaseBlock(kernel_size=5, in_size=96, expand_size=576, out_size=96, act=act, se=True, stride=1),
            BaseBlock(kernel_size=5, in_size=96, expand_size=576, out_size=96, act=act, se=True, stride=1),
        )

        self.encoder0 = nn.Sequential(self.bneck[0:1]) # inp: 16, oup: 16, shape: down
        self.encoder1 = nn.Sequential(self.bneck[1:3]) # inp: 16, oup: 24, shape: down
        self.encoder2 = nn.Sequential(self.bneck[3:8]) # inp: 24, oup: 48, shape: down
        self.encoder3 = nn.Sequential(self.bneck[8:])  # inp: 48, oup: 96, shape: down

        # self.lateral3 = nn.Conv2d(96, num_filters, kernel_size=1, bias=False) # inp: 96, oup: 128, shape: unchange
        # self.lateral2 = nn.Conv2d(48, num_filters // 2, kernel_size=1, bias=False) # inp: 48, oup: 64, shape: unchange
        # self.lateral1 = nn.Conv2d(24, num_filters // 4, kernel_size=1, bias=False) # inp: 24, oup: 32, shape: unchange
        # self.lateral0 = nn.Conv2d(16, num_filters // 8, kernel_size=1, bias=False) # inp: 16, oup: 16, shape: unchange


        self.lateral3 = nn.Conv2d(96, num_filters, kernel_size=1, bias=False) # inp: 96, oup: 128, shape: unchange
        self.lateral2 = nn.Conv2d(48, num_filters, kernel_size=1, bias=False) # inp: 48, oup: 128, shape: unchange
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False) # inp: 24, oup: 128, shape: unchange
        self.lateral0 = nn.Conv2d(16, num_filters, kernel_size=1, bias=False) # inp: 16, oup: 128, shape: unchange

        # self.td1 = nn.Sequential(nn.Conv2d(num_filters // 2, num_filters // 2, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(num_filters // 2),
        #                          nn.ReLU(inplace=True)) # inp: 64, oup: 64
        
        # self.td2 = nn.Sequential(nn.Conv2d(num_filters // 4, num_filters // 4, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(num_filters // 4),
        #                          nn.ReLU(inplace=True)) # inp: 64, oup: 128
        
        # self.td3 = nn.Sequential(nn.Conv2d(num_filters // 8, num_filters // 8, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(num_filters // 8),
        #                          nn.ReLU(inplace=True)) # inp: 128, oup: 128


        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True)) # inp: 64, oup: 64
        
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True)) # inp: 64, oup: 128
        
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True)) # inp: 128, oup: 128
        
        # self.conv3 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size=1)
        # self.conv2 = nn.Conv2d(num_filters // 2, num_filters, kernel_size=1)
        # self.conv1 = nn.Conv2d(num_filters // 4, num_filters // 2, kernel_size=1)
        
        self.padder_size = 2 ** 4

    def forward(self, inp):

        inp = self.intro(inp)

        # Bottom-up pathway, from MobileNet_v3
        enc0 = self.encoder0(inp) # inp: 16, oup: 16, shape: 1/2

        enc1 = self.encoder1(enc0) # inp: 16, oup: 24, shape: 1/4

        enc2 = self.encoder2(enc1) # inp: 24, oup: 48, shape: 1/8

        enc3 = self.encoder3(enc2) # inp: 48, oup: 96, shape: 1/16

        # Lateral connections
        # lateral3 = self.lateral3(enc3) # inp: 96, oup: 128, shape: 1/16
        # lateral2 = self.lateral2(enc2) # inp: 48, oup: 64, shape: 1/8
        # lateral1 = self.lateral1(enc1) # inp: 24, oup: 32, shape: 1/4
        # lateral0 = self.lateral0(enc0) # inp: 16, oup: 16, shape: 1/2

        lateral3 = self.lateral3(enc3) # inp: 96, oup: 128, shape: 1/16
        lateral2 = self.lateral2(enc2) # inp: 48, oup: 128, shape: 1/8
        lateral1 = self.lateral1(enc1) # inp: 24, oup: 128, shape: 1/4
        lateral0 = self.lateral0(enc0) # inp: 16, oup: 128, shape: 1/2

        # Top-down pathway
        map3 = lateral3 # channel: 128, shape: 1/16
        # map2 = self.td1(lateral2 + F.pixel_shuffle(self.conv3(map3), upscale_factor=2)) # channel: 64, shape: 1/8
        # map1 = self.td2(lateral1 + F.pixel_shuffle(self.conv2(map2), upscale_factor=2)) # channel: 32, shape: 1/4
        # map0 = self.td3(lateral0 + F.pixel_shuffle(self.conv1(map1), upscale_factor=2)) # channel: 16, shape: 1/2

        map2 = self.td1(lateral2 + F.interpolate(map3, scale_factor=2, mode="bilinear")) # channel: 128, shape: 1/8
        map1 = self.td2(lateral1 + F.interpolate(map2, scale_factor=2, mode="bilinear")) # channel: 128, shape: 1/4
        map0 = self.td3(lateral0 + F.interpolate(map1, scale_factor=2, mode="bilinear")) # channel: 128, shape: 1/2
        return map0, map1, map2, map3 
    

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.block0(x), inplace=True)
        x = F.relu(self.block1(x), inplace=True)
        return x
    

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
    
@ARCH_REGISTRY.register()
class BaseFPNNet(nn.Module):

    def __init__(self, inp_ch, output_ch=3, num_filters=64, num_filters_fpn=128):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = BaseFPN(inp_ch)

        # The segmentation heads on top of the FPN

        self.head0 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.dcn1 = ModulatedDeformConvWithOff(num_filters * 4, num_filters * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)
        self.dcn2 = ModulatedDeformConvWithOff(num_filters * 4, num_filters, kernel_size=3, padding=1, stride=1, deformable_groups=2)

        self.ending = nn.Conv2d(in_channels=num_filters, out_channels=output_ch, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.padder_size = 2 ** 4

    def forward(self, inp):

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        map0, map1, map2, map3 = self.fpn(inp)

        map3 = F.interpolate(self.head3(map3), scale_factor=8, mode="bilinear")
        map2 = F.interpolate(self.head2(map2), scale_factor=4, mode="bilinear")
        map1 = F.interpolate(self.head1(map1), scale_factor=2, mode="bilinear")
        map0 = F.interpolate(self.head0(map0), scale_factor=1, mode="bilinear")

        oup = torch.cat([map3, map2, map1, map0], dim=1)

        oup = self.dcn1(oup)

        oup = self.dcn2(oup)

        oup = F.interpolate(oup, scale_factor=2, mode="bilinear")

        oup = self.ending(oup) + inp

        return oup[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size #检查宽高是否为256的倍数
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    

if __name__ == "__main__":
    net = BaseFPNNet(3)

    # inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print(macs, params)

    inp = torch.randn((1, 128, 64, 64))

    net = nn.PixelShuffle(8)

    oup = net(inp)

    print()

