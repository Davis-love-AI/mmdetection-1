import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class PAFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(PAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.downup_sampling=ModuleList()
        self.res_convs=ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)
        for i in range(self.start_level, self.backbone_end_level+1):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        for i in range(self.start_level, self.backbone_end_level):
            down_sampling=nn.MaxPool2d(2,stride=2)
            self.downup_sampling.append(down_sampling)
        for i in range(self.start_level, self.backbone_end_level):
            res_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.res_convs.append(res_conv)
        self.conv_p6=ConvModule(in_channels[-1],out_channels,3,stride=2,padding=1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                activation=self.activation,
                                inplace=False)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # The inputs are four ndarrays. The first is (4,256,104,336).
        # The known inputs are 800 and 1333 in size. Here is the resnet output corresponding to a layer P3-P6 of stride 8.

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        #self.lateral_conv Four 1 * 1 convolutions unify the number of channels of four different input features [256, 512, 1024, 2048] to 256
        p6=self.conv_p6(inputs[-1])
        #p6_down=F.interpolate(p6,scale_factor=2,mode='nearest')
        p6_down = p6.Upsample(scale_factor=2,mode='nearest')
        # build top-down path
        used_backbone_levels = len(laterals)
        #laterals interpolate
        #new = F.interpolate(laterals[used_backbone_levels-1],[26,34],mode='bilinear',align_corners=False)
        laterals[used_backbone_levels-1]+=p6_down
        # new+=p6_down
        #laterals[used_backbone_levels-1]+=p6_down
        #laterals[used_backbone_levels-1] = laterals[used_backbone_levels-1] + p6_down

        # for i in range(used_backbone_levels - 1, 0, -1):
        #     laterals[i - 1] += F.interpolate(
        #         laterals[i], scale_factor=2, mode='nearest')

        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        #Upsampling using nearest neighbor difference values ​​Starting from P5 stride 32 layerals [3]
        #Take two sides [3] upsampling and add them to laterals [2]

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        #Is a set of 3 * 3 convolutions to eliminate the soul stack effect of the fused P3-P6 features
        outs.append(p6)
        #out0 is P3 maximum resolution
        #print(len(self.downup_sampling)) #4
        #print(used_backbone_levels) #4
        N=[]
        for i in range(used_backbone_levels+1):
            if i==0:
                N.append(outs[i])
            else:
                N.append(self.downup_sampling[i-1](N[i-1])+outs[i])
            # if i==5:
            #     test4 = F.interpolate(self.downup_sampling[4](N[4]),[13,17],mode='bilinear',align_corners=False)
            #     N.append(test4+outs[i])
            # if i < 5:
            #     print(self.downup_sampling[i-1](N[i-1]).size())
            #     print(outs[i].size())
            #     N.append(self.downup_sampling[i-1](N[i-1])+outs[i])
            #     #N.append(F.interpolate(N[i-1],scale_factor=0.5,mode='nearest')+outs[i])
            # else:
            #     print(self.downup_sampling[i-1](N[i-1]).size())
            #     print(outs[i].size())
            #     N.append(self.downup_sampling[i-1](N[i-1])+outs[i])
            #     #N.append(F.interpolate(N[i-1],scale_factor=0.5,mode='nearest')+outs[i])

        #Residual block
        res_out=[]
        for i in range(used_backbone_levels):
            res_out.append(N[i]+self.res_convs[i](inputs[i]))
        res_out.append(N[-1]+p6)
        #Another set of 33 convolutions to eliminate soul stacks

        outs2 = [
            self.fpn_convs[i](res_out[i]) for i in range(used_backbone_levels+1)]
        #It is a set of 3 * 3 convolutions to eliminate the soul stack effect of the fused P3-P6 features.

        # if self.num_outs>len(res_out):
        #     if not self.add_extra_convs:
        #         for i in range(self.num_outs-used_backbone_levels):
        #             outs2.append(F.max_pool2d(outs[-1],1,stride=2))
        return tuple(outs2)

        # # part 2: add extra levels
        # if self.num_outs > len(outs):
        #     # use max pool to get more levels on top of outputs
        #     # (e.g., Faster R-CNN, Mask R-CNN)
        #     if not self.add_extra_convs:
        #         for i in range(self.num_outs - used_backbone_levels):
        #             outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        #     # add conv layers on top of original feature maps (RetinaNet)
        #     else:
        #         if self.extra_convs_on_inputs:
        #             orig = inputs[self.backbone_end_level - 1]
        #             outs.append(self.fpn_convs[used_backbone_levels](orig))
        #         else:
        #             outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
        #         for i in range(used_backbone_levels + 1, self.num_outs):
        #             if self.relu_before_extra_convs:
        #                 outs.append(self.fpn_convs[i](F.relu(outs[-1])))
        #             else:
        #                 outs.append(self.fpn_convs[i](outs[-1]))
        # return tuple(outs)
