import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Common import FAFE, LLFR, TripletAttention 
# Full code will be made public after the paper is accepted.
            
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=nn.LeakyReLU(0.1, inplace=True)):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, pretrained=False):
        super().__init__()
        ids = [-1, -2, -3, -4]
        
        self.stem = Conv(3, transition_channels * 2, 3, 2)
        
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 4, n=n, ids=ids),
        )
        self.dark3 = nn.Sequential(
            MP(),
            Multi_Concat_Block(transition_channels * 4, block_channels * 4, transition_channels * 8, n=n, ids=ids),
        )
        self.dark4 = nn.Sequential(
            MP(),
            Multi_Concat_Block(transition_channels * 8, block_channels * 8, transition_channels * 16, n=n, ids=ids),
        )
        self.dark5 = nn.Sequential(
            MP(),
            Multi_Concat_Block(transition_channels * 16, block_channels * 16, transition_channels * 32, n=n, ids=ids),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

class SceneRouter(nn.Module):
    # Feature Compression Module (FCM) 
    # Scene Router (SR)
    # Scene Discriminator (SD)


class DomainSpecificBlock(nn.Module):
    # Fog-Aware Feature Enhancement (FAFE) 
    # Low-Light Feature Refinement (LLFR) 
    # Triplet Attention (TA)

class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))
    
class ELMProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.random_weights = nn.Parameter(
            torch.randn(hidden_size, in_features), requires_grad=False
        )
        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_size, in_features)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        hidden = self.activation(torch.matmul(x_flat, self.random_weights.t()))  # [B, HW, hidden]
        output = self.linear(hidden)  # [B, HW, C]
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output

    
def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

class SSAN(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(SSAN, self).__init__()
        transition_channels = 16
        block_channels = 16
        panet_channels = 16
        e = 1
        n = 2
        ids = [-1, -2, -3, -4]

        self.backbone = Backbone(transition_channels, block_channels, n)
        self.scene_router = SceneRouter()

        self.fog_branch = DomainSpecificBlock(transition_channels * 4, is_fog=True)
        self.dark_branch = DomainSpecificBlock(transition_channels * 4, is_fog=False)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv3_for_upsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)
        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv3_for_upsample2 = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)
        self.down_sample1 = Conv(transition_channels * 4, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)
        self.down_sample2 = Conv(transition_channels * 8, transition_channels * 16, k=3, s=2)
        self.conv3_for_downsample2 = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)
        self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 32, 3, 1)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

        self.elm_proj_P3 = ELMProjection(transition_channels * 8, hidden_size=64)
        self.elm_proj_P4 = ELMProjection(transition_channels * 16, hidden_size=64)
        self.elm_proj_P5 = ELMProjection(transition_channels * 32, hidden_size=64)


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def adaptive_fusion(self, shared_feat, route_weights):
        fog_feat = self.fog_branch(shared_feat)
        dark_feat = self.dark_branch(shared_feat)
        return (route_weights[:,0].view(-1,1,1,1) * fog_feat + 
                route_weights[:,1].view(-1,1,1,1) * dark_feat)

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)  # [1,64,80,80], [1,128,40,40], [1,256,20,20]
        
        route_weights, scene_pred = self.scene_router([feat1, feat2, feat3])
        
        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)
        
        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)

        P3 = torch.cat([self.adaptive_fusion(self.conv_for_feat1(feat1), route_weights), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)


        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        P3 = self.elm_proj_P3(P3)
        P4 = self.elm_proj_P4(P4)
        P5 = self.elm_proj_P5(P5)

        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2], scene_pred, route_weights