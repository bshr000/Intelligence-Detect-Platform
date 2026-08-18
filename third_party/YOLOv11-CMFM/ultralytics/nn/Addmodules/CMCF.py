# GACF:Gated Attention-guided Cross-modality Fusion Block (GACF Block)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import  DropPath, trunc_normal_
from math import log


####### CombinedAttention--Compare #######
class SE(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(channels, channels // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(channels // ratio, channels, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out

class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

##PPA
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class ECAmodule(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECAmodule, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECAmodule(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

class GAMAttention(nn.Module):

    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1),
        )
        self.spatial_attention = nn.Sequential(
            (
                nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            (
                nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


##MSA
def get_freq_indices(method):
    # ----------------- 工具函数：频率选择 -------------------
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    # ----------------- DCT主干模块 -------------------
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

class MultiSpectralAttentionLayer(torch.nn.Module):
    # ----------------- MultiSpectralAttentionLayer -------------------
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)

# x = torch.randn(4, 64, 32, 32)  # B, C, H, W
# msa = MultiSpectralAttentionLayer(channel=64, dct_h=7, dct_w=7, reduction=16, freq_sel_method='top16')
# out = msa(x)
# print("输出形状:", out.shape)

#################################################
# LEGNet: Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection

import math
import torch
import torch.nn as nn
from math import log
from timm.layers import DropPath

class Conv_Extra(nn.Module):
    def __init__(self, channel,  act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   nn.BatchNorm2d(channel)
                                   )
    def forward(self, x):
        out = self.block(x)
        return out

#边缘引导注意力
class Scharr(nn.Module):
    def __init__(self, channel, act_layer):
        super(Scharr, self).__init__()
        # 定义Scharr滤波器
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # 将Sobel滤波器分配给卷积层
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)

        # 将mmcv.cvv build_norm_layer，更改为Pytorch 原生BatchNorm
        # self.norm = build_norm_layer(norm_layer, channel)[1]
        self.norm = nn.BatchNorm2d(channel)
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel,  act_layer)

    def forward(self, x):
        # show_feature(x)
        # 应用卷积操作
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # 计算边缘和高斯分布强度（可以选择不同的方式进行融合，这里使用平方和开根号）
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)
        # show_feature(out)

        return out

# 高斯引导注意力
class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        # 生成高斯核
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        # 创建高斯卷积层
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))

        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

# 轻量特征增强注意力
class LFEA(nn.Module):
    # Lightweight Feature Enhancement Attention
    # 类似SE/CBAM结构
    def __init__(self, channel,  act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        # 动态计算1D卷积核大小k，核大小与通道数相关，自适应生成有效通道感知范围
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1

        # 空间卷积分支，对融合特征做空间卷积增强
        self.conv2d = self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channel),
            act_layer())
        # self.norm = nn.BatchNorm2d(channel)
        # 通道注意力分支： 全局平均池化（Squeeze）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        #  进行BN归一化，残差融合后的输出
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        # 输入特征c与att相乘，再加上自身，作为空间注意力融合结果。
        x = self.norm(c + att * wei)

        return x

class LEG(nn.Module):
    def __init__(self,
                 dim,  # 当前特征图的通道数
                 stage, # 当前所在 stage 编号（决定使用 Scharr 还是 Gaussian）
                 mlp_ratio = 2, # 通道扩展倍数（用于 MLP 内部宽度）
                 drop_path = 0.1, # 随机深度残差丢弃率
                 act_layer = nn.ReLU, # 激活函数类（如 nn.ReLU）
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 轻量化MLP（dim → dim × mlp_ratio → dim）仅增强通道建模能力
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )

        self.LFEA = LFEA(dim, act_layer)

        # stage == 0 时使用 Scharr 边缘检测 作为 attention 来源（对初始特征更敏感），否则使用 Gaussian 模糊卷积 作为注意力导向
        if stage == 0:
            self.Scharr_edge = Scharr(dim, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0,  act_layer)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        # show_feature(x)
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

class EGA(nn.Module):
    def __init__(self,
                 dim,              # 输入通道数
                 stage,            # 决定使用 Scharr 或 Gaussian
                 drop_path=0.1,    # DropPath 概率
                 act_layer=nn.ReLU # 激活函数
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.BatchNorm2d(dim)

        # 空间卷积增强分支
        self.conv2d = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            act_layer()
        )

        # CBAM 注意力模块
        self.cbam = CBAM(channels=dim)
        self.sigmoid = nn.Sigmoid()

        # 引导特征：Scharr or Gaussian
        if stage == 0:
            self.att_source = Scharr(dim, act_layer)
        else:
            self.att_source = Gaussian(dim, size=5, sigma=1.0, act_layer=act_layer)

    def forward(self, x):
        # 获取引导注意力图
        att = self.att_source(x)

        # 特征融合：加权 + 卷积增强(顺序是不是需要改下？先 卷积增强+加权)
        att = x * att + x
        att = self.conv2d(att)

        # CBAM 权重
        wei = self.sigmoid(self.cbam(att))

        # 通道加权 + 残差连接 + DropPath + Norm
        out = x + self.norm(self.drop_path(att * wei))
        return out

#############################################################################################

class CombinedAttention(nn.Module):
    """融合CBAM、MSA、边缘注意力"""
    def __init__(self, channels):
        super(CombinedAttention, self).__init__()
        self.se = SE(channels = channels)
        self.cbam = CBAM(channels=channels)
        self.eca = ECA(channels = channels)
        self.ca = CoordAtt(inp=channels, oup=channels)
        self.simam = SimAM(channels=channels, out_channels=channels)
        self.ema = EMA(channels=channels, factor=16)
        self.gma = GAMAttention(c1=channels, c2=channels, group=True, rate=4)
        self.ppa = PPA(in_features=channels, filters=channels)
        self.triplet = TripletAttention(no_spatial=False)  # 默认false是三条路径
        self.msa = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction=16, freq_sel_method='top16')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        SE = self.sigmoid(self.se(x))
        CBAM = self.sigmoid(self.cbam(x))  # 空间注意力
        MSA = self.sigmoid(self.msa(x))  # 频谱通道注意力
        ECA = self.sigmoid(self.eca(x))
        CA = self.sigmoid(self.ca(x))
        EMA = self.sigmoid(self.ema(x))
        PPA = self.sigmoid(self.ppa(x))
        GMA = self.sigmoid(self.gma(x))
        TRIPLET = self.sigmoid(self.triplet(x))
        # return x * CBAM * MSA
        # return x * CBAM * MSA
        return x    # 无注意力； 无注意力

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm1 = nn.LayerNorm(dim)

            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        # B num_heads N C//num_heads
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # B C//num_head N num_heads -> B N C//num_heads num_heads -> B C H W
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            # B C H W -> B C H/R W/R -> B C HW/R² -> B HW/R² C
            x_1 = self.sr1(x_1).reshape(B, C, -1).permute(0, 2, 1)
            x_1 = self.norm1(x_1)
            # B HW/R² C -> B HW/R² 2C -> B HW/R² 2 num_heads C//num_heads -> 2 B num_heads HW/R² C//num_heads
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2).reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # B num_heads HW/R² C//num_heads
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        # B num_heads N HW/R²
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        # B num_heads N C//num_heads -> B N num_heads C//num_heads -> B N C
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        return main_out, aux_out

class FeatureInteraction(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=2, sr_ratio=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, H, W):
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        c1, c2 = self.cross_attn(z1, z2, H, W)
        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out = self.norm2(x2 + self.end_proj2(y2))
        return main_out, aux_out

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)

        return out

class GACF(nn.Module):
    def __init__(self, c1,c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(GACF, self).__init__()
        self.att1 = CombinedAttention(c1)
        self.att2 = CombinedAttention(c1)


        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.channel_emb = ChannelEmbed(in_channels=c1 * 2, out_channels=c1, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)
        self.gate_conv = nn.Conv2d(c1 * 2, 2, kernel_size=1) # softmax 门控机制，二者进行竞争式加权

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #  Combined Attention:CBAM + MSA
        x1 = self.att1(x[0])
        x2 = self.att2(x[1])

        #  Cross Attention
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B C H W
        B, N, _C = x1.shape
        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W)

        # Gated fusion（Softmax门控）
        gate_input = torch.cat([x1, x2], dim=1)  # B, 2C, H, W
        gate = torch.softmax(self.gate_conv(gate_input), dim=1)  # B, 2, H, W
        gate1 = gate[:, 0:1, :, :]
        gate2 = gate[:, 1:2, :, :]

        gated_x1 = gate1 * x1
        gated_x2 = gate2 * x2
        # fuse = gated_x1 + gated_x2 + (x[0] + x[1]) # 残差连接，增强原始特征表达
        fuse = gated_x1 + gated_x2
        return fuse



class GACF_sigmoid(nn.Module):
    def __init__(self, c1,c2, reduction=16, sr_ratio=4, num_heads=4, norm_layer=nn.BatchNorm2d):  # 修改头数
        super(GACF_sigmoid, self).__init__()
        self.att1 = CombinedAttention(c1)
        self.att2 = CombinedAttention(c1)

        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.fft= FeedForwardNetwork(in_channels=c1 * 2, out_channels=c1*2, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

        self.norm = norm_layer(c2)

        # self.gate_conv = nn.Sequential(
        #     nn.Conv2d(c1 * 2, c1, kernel_size=1),
        #     nn.Sigmoid()
        # )

        # self.gate_conv = nn.Conv2d(c1 * 2, 2, kernel_size=1) # softmax 门控机制，二者进行竞争式加权
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1) #sigmoid 门控机制
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #  Combined Attention:CBAM + MSA
        x1 = self.att1(x[0])
        x2 = self.att2(x[1])

        #  Cross Attention
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2, H, W)# B N(HW) C

        # B N(HW) C --> B C H W
        B, N, _C = x1.shape
        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()

        #前馈网络ffn

        # Gated fusion（Softmax门控） fft前馈网络

        gate_input = torch.cat([x1, x2], dim=1)  # B, 2C, H, W
        gate_input = self.fft(gate_input)
        # gate = torch.softmax(self.gate_conv(gate_input), dim=1)  # B, 2, H, W  softmax门控机制
        # gate1 = gate[:, 0:1, :, :]
        # gate2 = gate[:, 1:2, :, :]
        gate = torch.sigmoid(self.gate_conv(gate_input))  # B, 2, H, W    sigmoid门控机制
        gate1 = gate
        gate2 = 1-gate        

        gated_x1 = gate1 * x1
        gated_x2 = gate2 * x2
        residual = x[0] + x[1]
        fuse = self.norm(gated_x1 + gated_x2+ residual) # 残差连接，增强原始特征表达,训练更稳定
        return fuse


class GACF_LEG(nn.Module):
    def __init__(self, c1,c2,stage,  reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(GACF_LEG, self).__init__()
        self.att1 = CombinedAttention(c1)
        self.att2 = CombinedAttention(c1)
        # stage = stage
        self.att_leg1 = LEG(dim=c1, stage=stage)
        self.att_leg2 = LEG(dim=c1, stage=stage)

        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.channel_emb = ChannelEmbed(in_channels=c1 * 2, out_channels=c1, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)
        self.gate_conv = nn.Conv2d(c1 * 2, 2, kernel_size=1)  # softmax 门控机制，二者进行竞争式加权

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # #  Combined Attention:CBAM + MSA
        # x1 = self.att1(x[0])
        # x2 = self.att2(x[1])


        x1 = self.att_leg1(x[0])
        x2 = self.att_leg1(x[1])

        #  Cross Attention
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B C H W
        B, N, _C = x1.shape
        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W)


        # Gated fusion（Softmax门控）
        gate_input = torch.cat([x1, x2], dim=1)  # B, 2C, H, W
        gate = torch.softmax(self.gate_conv(gate_input), dim=1)  # B, 2, H, W

        gate1 = gate[:, 0:1, :, :]
        gate2 = gate[:, 1:2, :, :]

        gated_x1 = gate1 * x1
        gated_x2 = gate2 * x2
        # fuse = gated_x1 + gated_x2 + (x[0] + x[1]) # 残差连接，增强原始特征表达
        fuse = gated_x1 + gated_x2
        return fuse

###########################   8.21   ##############################
class BCFM_EGA(nn.Module):
    def __init__(self, c1,c2,stage,  reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(BCFM_EGA, self).__init__()
        # stage = stage
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.apply(self._init_weights)
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)  # softmax 门控机制，二者进行竞争式加权

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        #  Cross Attention
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B C H W
        B, N, _C = x1.shape
        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W)


        # Gated fusion
        gate_input = torch.cat([x1, x2], dim=1)  # B, 2C, H, W
        gate = torch.sigmoid(self.gate_conv(gate_input))  # B,1,H,W
        gate1 = gate
        gate2 = 1-gate

        gated_x1 = gate1 * x1
        gated_x2 = gate2 * x2
        fuse = gated_x1 + gated_x2 + (x[0] + x[1]) # 残差连接，增强原始特征表达
        # fuse = gated_x1 + gated_x2
        return fuse

#########################################
class FeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )

    def forward(self, x):
        return self.ffn(x)

class GACF_fft(nn.Module):
    def __init__(self, c1,c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(GACF_fft, self).__init__()
        self.att1 = CombinedAttention(c1)
        self.att2 = CombinedAttention(c1)

        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        self.fft= FeedForwardNetwork(in_channels=c1 * 2, out_channels=c1*2, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

        self.norm = norm_layer(c2)

        # self.gate_conv = nn.Sequential(
        #     nn.Conv2d(c1 * 2, c1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # softmax 门控机制，二者进行竞争式加权
        self.gate_conv = nn.Conv2d(c1 * 2, 2, kernel_size=1)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #  Combined Attention:CBAM + MSA
        x1 = self.att1(x[0])
        x2 = self.att2(x[1])

        #  Cross Attention
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2, H, W)# B N(HW) C

        # B N(HW) C --> B C H W
        B, N, _C = x1.shape
        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()

        #前馈网络ffn

        # Gated fusion（Softmax门控） fft前馈网络

        gate_input = torch.cat([x1, x2], dim=1)  # B, 2C, H, W
        gate_input = self.fft(gate_input)
        gate = torch.softmax(self.gate_conv(gate_input), dim=1)  # B, 2, H, W

        gate1 = gate[:, 0:1, :, :]
        gate2 = gate[:, 1:2, :, :]

        gated_x1 = gate1 * x1
        gated_x2 = gate2 * x2
        residual = x[0] + x[1]
        fuse = self.norm(gated_x1 + gated_x2+ residual) # 残差连接，增强原始特征表达,训练更稳定
        return fuse

class BCFM_EGA_ffn(nn.Module):
    def __init__(self, c1,c2,stage,  reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(BCFM_EGA_ffn, self).__init__()
        # stage = stage
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)
        
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.apply(self._init_weights)
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)  
        
        self.fft= FeedForwardNetwork(in_channels=c1, out_channels=c1, reduction=reduction,
                                        norm_layer=norm_layer)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        #  Cross Attention
        B, C, H, W = x1.shape
        # B C (HW)->B N(HW) C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B C H W
        B, N, _C = x1.shape
        x1 = x1.permute(0, 2, 1).reshape(B, _C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, _C, H, W)


        # Gated fusion
        gate_input = torch.cat([x1, x2], dim=1)  # B, 2C, H, W
        gate = torch.sigmoid(self.gate_conv(gate_input))  # B,1,H,W
        gate1 = gate
        gate2 = 1-gate
        gated_x1 = gate1 * x1
        gated_x2 = gate2 * x2
        fuse = gated_x1 + gated_x2 + (x[0] + x[1]) # 残差连接，增强原始特征表达
        
        out = self.fft(fuse)
        return out

################################################  2025.9.2  ##########################################################
from timm.models.layers import trunc_normal_
class CMFM(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        # === 融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_abla_1(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_abla_1, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = x[0]
        x2 = x[1]

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)

        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_abla_2(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_abla_2, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_abla_3(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_abla_3, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()
        
        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)

        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Sigmoid 门控 ===
        gate =0.5
        
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse


    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()


        # === Sigmoid 门控 ===
        gate = 0.5
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse


################################################  2025.9.4 ablation  ##########################################################    
class CMFM_ega(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_ega, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])
        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()


        gate = 0.5
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        
        return fuse

class CMFM_cross(nn.Module):
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_cross, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=1)
        self.att_ega2 = EGA(dim=c1, stage=1)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = x[0]
        x2 = x[1]


        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)

        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        
        gate = 0.5
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        
        return fuse

class CMFM_gate(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_gate, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = x[0]
        x2 = x[1]

        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        
        return fuse

class CMFM_ega_cross(nn.Module):
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_ega_cross, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=1)
        self.att_ega2 = EGA(dim=c1, stage=1)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])
        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        
        gate = 0.5
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        return fuse

class CMFM_ega_gate(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_ega_gate, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])
        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        
        return fuse

class CMFM_cross_gate(nn.Module):
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_cross_gate, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = x[0]
        x2 = x[1]
        
        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()


        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        
        return fuse


################################################  2025.9.5 improve  ##########################################################  
class CMFM_imp1(nn.Module):
    # α/β 初值 0。
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_imp1, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.0))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)

        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_imp2(nn.Module):
    # α/β 初值 0   gate改为双通道softmax-gate
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d,gate_tau=1.0):
        
        super(CMFM_imp2, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # # === sigmoid 门控 ===
        # # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        # self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === softmax 门控 ===
        # 软分配 输出 2C，分别对应两模态的权重
        self.gate_conv = nn.Conv2d(c1 * 2, c1 * 2, kernel_size=1, bias=True)
        self.gate_tau = gate_tau  # 温度系数 τ（标量），默认 1.0

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.0))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)

        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        
        # === Softmax 软分配门控 ===
        gate_logits = self.gate_conv(torch.cat([x1, x2], dim=1))   # [B, 2C, H, W]
        gate_logits = gate_logits.view(B, 2, C, H, W)              # [B, 2, C, H, W]
        if self.gate_tau != 1.0:
            gate_logits = gate_logits / self.gate_tau
        gate = torch.softmax(gate_logits, dim=1)                   # 在“模态维=1”做softmax
        g1, g2 = gate[:, 0], gate[:, 1]                            # [B, C, H, W] 各自权重

        gated = g1 * x1 + g2 * x2                                  # 软分配融合

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_imp3(nn.Module):
    # 只有ega 且rgb scharr边缘合适    sar gaussian合适
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_imp3, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=0)
        self.att_ega2 = EGA(dim=c1, stage=1)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        
        gate = 0.5      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        
        return fuse

class CMFM_imp4(nn.Module):
    # 只有ega 且rgb scharr边缘合适    sar gaussian合适  边缘增强改为边缘调制
    
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_imp4, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=0)  # for RGB
        self.att_ega2 = EGA(dim=c1, stage=1)  # for SAR
        
        # === 边缘调制头（新增）：把 EGA 输出映射为逐通道缩放图 ===
        self.mod_head1 = nn.Conv2d(c1, c1, kernel_size=1, bias=True)  # RGB 调制
        self.mod_head2 = nn.Conv2d(c1, c1, kernel_size=1, bias=True)  # SAR 调制
        # 可学习限幅系数（建议：RGB 略强，SAR 更弱）
        self.delta_rgb = nn.Parameter(torch.tensor(0.10))  # 0.08~0.15 之间微调
        self.delta_sar = nn.Parameter(torch.tensor(0.05))  # 0.03~0.08 之间微调


        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
        # 调制头初始化为“几乎不调制”（很重要，提升稳定性）
        nn.init.zeros_(self.mod_head1.weight); nn.init.zeros_(self.mod_head1.bias)
        nn.init.zeros_(self.mod_head2.weight); nn.init.zeros_(self.mod_head2.bias)

    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x_rgb = x[0]
        x_sar = x[1]

        # === 边缘调制（位置不变：在 Cross/Gate 之前）===
        # 1) 用 EGA 提取引导特征（注意：此处仍按你的 EGA 实现，RGB=Scharr, SAR=Gaussian）
        e1 = self.att_ega1(x_rgb)      # [B, C, H, W]
        e2 = self.att_ega2(x_sar)      # [B, C, H, W]

        # 2) 生成逐通道缩放图，tanh 限制在 [-1,1]
        scale1 = torch.tanh(self.mod_head1(e1))  # [B, C, H, W]
        scale2 = torch.tanh(self.mod_head2(e2))  # [B, C, H, W]

        # 3) 小幅乘法调制：F' = F * (1 + δ * scale)，δ 很小以稳为主
        # 可选稳态：若训练早期不稳，可把 e1/e2 用 detach 的版本（仅作先验，不回传梯度）
        # e1 = self.att_ega1(x_rgb.detach()); e2 = self.att_ega2(x_sar.detach())
        x1 = x_rgb * (1.0 + torch.clamp(self.delta_rgb, 0.0, 0.3) * scale1)
        x2 = x_sar * (1.0 + torch.clamp(self.delta_sar, 0.0, 0.3) * scale2)

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        
        gate = 0.5      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        
        return fuse


################################################  2025.9.5 order sensitivity  #########################################################
class CMFM_order1(nn.Module):
    # ega--cross-gate
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_order1, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)

        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Sigmoid 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_order2(nn.Module):
    # ega--gate--cross
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_order2, self).__init__()
        # === EGA:两分支原始注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Gate: 互补门控（sigmoid 与 1-sigmoid） ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数（稳定训练 & 保底信息通路） ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        # 参数初始化
        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_rgb, x_ir = x[0], x[1]
        
        # === 1) EGA ===
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])
        # === 局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === 2) Gate（互补门控）===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        x1 = gate * x1
        x2 = (1.0 - gate) * x2
        
        # === 3) Cross（跨模态交互）===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

         # === 4) 融合 + 残差（α/β）===
        fuse = x1 + x2 + self.alpha * x_rgb + self.beta * x_ir

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_order3(nn.Module):
    # cross--ega--gate
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_order3, self).__init__()
        
         # === EGA: 两分支显著性/边缘引导 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)
        # === 局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

         # === Cross: 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Gate: 互补门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_rgb, x_sar = x[0], x[1]
        
        # === 1) Cross ===
        B, C, H, W = x_rgb.shape
        x1_t = x_rgb.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x_sar.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        
        # === 2) EGA ===
        x1 = self.att_ega1(x1)
        x2 = self.att_ega2(x2)
        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === 3) Gate（互补门控）===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === D：可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x_rgb + self.beta * x_sar

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_order4(nn.Module):
    # cross--gate--ega
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_order4, self).__init__()
        # === EGA: 两分支边缘/显著性引导 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === Cross: 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Gate: 互补门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x_rgb, x_sar = x[0], x[1]
        
        # === 1) Cross ===
        B, C, H, W = x_rgb.shape
        x1_t = x_rgb.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x_sar.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x1 = self.att_ega1(x[0])
        x2 = self.att_ega2(x[1])

        # === 2) Gate（互补门控）===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        x1 = gate * x1
        x2 = (1.0 - gate) * x2
        
        # === 3) EGA（细化）===
        x1 = self.att_ega1(x1)
        x2 = self.att_ega2(x2)
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === D：可学习残差缩放 + 融合 ===
        fuse = x1 + x2 + self.alpha * x_rgb + self.beta * x_sar

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_order5(nn.Module):
    # gate--ega--cross
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_order5, self).__init__()
        
        # === EGA: 显著性/边缘引导 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)
        # === 局部先验（增强位置/局部性） ===

        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === Cross: 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Gate: 互补门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_rgb, x_sar = x[0], x[1]
        
        # === 1) Gate ===
        gate_input = torch.cat([x_rgb, x_sar], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        x1 = gate * x_rgb
        x2 = (1.0 - gate) * x_sar
        
        # === 2) EGA ===
        x1 = self.att_ega1(x1)
        x2 = self.att_ega2(x2)
        # === 局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === 3) Cross ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 4) 融合 + 残差 ===
        fuse = x1 + x2 + self.alpha * x_rgb + self.beta * x_sar

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_order6(nn.Module):
    # gate--crosss--ega
    def __init__(self, c1, c2, stage, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_order6, self).__init__()
        
        # === EGA: 边缘/显著性引导 ===
        self.att_ega1 = EGA(dim=c1, stage=stage)
        self.att_ega2 = EGA(dim=c1, stage=stage)

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === Cross: 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Gate: 互补门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_rgb, x_sar = x[0], x[1]
        
        # === 1) Gate ===
        gate_input = torch.cat([x_rgb, x_sar], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        x1 = gate * x_rgb
        x2 = (1.0 - gate) * x_sar
        
         # === 2) Cross ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        
        # === 3) EGA ===
        x1 = self.att_ega1(x1)
        x2 = self.att_ega2(x2)
        # === C：局部先验（增强位置/局部性） ===
        x1 = self.pre_local(x1).contiguous()
        x2 = self.pre_local(x2).contiguous()

        # === 4) 融合 + 残差 ===
        fuse = x1 + x2 + self.alpha * x_rgb + self.beta * x_sar

        # === D：融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

################################################  2025.9.7 Improve: LEGA: Learnable Edge-Guided Attention ######################
class LEGA(nn.Module):
    """
    Learnable Edge-Guided Attention (LEGA)
    改进点：
    1. 可学习系数 gamma 控制整体增强强度，避免训练初期过拟合或过强干预；
    2. alpha/beta 控制边缘增强分量与原始残差的平衡；
    3. 保留 Scharr/ Gaussian 作为引导特征，可按 stage/mode 区分模态；
    4. 保持与原 EGA 一致的接口，便于替换。
    """
    def __init__(self,
                 dim,              # 输入通道数
                 stage,            # 决定使用 Scharr 或 Gaussian
                 drop_path=0.1,    # DropPath 概率
                 act_layer=nn.ReLU # 激活函数
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.BatchNorm2d(dim)

        # 引导特征：Scharr or Gaussian
        if stage == 0:
            self.att_source = Scharr(dim, act_layer)
        else:
            self.att_source = Gaussian(dim, size=5, sigma=1.0, act_layer=act_layer)
            
        # 空间卷积增强分支
        self.conv2d = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            act_layer()
        )

        # CBAM 注意力模块
        self.cbam = CBAM(channels=dim)
        
        # === 新增：可学习系数 ===
        self.gamma = nn.Parameter(torch.tensor(0.0))  # 控制整体强度
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 控制增强分量
        self.beta  = nn.Parameter(torch.tensor(1.0))  # 控制残差分量

        
    def forward(self, x):
        # 1. 计算引导特征
        att = self.att_source(x)

        # 2. 加权融合：控制乘法增强和残差保持
        att = self.alpha * (x * att) + self.beta * x
        att = self.conv2d(att)

        # 3. CBAM 权重
        wei = torch.sigmoid(self.cbam(att))

        # 4. 最终输出：受控残差
        out = x + self.gamma * self.norm(self.drop_path(att * wei))
        return out

class CMFM_lega1(nn.Module):
    # RGB：lega0 scharr
    # SAR：lega0 scharr

    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega1, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=0)  # RGB -> Scharr
        self.att_lega2 = LEGA(dim=c1, stage=0)  # SAR -> Scharr

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega2(nn.Module):
    # RGB：lega0 scharr
    # SAR：lega1 gaussian

    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega2, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=0)  # RGB -> Scharr
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        # === 融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega3(nn.Module):
    # RGB：lega1 gaussian
    # SAR：lega0 scharr

    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega3, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=0)  # SAR -> Scharr

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega4(nn.Module):
    # RGB：lega1 gaussian
    # SAR：lega1 gaussian

    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega4, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega5(nn.Module):
    # RGB：lega1 gaussian
    # SAR：lega1 gaussian
    # 无局部先验

    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega5, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        # 逐通道门控：两支竞争但采用 sigmoid + (1 - sigmoid) 的互补方式
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.att_lega1(x[0])
        x2 = self.att_lega2(x[1])


        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        # === 融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega6(nn.Module):
    # RGB：lega1 gaussian
    # SAR：CBAM
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega6, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        self.att_cbam1 = CBAM(channels=c1)  
        self.att_cbam2 = CBAM(channels=c1) 
        
        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = x[0]
        x2 = x[1]
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_cbam2(x[1])).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega7(nn.Module):
    # RGB：CBAM
    # SAR：lega1 gaussian
    
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega7, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        self.att_cbam1 = CBAM(channels=c1)  
        self.att_cbam2 = CBAM(channels=c1)  
        
        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_cbam1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        # === 2: Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === 3:Gate 门控 ===
        gate_input = torch.cat([x1, x2], dim=1)              # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_conv(gate_input))      # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse


################################################  2025.9.7 G-Ablation: Gate 内部机制的对比消融 ####################################
class CMFM_gate2(nn.Module):
    # # G1: Softmax 门控 —— 使用两路 logits 经过 softmax 得到严格互补的 (g1,g2)，避免 Sigmoid 半开半闭
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_gate2, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=1)
        self.att_ega2 = EGA(dim=c1, stage=1)
        # === C：局部先验（增强位置/局部性） ===
        # 轻量深度可分离卷积：引入局部性与位置先验，几乎不增 FLOPs
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === G1: Softmax 门控（通道级严格互补） ===
        # 原来是 out_channels=c1（sigmoid）；现在改为 2*c1（两路 logits）
        self.gate_conv = nn.Conv2d(c1 * 2, c1 *2, kernel_size=1, bias = True)
        nn.init.zeros_(self.gate_conv.bias) # 建议：初始化 bias=0，让初始 g1≈g2≈0.5（更平衡）
        
        # === D：可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === D：融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 若此处再次置零会覆盖 gate_conv.bias=0，但不影响正确性
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        
        # === G1: Softmax 门控 ===
        
        B, C, H, W = x1.shape
        gate_input = torch.cat([x1, x2], dim=1)                 # [B, 2C, H, W]
        logits = self.gate_conv(gate_input)                     # [B, 2C, H, W]
        logits = logits.view(B, 2, C, H, W)                     # [B, 2, C, H, W]
        g = torch.softmax(logits, dim=1)                        # 沿分支维度softmax
        g1, g2 = g[:, 0], g[:, 1]                               # [B, C, H, W] x2

        gated_x1 = g1 * x1
        gated_x2 = g2 * x2
        
        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        
        # === 融合后轻量整形（统计对齐） ===
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_gate3(nn.Module):
    # G3: 通道 × 空间双门控 (Sigmoid 版)
    def __init__(self, c1, c2,  reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_gate3, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=1)
        self.att_ega2 = EGA(dim=c1, stage=1)

        # === 局部先验 ===
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === 通道门控 (SE结构) ===
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(2 * c1, c1 // reduction, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c1 // reduction, c1, 1, bias=True)

        # === 空间门控 ===
        self.spatial_conv = nn.Conv2d(2 * c1, 1, kernel_size=7, padding=3, bias=False)

        # === 残差缩放系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

        # === 融合后轻量整形 ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        
        B, C, H, W = x1.shape
        feat_cat = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]

        # === 通道注意力 ===
        ch = self.avgpool(feat_cat)                 # [B, 2C, 1, 1]
        ch = self.fc2(self.act(self.fc1(ch)))       # [B, C, 1, 1]
        g_c = torch.sigmoid(ch)                     # [B, C, 1, 1]

        # === 空间注意力 ===
        sp = self.spatial_conv(feat_cat)            # [B, 1, H, W]
        g_s = torch.sigmoid(sp)                     # [B, 1, H, W]

        # === 通道 × 空间联合权重 ===
        g = g_c * g_s                               # [B, C, H, W]

        # === 应用到两模态 ===
        g1 = g
        g2 = 1.0 - g
        gated_x1 = g1 * x1
        gated_x2 = g2 * x2

        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_gate4(nn.Module):
    # G4: 通道 × 空间双门控 (Softmax 版)
    def __init__(self, c1, c2,  reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_gate4, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=4)
        self.att_ega2 = EGA(dim=c1, stage=4)

        # === 局部先验 ===
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === 通道门控 (SE) ===
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(2 * c1, c1 // reduction, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c1 // reduction, c1, 1, bias=True)

        # === 空间门控 ===
        self.spatial_conv = nn.Conv2d(2 * c1, 1, kernel_size=7, padding=3, bias=False)

        # === 残差缩放系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

        # === 融合后轻量整形 ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        B, C, H, W = x1.shape
        feat_cat = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]

        # === 通道注意力 ===
        ch = self.avgpool(feat_cat)                 # [B, 2C, 1, 1]
        ch = self.fc2(self.act(self.fc1(ch)))       # [B, C, 1, 1]
        g_c = torch.sigmoid(ch)                     # [B, C, 1, 1]

        # === 空间注意力 ===
        sp = self.spatial_conv(feat_cat)            # [B, 1, H, W]
        g_s = torch.sigmoid(sp)                     # [B, 1, H, W]

        # === 通道 × 空间联合分数 ===
        score = g_c * g_s                           # [B, C, H, W]

        # === Softmax 互补 ===
        logits = torch.stack([score, 1.0 - score], dim=1)  # [B, 2, C, H, W]
        g = torch.softmax(logits, dim=1)                   # 沿模态维度归一化
        g1, g2 = g[:, 0], g[:, 1]

        gated_x1 = g1 * x1
        gated_x2 = g2 * x2

        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_cross_gate2(nn.Module):
    # G3: Cross-Context + Sigmoid —— 先用 Cross 得到上下文摘要，再用 sigmoid 门控互补 (g, 1-g)
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d, ctx_c=64):
        super(CMFM_cross_gate2, self).__init__()
        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Cross-Context (全局摘要) + Sigmoid 门控 ===
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)           # GAP: [B,*,H,W] -> [B,*,1,1]
        self.ctx_proj = nn.Conv2d(2 * c1, ctx_c, 1, bias=False)
        
        self.gate_conv_sig = nn.Conv2d(2 * c1 + ctx_c, c1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_conv_sig.bias)           # 初始 g≈0.5，避免偏置

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1, x2 = x[0], x[1]
        x1_f = x1
        x2_f = x2
        
        # === Cross 交互得到对齐后的特征 ===
        B, C, H, W = x1_f.shape
        x1_t = x1_f.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2_f.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1c = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Cross-Context 上下文摘要 ===
        ctx = torch.cat([x1c, x2c], dim=1)        # [B,2C,H,W]
        ctx = self.ctx_proj(self.ctx_pool(ctx))             # [B,ctx_c,1,1]
        
        
        ctx_expand = ctx.expand(-1, -1, H, W)               # [B,ctx_c,H,W]
        
       # === Sigmoid 门控（带上下文）===
        gate_in = torch.cat([x1_f, x2_f, ctx_expand], dim=1)    # [B,2C+ctx_c,H,W]
        g = torch.sigmoid(self.gate_conv_sig(gate_in))      # [B,C,H,W]
        g1, g2 = g, (1.0 - g)

        gated_x1 = g1 * x1_f
        gated_x2 = g2 * x2_f
        
        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_cross_gate3(nn.Module):
    # G3: Cross-Context + Sigmoid —— 先用 Cross 得到上下文摘要，再用 sigmoid 门控互补 (g, 1-g)
    def __init__(self, c1, c2, reduction=16, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d, ctx_c=64):
        super(CMFM_cross_gate3, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = EGA(dim=c1, stage=1)
        self.att_ega2 = EGA(dim=c1, stage=1)
        self.pre_local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Cross-Context (全局摘要) + Softmax 门控 ===
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)           # GAP: [B,*,H,W] -> [B,*,1,1]
        self.ctx_proj = nn.Conv2d(2 * c1, ctx_c, 1, bias=False)
        
        # 输出 2*C 通道作为两模态 logits；softmax 严格互补
        self.gate_conv = nn.Conv2d(2 * c1 + ctx_c, 2 * c1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_conv.bias)                    # 初始 g1≈g2≈0.5，更平衡

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1_in, x2_in = x[0], x[1]
        x1_f, x2_f = x1_in, x2_in
        
        # === Cross 交互得到对齐后的特征 ===
        B, C, H, W = x1_f.shape
        x1_t = x1_f.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2_f.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1c = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Cross-Context 上下文摘要 ===
        ctx = torch.cat([x1c, x2c], dim=1)        # [B,2C,H,W]
        ctx = self.ctx_proj(self.ctx_pool(ctx))             # [B,ctx_c,1,1]
        ctx_expand = ctx.expand(-1, -1, H, W)               # [B,ctx_c,H,W]
        
       # === Softmax 门控（带上下文）===
        gate_in = torch.cat([x1_f, x2_f, ctx_expand], dim=1)  # [B,2C+ctx_c,H,W]
        logits = self.gate_conv(gate_in)                      # [B,2C,H,W]
        logits = logits.view(B, 2, C, H, W)                   # [B,2,C,H,W]
        g = torch.softmax(logits, dim=1)                      # 严格互补
        g1, g2 = g[:, 0], g[:, 1]                             # [B,C,H,W]

        gated_x1 = g1 * x1_f
        gated_x2 = g2 * x2_f

        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x1_in + self.beta * x2_in
        fuse = self.post_fuse(fuse)
        return fuse

################################################  2025.9.7 cross_gate + lega: lega模块的增益作用消融   #################################

class CMFM_lega_cross_gate2(nn.Module):
    # FiLM + Cross-Context + Sigmoid 
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d, ctx_c=64):
        super(CMFM_lega_cross_gate2, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = LEGA(dim=c1, stage=1)
        self.att_ega2 = LEGA(dim=c1, stage=1)
        
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        # === 跨模态交互 ===
        self.cross = Cross(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Cross-Context (全局摘要) + Sigmoid 门控 ===
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)           # GAP: [B,*,H,W] -> [B,*,1,1]
        self.ctx_proj = nn.Conv2d(2 * c1, ctx_c, 1, bias=False)
        
        self.gate_conv_sig = nn.Conv2d(2 * c1 + ctx_c, c1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_conv_sig.bias)           # 初始 g≈0.5，避免偏置

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # 显著图头（FiLM 所需）
        self.mask_head1 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.mask_head2 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        
        # === EGA + 局部先验 ===
       # === 显著图（供 FiLM 使用） ===
        x1_f = self.pre_local1(self.att_ega1(x1))
        x2_f = self.pre_local2(self.att_ega2(x2))
        m1 = self.mask_head1(x1_f)  # [B,1,H,W]
        m2 = self.mask_head2(x2_f)
        
        # === Cross 交互得到对齐后的特征 ===
        B, C, H, W = x1_f.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1c_t, x2c_t = self.cross(x1_t, x2_t, H, W, m1=m1, m2=m2, detach_mask=False)
        x1c = x1c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Cross-Context 上下文摘要 ===
        ctx = torch.cat([x1c, x2c], dim=1)        # [B,2C,H,W]
        ctx = self.ctx_proj(self.ctx_pool(ctx))             # [B,ctx_c,1,1]
        ctx_expand = ctx.expand(-1, -1, H, W)               # [B,ctx_c,H,W]
        
       # === Sigmoid 门控（带上下文）===
        gate_in = torch.cat([x1, x2, ctx_expand], dim=1)    # [B,2C+ctx_c,H,W]
        g = torch.sigmoid(self.gate_conv_sig(gate_in))      # [B,C,H,W]
        g1, g2 = g, (1.0 - g)
        gated_x1 = g1 * x1
        gated_x2 = g2 * x2
        
        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega_cross_gate3(nn.Module):
    # KV_pooling + Cross-Context + Sigmoid 
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d, ctx_c=64):
        super(CMFM_lega_cross_gate3, self).__init__()
        # === 原始分支注意力 ===
        self.att_ega1 = LEGA(dim=c1, stage=1)
        self.att_ega2 = LEGA(dim=c1, stage=1)

        # === 局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = Cross_kvpool(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Cross-Context (全局摘要) + Sigmoid 门控 ===
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)           # GAP: [B,*,H,W] -> [B,*,1,1]
        self.ctx_proj = nn.Conv2d(2 * c1, ctx_c, 1, bias=False)
        
        self.gate_conv_sig = nn.Conv2d(2 * c1 + ctx_c, c1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_conv_sig.bias)           # 初始 g≈0.5，避免偏置

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # 显著图头
        self.mask_head1 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.mask_head2 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # === EGA + 局部先验 ===
        # === 显著图===
        x1_f = self.pre_local1(self.att_ega1(x1))
        x2_f = self.pre_local2(self.att_ega2(x2))
        m1 = self.mask_head1(x1_f)  # [B,1,H,W]
        m2 = self.mask_head2(x2_f)
        
        # === Cross 交互得到对齐后的特征 ===
        B, C, H, W = x1_f.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1c_t, x2c_t = self.cross(x1_t, x2_t, H, W, m1=m1, m2=m2, detach_mask=False)
        x1c = x1c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Cross-Context 上下文摘要 ===
        ctx = torch.cat([x1c, x2c], dim=1)        # [B,2C,H,W]
        ctx = self.ctx_proj(self.ctx_pool(ctx))             # [B,ctx_c,1,1]
        ctx_expand = ctx.expand(-1, -1, H, W)               # [B,ctx_c,H,W]
        
       # === Sigmoid 门控（带上下文）===
        gate_in = torch.cat([x1, x2, ctx_expand], dim=1)    # [B,2C+ctx_c,H,W]
        g = torch.sigmoid(self.gate_conv_sig(gate_in))      # [B,C,H,W]
        g1, g2 = g, (1.0 - g)
        gated_x1 = g1 * x1
        gated_x2 = g2 * x2
        
        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega_cross_gate4(nn.Module):
    # FiLM + KV_pooling + Cross-Context + Sigmoid 
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d, ctx_c=64):
        super(CMFM_lega_cross_gate4, self).__init__()

        self.att_ega1 = LEGA(dim=c1, stage=1)
        self.att_ega2 = LEGA(dim=c1, stage=1)

        # === 局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = Cross_FiLM_kvpool(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === Cross-Context (全局摘要) + Sigmoid 门控 ===
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)           # GAP: [B,*,H,W] -> [B,*,1,1]
        self.ctx_proj = nn.Conv2d(2 * c1, ctx_c, 1, bias=False)
        
        self.gate_conv_sig = nn.Conv2d(2 * c1 + ctx_c, c1, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_conv_sig.bias)           # 初始 g≈0.5，避免偏置

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # 显著图头
        self.mask_head1 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.mask_head2 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # === EGA + 局部先验 ===
        # === 显著图===
        x1_f = self.pre_local1(self.att_ega1(x1))
        x2_f = self.pre_local2(self.att_ega2(x2))
        m1 = self.mask_head1(x1_f)  # [B,1,H,W]
        m2 = self.mask_head2(x2_f)
        
        # === Cross 交互得到对齐后的特征 ===
        B, C, H, W = x1_f.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1c_t, x2c_t = self.cross(x1_t, x2_t, H, W, m1=m1, m2=m2, detach_mask=False)
        x1c = x1c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        # === Cross-Context 上下文摘要 ===
        ctx = torch.cat([x1c, x2c], dim=1)        # [B,2C,H,W]
        ctx = self.ctx_proj(self.ctx_pool(ctx))             # [B,ctx_c,1,1]
        ctx_expand = ctx.expand(-1, -1, H, W)               # [B,ctx_c,H,W]
        
       # === Sigmoid 门控（带上下文）===
        gate_in = torch.cat([x1, x2, ctx_expand], dim=1)    # [B,2C+ctx_c,H,W]
        g = torch.sigmoid(self.gate_conv_sig(gate_in))      # [B,C,H,W]
        g1, g2 = g, (1.0 - g)
        gated_x1 = g1 * x1
        gated_x2 = g2 * x2
        
        # === 残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

################################################  2025.9.7 C-Ablation: Cross 机制的对比消融 ####################################
# 让 EGA 真正“指导” Cross，并修好收敛细节

class CrossAtt(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        super(CrossAtt, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm1 = nn.LayerNorm(dim)

            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        # B num_heads N C//num_heads
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # B C//num_head N num_heads -> B N C//num_heads num_heads -> B C H W
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            # B C H W -> B C H/R W/R -> B C HW/R² -> B HW/R² C
            x_1 = self.sr1(x_1).reshape(B, C, -1).permute(0, 2, 1)
            x_1 = self.norm1(x_1)
            # B HW/R² C -> B HW/R² 2C -> B HW/R² 2 num_heads C//num_heads -> 2 B num_heads HW/R² C//num_heads
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2).reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # B num_heads HW/R² C//num_heads
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        # B num_heads N HW/R²
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        # B num_heads N C//num_heads -> B N num_heads C//num_heads -> B N C
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        return main_out, aux_out

class Cross(nn.Module):
    # 加入FiLM（Feature-wise Linear Modulation） 是对特征做逐元素/逐通道的线性调制
    def __init__(self, dim, reduction=1, num_heads=2, sr_ratio=4, norm_layer=nn.LayerNorm):
        super(Cross, self).__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAtt(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # FiLM 强度（可学习标量）
        self.film_lambda1 = nn.Parameter(torch.tensor(0.5))
        self.film_lambda2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2, H, W, m1=None, m2=None, detach_mask=False):
        """
        x1, x2: [B, N, C]
        m1, m2: 可选显著图 [B,1,H,W] 或可展平成 [B,N,1]
        """
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1) # z1: [B,N,Cr]
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        
        # ---- B1: FiLM 调制（在降维后的 z1/z2 上）----
        if m1 is not None:
            m1_flat = m1.flatten(2).transpose(1, 2)  # [B,N,1]
            if detach_mask:
                m1_flat = m1_flat.detach()
            z1 = z1 * (1 + self.film_lambda1 * m1_flat)

        if m2 is not None:
            m2_flat = m2.flatten(2).transpose(1, 2)  # [B,N,1]
            if detach_mask:
                m2_flat = m2_flat.detach()
            z2 = z2 * (1 + self.film_lambda2 * m2_flat)
        
        # Cross   
        c1, c2 = self.cross_attn(z1, z2, H, W)
        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out = self.norm2(x2 + self.end_proj2(y2))
        return main_out, aux_out

class CMFM_ega_cross2(nn.Module):
    def __init__(self, c1, c2,  reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        
        super(CMFM_ega_cross2, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)
        self.att_lega2 = LEGA(dim=c1, stage=1)
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = Cross(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        # 从特征预测显著图（供 FiLM 使用）
        self.mask_head1 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.mask_head2 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())


        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # EGA + 位置先验
        x1_f = self.pre_local1(self.att_lega1(x1))
        x2_f = self.pre_local2(self.att_lega2(x2)) 

        m1 = self.mask_head1(x1_f)  # [B,1,H,W]
        m2 = self.mask_head2(x2_f)
        
        #  Cross
        B, C, H, W = x1_f.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1c_t, x2c_t = self.cross(x1_t, x2_t, H, W, m1=m1, m2=m2, detach_mask=False)
        x1c = x1c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        gate = 0.5
        gated_x1 = gate * x1c
        gated_x2 = (1.0 - gate) * x2c
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CrossAtt_kvpool(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None,
                 wp_eps: float = 1e-6, den_thr: float = 1e-4, use_fallback: bool = True):
        super(CrossAtt_kvpool, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 作为 fallback 的原下采样路径
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio,
                                 padding=sr_ratio // 2, groups=dim)
            self.norm1 = nn.LayerNorm(dim)
            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio,
                                 padding=sr_ratio // 2, groups=dim)
            self.norm2 = nn.LayerNorm(dim)

        # 安全池化参数
        self.wp_eps = float(wp_eps)      # 分母 clamp 的最小值
        self.den_thr = float(den_thr)    # 回退阈值：den<=den_thr 时回退 avg_pool(x)
        self.use_fallback = bool(use_fallback)
    @staticmethod
    def _mask_to_b1hw(m, B, H, W):
        """把 mask 统一成 [B,1,H,W]；支持 [B,1,H,W]/[B,C,H,W]/[B,N]/[B,N,1]"""
        if m is None:
            return None
        if m.dim() == 4:  # [B,C,H,W] or [B,1,H,W]
            return m if m.size(1) == 1 else m.mean(1, keepdim=True)
        N = H * W
        if m.dim() == 3:  # [B,N,1]
            m = m.squeeze(-1)
        if m.dim() == 2:  # [B,N]
            return m.view(B, 1, H, W)
        raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")
    
    def _weighted_pool(self, x, m, sr):
        """
        安全加权池化：
            out = avg_pool(x*m) / clamp_min(avg_pool(m), wp_eps)
            若 avg_pool(m) <= den_thr，则回退到 avg_pool(x)
        x: [B,C,H,W], m: [B,1,H,W]
        """
        if sr <= 1:
            return x
        num = F.avg_pool2d(x * m, kernel_size=sr, stride=sr)
        den = F.avg_pool2d(m,     kernel_size=sr, stride=sr)   # [B,1,h,w]
        out = num / den.clamp_min(self.wp_eps)
        if self.use_fallback:
            avg_x = F.avg_pool2d(x, kernel_size=sr, stride=sr)
            out = torch.where(den > self.den_thr, out, avg_x)
        return out
    
    def forward(self, x1, x2, H, W, m1=None, m2=None, detach_mask=False):
        """
        x1, x2: [B, N, C]，N=H*W；维度 C 是降维后的 dim//reduction
        m1, m2: 可选显著图（来自 EGA），推荐 [B,1,H,W]
        """
        B, N, C = x1.shape
        # B num_heads N C//num_heads
        # Q 不变
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # 规范化 mask
        m1 = self._mask_to_b1hw(m1, B, H, W)
        m2 = self._mask_to_b1hw(m2, B, H, W)
        if detach_mask:
            if m1 is not None: m1 = m1.detach()
            if m2 is not None: m2 = m2.detach()
        
        # K/V 的下采样：优先用“加权池化”，否则回退到原下采样
        if self.sr_ratio > 1 and (m1 is not None) and (m2 is not None):
            x1_img = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x2_img = x2.permute(0, 2, 1).reshape(B, C, H, W)

            x1_ds = self._weighted_pool(x1_img, m1, self.sr_ratio)  # [B,C,H/R,W/R]
            x2_ds = self._weighted_pool(x2_img, m2, self.sr_ratio)

            x1_ds = x1_ds.reshape(B, C, -1).permute(0, 2, 1)
            x2_ds = x2_ds.reshape(B, C, -1).permute(0, 2, 1)

            x1_ds = self.norm1(x1_ds)
            x2_ds = self.norm2(x2_ds)

            kv1 = self.kv1(x1_ds).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2_ds).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            if self.sr_ratio > 1:
                x1_img = x1.permute(0, 2, 1).reshape(B, C, H, W)
                x2_img = x2.permute(0, 2, 1).reshape(B, C, H, W)

                x1_ds = self.sr1(x1_img).reshape(B, C, -1).permute(0, 2, 1)
                x2_ds = self.sr2(x2_img).reshape(B, C, -1).permute(0, 2, 1)

                x1_ds = self.norm1(x1_ds)
                x2_ds = self.norm2(x2_ds)

                kv1 = self.kv1(x1_ds).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x2_ds).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                

        # B num_heads HW/R² C//num_heads
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        # B num_heads N HW/R²
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        # B num_heads N C//num_heads -> B N num_heads C//num_heads -> B N C
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        return main_out, aux_out

class Cross_kvpool(nn.Module):
    # 加入KVPOOL
    def __init__(self, dim, reduction=1, num_heads=2, sr_ratio=4, norm_layer=nn.LayerNorm):
        super(Cross_kvpool, self).__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAtt_kvpool(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        

    def forward(self, x1, x2, H, W, m1=None, m2=None, detach_mask=False):
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)

        # 仅把 m1/m2 传给 CrossAttention（启用 K/V 加权池化）
        c1, c2 = self.cross_attn(z1, z2, H, W, m1=m1, m2=m2, detach_mask=detach_mask)

        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out  = self.norm2(x2 + self.end_proj2(y2))
        return main_out, aux_out

class CMFM_ega_cross3(nn.Module):
    # KV_Pool
    def __init__(self, c1, c2,  reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_ega_cross3, self).__init__()
        
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)
        self.att_lega2 = LEGA(dim=c1, stage=1)
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = Cross_kvpool(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        
        # === 新增：显著图头，产出 m1/m2 ===
        self.mask_head1 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.mask_head2 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # EGA + 位置先验
        x1_f = self.pre_local1(self.att_lega1(x1))
        x2_f = self.pre_local2(self.att_lega2(x2)) 

        m1 = self.mask_head1(x1_f)  # [B,1,H,W]
        m2 = self.mask_head2(x2_f)
    
        # Cross
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1c_t, x2c_t = self.cross(x1_t, x2_t, H, W, m1=m1, m2=m2, detach_mask=False)
        x1c = x1c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        gate = 0.5
        gated_x1 = gate * x1c
        gated_x2 = (1.0 - gate) * x2c
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class Cross_FiLM_kvpool(nn.Module):
    # 加入FiLM + KV pooling
    def __init__(self, dim, reduction=1, num_heads=2, sr_ratio=4, norm_layer=nn.LayerNorm):
        super(Cross_FiLM_kvpool, self).__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAtt_kvpool(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # FiLM 强度（可学习标量）
        self.film_lambda1 = nn.Parameter(torch.tensor(0.5))
        self.film_lambda2 = nn.Parameter(torch.tensor(0.5))
        

    def forward(self, x1, x2, H, W, m1=None, m2=None, detach_mask=False):
        """
        x1, x2: [B, N, C]
        m1, m2: 可选显著图 [B,1,H,W] 或可展平成 [B,N,1]
        """
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)

        # ---- B1: FiLM 调制（在降维后的 z1/z2 上）----
        if m1 is not None:
            m1_flat = m1.flatten(2).transpose(1, 2)  # [B,N,1]
            if detach_mask:
                m1_flat = m1_flat.detach()
            z1 = z1 * (1 + self.film_lambda1 * m1_flat)

        if m2 is not None:
            m2_flat = m2.flatten(2).transpose(1, 2)  # [B,N,1]
            if detach_mask:
                m2_flat = m2_flat.detach()
            z2 = z2 * (1 + self.film_lambda2 * m2_flat)
            
        # ---- Cross with EGA-KV Pooling（把 m1/m2 传下去启用 B2）----
        c1, c2 = self.cross_attn(z1, z2, H, W, m1=m1, m2=m2, detach_mask=detach_mask)

        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out  = self.norm2(x2 + self.end_proj2(y2))
        return main_out, aux_out

class CMFM_ega_cross4(nn.Module):
    # KV_Pool
    def __init__(self, c1, c2,  reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_ega_cross4, self).__init__()
        
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)
        self.att_lega2 = LEGA(dim=c1, stage=1)
        
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互（带 FiLM + KV Pooling） ===
        self.cross = Cross_FiLM_kvpool(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === 显著图头（供 FiLM 与 KV Pooling 使用） ===
        self.mask_head1 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())
        self.mask_head2 = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())


        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # EGA + 位置先验
        x1_f = self.pre_local1(self.att_lega1(x1))
        x2_f = self.pre_local2(self.att_lega2(x2)) 

        m1 = self.mask_head1(x1_f)  # [B,1,H,W]
        m2 = self.mask_head2(x2_f)
    
        # Cross
        B, C, H, W = x1.shape
        x1_t = x1_f.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2_f.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1c_t, x2c_t = self.cross(x1_t, x2_t, H, W, m1=m1, m2=m2, detach_mask=False)
        x1c = x1c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2c = x2c_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        gate = 0.5
        gated_x1 = gate * x1c
        gated_x2 = (1.0 - gate) * x2c
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse


################################################  2025.9.10 LEGA-Ablation: lega_in scharr gaussian的消融 ####################################

class CMFM_lega1_in(nn.Module):
    # RGB：lega0 scharr
    # SAR：lega0 scharr

    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega1_in, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=0)  # RGB -> Scharr
        self.att_lega2 = LEGA(dim=c1, stage=0)  # SAR -> Scharr

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        gate = 0.5  # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega2_in(nn.Module):
    # RGB：lega0 scharr
    # SAR：lega1 gaussian

    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega2_in, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=0)  # RGB -> Scharr
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        gate = 0.5  # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega3_in(nn.Module):
    # RGB：lega1 gaussian
    # SAR：lega0 scharr

    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega3_in, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=0)  # SAR -> Scharr

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        gate = 0.5  # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega4_in(nn.Module):
    # RGB：lega1 gaussian
    # SAR：lega1 gaussian
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega4_in, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        gate = 0.5  # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega5_in(nn.Module):
    # RGB：lega1 gaussian
    # SAR：CBAM
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega5_in, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        self.att_cbam1 = CBAM(channels=c1)  
        self.att_cbam2 = CBAM(channels=c1) 
        
        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = x[0]
        x2 = x[1]
        x1 = self.pre_local1(self.att_lega1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_cbam2(x[1])).contiguous()

        gate = 0.5  # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

class CMFM_lega6_in(nn.Module):
    # RGB：CBAM
    # SAR：lega1 gaussian
    
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_lega6_in, self).__init__()
        # === 原始分支注意力 ===
        self.att_lega1 = LEGA(dim=c1, stage=1)  # RGB -> Gaussian
        self.att_lega2 = LEGA(dim=c1, stage=1)  # SAR -> Gaussian

        self.att_cbam1 = CBAM(channels=c1)  
        self.att_cbam2 = CBAM(channels=c1)  
        
        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        # === 1：EGA ===
        x1 = self.pre_local1(self.att_cbam1(x[0])).contiguous()
        x2 = self.pre_local2(self.att_lega2(x[1])).contiguous()

        gate = 0.5  # [B, C, H, W]
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2

        # === 可学习残差缩放 + 融合 ===
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        fuse = self.post_fuse(fuse)
        return fuse

###################################  2025.9.12  Supplementary Experiments ####################################
class CMFM_ega_cross1(nn.Module):
    # direct 直接连接
    def __init__(self, c1, c2, reduction=4, sr_ratio=4, num_heads=2, norm_layer=nn.BatchNorm2d):
        super(CMFM_ega_cross1, self).__init__()
        
        self.att_lega1 = LEGA(dim=c1, stage=1)
        self.att_lega2 = LEGA(dim=c1, stage=1)
        # === C：局部先验（增强位置/局部性） ===
        self.pre_local1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        
        self.pre_local2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # === 跨模态交互 ===
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)

        # === sigmoid 门控 ===
        self.gate_conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

        # === 可学习残差系数 ===
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 对 x[0] 的缩放
        self.beta  = nn.Parameter(torch.tensor(0.5))  # 对 x[1] 的缩放

        # === 融合后轻量整形（对齐统计分布） ===
        self.post_fuse = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.apply(self._init_weights)
    
    # 结合 Transformer 初始化（trunc_normal_） 和 卷积网络初始化（Kaiming Normal） 的混合策略:  在训练早期更稳定、更快收敛
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = self.att_lega1(x[0])
        x2 = self.att_lega2(x[1])
        x1 = self.pre_local1(x1).contiguous()
        x2 = self.pre_local2(x2).contiguous()

        # === Cross Attention ===
        B, C, H, W = x1.shape
        x1_t = x1.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x2_t = x2.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        x1_t, x2_t = self.cross(x1_t, x2_t, H, W)
        x1 = x1_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2_t.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        
        gate = 0.5
        gated_x1 = gate * x1
        gated_x2 = (1.0 - gate) * x2
        fuse = gated_x1 + gated_x2 + self.alpha * x[0] + self.beta * x[1]
        return fuse

###################################  2025.9.13  OSPRC  ####################################

