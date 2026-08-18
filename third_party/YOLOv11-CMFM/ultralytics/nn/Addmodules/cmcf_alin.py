import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import  DropPath, trunc_normal_
from math import log

#######################################################
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

class GatingMechanism(nn.Module):
    """门控机制：自适应权重分配"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CMCF(nn.Module):
    def __init__(self, c1,c2, reduction=16, sr_ratio=4, num_heads=2 ,use_gating=True):
        super(CMCF, self).__init__()
        # 门控机制
        self.use_gating = use_gating
        if use_gating:
            self.gate1 = GatingMechanism(dim = c1)
            self.gate2 = GatingMechanism(dim = c1)
        # 特征交互
        self.cross = FeatureInteraction(dim=c1, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 1)
        )
        # 输出投影
        self.output_proj = nn.Conv2d(c1, c1, 1)
        # 残差连接
        self.shortcut = nn.Conv2d(c1 * 2, c1, 1) 
        
    def forward(self, x):
        x1_feat = x[0]
        x2_feat = x[1]
        
        # shortcut
        res1 = x1_feat
        res2 = x2_feat

        # Gating Mechanism
        if self.use_gating:
            x1_feat = self.gate1(x1_feat)
            x2_feat = self.gate2(x2_feat)

        #  Cross Attention
        B, C, H, W = x1_feat.shape
        # B C (HW)->B N(HW) C
        x1 = x1_feat.flatten(2).transpose(1, 2)
        x2 = x2_feat.flatten(2).transpose(1, 2)
        # B N(HW) C
        x1, x2 = self.cross(x1, x2, H, W)
        # B C H W
        B, N, _C = x1.shape
        x1_cross = x1.permute(0, 2, 1).reshape(B, _C, H, W)
        x2_cross = x2.permute(0, 2, 1).reshape(B, _C, H, W)
        
        # Feature Fusion
        fused_feat = torch.cat([x1_cross, x2_cross], dim=1)  # [B, fusion_dim*2, H, W]
        fused_feat = self.fusion_conv(fused_feat)
        
        # 残差连接
        shortcut = self.shortcut(torch.cat([x1_feat, x2_feat], dim=1))
        fused_feat = fused_feat + shortcut
        
        # 输出投影
        output = self.output_proj(fused_feat)
        return output

#######################################################
from typing import List, Tuple, Optional
class StrongChannelAttention(nn.Module):
    """强通道注意力机制 - 用于AFR阶段，注意力强度0.7"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.attention_strength = 0.7  # 强注意力强度
        
        # 通道注意力网络
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # 通道权重
        self.channel_weight = nn.Parameter(torch.ones(channels))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 平均池化分支
        avg_out = self.shared_mlp(self.avg_pool(x))
        
        # 最大池化分支
        max_out = self.shared_mlp(self.max_pool(x))
        
        # 通道注意力权重
        channel_att = torch.sigmoid(avg_out + max_out)
        
        # 应用通道权重
        channel_att = channel_att * self.channel_weight.view(1, -1, 1, 1)
        
        # 强通道注意力融合：保护原始特征
        result = x * (1 - self.attention_strength) + x * channel_att * self.attention_strength
        
        return result

class ProgressiveSelfAttention(nn.Module):
    """递进式自注意力机制 - 用于HCMT阶段，注意力强度0.5"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.attention_strength = 0.5  # 中等注意力强度
        
        # 自注意力组件
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        # 渐进式权重参数
        self.gamma = nn.Parameter(torch.tensor(0.5))
        
        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q, K, V
        query = self.query(x).reshape(B, C, -1)  # B, C, HW
        key = self.key(x).reshape(B, C, -1)      # B, C, HW
        value = self.value(x).reshape(B, C, -1)   # B, C, HW
        
        # 计算注意力分数
        scores = torch.bmm(query.transpose(1, 2), key)  # B, HW, HW
        scores = F.softmax(scores, dim=-1)
        
        # 应用注意力
        attended = torch.bmm(value, scores.transpose(1, 2))  # B, C, HW
        attended = attended.reshape(B, C, H, W)
        
        # 输出投影
        attended = self.out_proj(attended)
        
        # 渐进式自注意力融合：保护原始特征
        result = x * (1 - self.attention_strength) + attended * self.attention_strength
        
        return result

class MultiScaleProgressiveEnhancement(nn.Module):
    """多尺度递进式增强 - 用于HCMT阶段"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 多尺度卷积路径
        self.conv1 = nn.Conv2d(channels, channels, 1)      # 1x1
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)  # 3x3
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2)  # 5x5
        
        # 自适应权重
        self.scale_weights = nn.Parameter(torch.ones(3))
        
        # 融合层
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        
    def forward(self, x):
        # 多尺度特征提取
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        
        # 自适应权重融合
        weights = F.softmax(self.scale_weights, dim=0)
        fused = torch.cat([
            conv1_out * weights[0],
            conv3_out * weights[1],
            conv5_out * weights[2]
        ], dim=1)
        
        return self.fusion(fused)

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络 - 用于HCMT阶段"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 自顶向下路径
        self.top_down = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 3, padding=1)
        ])
        
        # 横向连接
        self.lateral = nn.ModuleList([
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 1)
        ])
        
    def forward(self, x):
        # 简化的FPN实现
        features = [x]
        
        # 自顶向下路径
        for i, layer in enumerate(self.top_down):
            if i == 0:
                feat = F.avg_pool2d(x, 2)
            else:
                feat = F.avg_pool2d(features[-1], 2)
            
            feat = layer(feat)
            features.append(feat)
        
        # 自底向上路径
        for i in range(len(features) - 2, -1, -1):
            # 上采样
            up_feat = F.interpolate(features[i + 1], size=features[i].shape[2:], mode='nearest')
            # 横向连接
            lateral_feat = self.lateral[i](features[i])
            # 融合
            features[i] = up_feat + lateral_feat
        
        return features[0]  # 返回最高分辨率特征

class HCMT(nn.Module):
    """递进式HCMT模块 - 第二阶段：多尺度增强"""
    def __init__(self, c1, c2, num_heads=2, use_attention=True):
        super(HCMT, self).__init__()
        self.fusion_dim = c1
        
        # 特征投影层
        self.modal1_proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
        self.modal2_proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
        # 递进式自注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.self_attention = ProgressiveSelfAttention(c1)
        
        # 多尺度递进式增强
        self.multiscale_enhancement = MultiScaleProgressiveEnhancement(c1)
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(c1)
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        # 残差连接
        self.shortcut = nn.Conv2d(c1 * 2, c1, 1)
        
        # 输出处理
        self.output_proj = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
    def forward(self, x):
        x1_feat = x[0]
        x2_feat = x[1]
        
        # 特征投影
        modal1_proj = self.modal1_proj(x1_feat)
        modal2_proj = self.modal2_proj(x2_feat)
        # 特征融合
        combined_feat = torch.cat([modal1_proj, modal2_proj], dim=1)
        combined_feat = self.fusion_conv(combined_feat)
        
        # 递进式自注意力 - 多尺度增强
        if self.use_attention:
            enhanced_feat = self.self_attention(combined_feat)
        else:
            enhanced_feat = combined_feat
        # 多尺度递进式增强
        multiscale_feat = self.multiscale_enhancement(enhanced_feat)
        # 特征金字塔网络处理
        fpn_features = self.fpn(multiscale_feat)
        # 残差连接
        shortcut = self.shortcut(torch.cat([x1_feat, x2_feat], dim=1))
        output = fpn_features + shortcut
        # 输出处理
        output = self.output_proj(output)
        return output

#######################################################

def window_partition(x, win):  # x: B,C,H,W -> B*nw,C,win,win
    B, C, H, W = x.shape
    assert H % win == 0 and W % win == 0, "H,W 必须能被 window_size 整除"
    x = x.view(B, C, H // win, win, W // win, win)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # B, nh, nw, C, win, win
    x = x.view(B * (H // win) * (W // win), C, win, win)  # B*nw, C, win, win
    return x

def window_reverse(windows, win, H, W):  # windows: B*nw,C,win,win -> B,C,H,W
    Bnw, C, _, _ = windows.shape
    B = Bnw // ((H // win) * (W // win))
    windows = windows.view(B, H // win, W // win, C, win, win)
    x = windows.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x



class ProgressiveSpatialAttention(nn.Module):
    """递进式空间注意力机制 - 用于AFR阶段"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.attention_strength = 0.7  # 强注意力强度
        
        # 空间注意力网络
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 空间权重
        self.spatial_weight = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 空间统计特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 空间注意力权重
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        
        # 强空间注意力融合：保护原始特征
        result = x * (1 - self.attention_strength) + x * spatial_att * self.attention_strength
        
        return result

class AFR(nn.Module):
    """递进式AFR模块 - 第三阶段：特征重校准"""
    def __init__(self, c1, c2,  use_attention=True):
        """
        - c1: 第一分支通道数，同时作为fusion_dim使用
        - use_attention: 是否启用注意力（保持原AFR逻辑）
        """
        super(AFR, self).__init__()
        self.modal1_dim = c1
        self.modal2_dim = c1
        self.fusion_dim = c1
        self.use_attention = use_attention

        # 特征投影层（保持原功能：各自->fusion_dim）
        self.modal1_proj = nn.Sequential(
            nn.Conv2d(self.modal1_dim, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU()
        )
        self.modal2_proj = nn.Sequential(
            nn.Conv2d(self.modal2_dim, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU()
        )

        # 强通道注意力 + 逐步空间注意力（不使用交叉注意力）
        if use_attention:
            self.channel_attention = StrongChannelAttention(self.fusion_dim)
            self.spatial_attention = ProgressiveSpatialAttention(self.fusion_dim)

        # 深度融合网络（保持原结构）
        self.deep_fusion = nn.Sequential(
            nn.Conv2d(self.fusion_dim * 2, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU(),
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 3, padding=1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU(),
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 3, padding=1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU(),
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim)
        )

        # 残差连接（concat原始两模态 -> 匹配fusion_dim）
        in_sc = self.modal1_dim + self.modal2_dim
        self.shortcut = nn.Conv2d(in_sc, self.fusion_dim, 1) if in_sc != self.fusion_dim else nn.Identity()

        # 输出处理（保持原结构）
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 3, padding=1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU()
        )

    def forward(self, x):
        x1_feat, x2_feat = x[0], x[1]

        # 特征投影
        modal1_proj = self.modal1_proj(x1_feat)
        modal2_proj = self.modal2_proj(x2_feat)

        # 特征融合
        combined_feat = torch.cat([modal1_proj, modal2_proj], dim=1)
        fused_feat = self.deep_fusion(combined_feat)

        # 强注意力机制 - 特征重校准
        if self.use_attention:
            fused_feat = self.channel_attention(fused_feat)   # 通道注意力
            fused_feat = self.spatial_attention(fused_feat)   # 空间注意力

        # 残差连接（拼接原始输入做shortcut）
        shortcut = self.shortcut(torch.cat([x1_feat, x2_feat], dim=1))
        fused_feat = fused_feat + shortcut

        # 输出处理
        output = self.output_proj(fused_feat)
        return output

#############################################################


def _pad_to_multiple_hw(x, multiple: int):
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, pad_h, pad_w

def _unpad_hw(x, pad_h: int, pad_w: int):
    if pad_h or pad_w:
        return x[:, :, :-pad_h if pad_h else x.shape[-2], :-pad_w if pad_w else x.shape[-1]]
    return x

def window_partition(x, win):
    B, C, H, W = x.shape
    x = x.view(B, C, H // win, win, W // win, win)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(B * (H // win) * (W // win), C, win, win)
    return x

def window_reverse(windows, win, H, W):
    Bnw, C, _, _ = windows.shape
    B = Bnw // ((H // win) * (W // win))
    windows = windows.view(B, H // win, W // win, C, win, win)
    x = windows.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x

class StrongCrossModalAttention(nn.Module):
    """
    跨模态窗口注意力（省显存版，无SDPA）
    - window_size: 窗口注意力
    - reduction:   Q/K/V 通道降维
    - kv_stride:   K/V 下采样（窗口内池化）
    """
    def __init__(self, query_dim, key_dim, num_heads=2,
                 reduction: int = 4, window_size: int = 8, kv_stride: int = 2,
                 attention_strength: float = 0.9, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.kv_stride = kv_stride
        self.attention_strength = attention_strength

        # 降维后的通道
        reduced_dim = max(1, query_dim // reduction)
        assert reduced_dim % num_heads == 0
        self.head_dim = reduced_dim // num_heads
        self.reduced_dim = reduced_dim
        self.query_dim = query_dim

        # QKV 投影
        self.q_proj = nn.Linear(query_dim, reduced_dim, bias=True)
        self.k_proj = nn.Linear(key_dim,   reduced_dim, bias=True)
        self.v_proj = nn.Linear(key_dim,   reduced_dim, bias=True)
        self.o_proj = nn.Linear(reduced_dim, query_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    def _attention(self, q_seq, k_seq, v_seq):
        """
        显式 QK^T + softmax + matmul
        q_seq: [B*, N, Cr]
        k_seq: [B*, Nk, Cr]
        v_seq: [B*, Nk, Cr]
        """
        Bm, N, Cr = q_seq.shape
        h, d = self.num_heads, self.head_dim

        Q = self.q_proj(q_seq).reshape(Bm, N, h, d).transpose(1, 2)  # [Bm,h,N,d]
        K = self.k_proj(k_seq).reshape(Bm, -1, h, d).transpose(1, 2) # [Bm,h,Nk,d]
        V = self.v_proj(v_seq).reshape(Bm, -1, h, d).transpose(1, 2) # [Bm,h,Nk,d]

        # QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)  # [Bm,h,N,Nk]
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # 乘 V
        out = torch.matmul(attn, V)  # [Bm,h,N,d]
        out = out.transpose(1, 2).reshape(Bm, N, h * d)  # [Bm,N,Cr]
        out = self.o_proj(out)                            # [Bm,N,C]
        out = self.proj_drop(out)
        return out

    def forward(self, query, key):
        """
        query: B,C,H,W
        key:   B,Ck,H,W
        """
        B, C, H, W = query.shape
        res = query

        # pad到 window_size
        win = max(1, self.window_size)
        q_pad, pad_h, pad_w = _pad_to_multiple_hw(query, win)
        k_pad, _, _ = _pad_to_multiple_hw(key, win)
        _, _, Hp, Wp = q_pad.shape

        # 窗口切分
        q_win = window_partition(q_pad, win)  # [B*nw,C,win,win]
        k_win = window_partition(k_pad, win)

        # K/V 下采样
        if self.kv_stride > 1:
            kd = F.avg_pool2d(k_win, kernel_size=self.kv_stride, stride=self.kv_stride)
            vd = F.avg_pool2d(k_win, kernel_size=self.kv_stride, stride=self.kv_stride)
        else:
            kd, vd = k_win, k_win

        # 展平成序列
        q_seq = q_win.flatten(2).transpose(1, 2)  # [B*nw,N,C]
        k_seq = kd.flatten(2).transpose(1, 2)     # [B*nw,Nk,C]
        v_seq = vd.flatten(2).transpose(1, 2)

        # 注意力
        out_win = self._attention(q_seq, k_seq, v_seq)               # [B*nw,N,C]
        out_win = out_win.transpose(1, 2).reshape(q_win.shape[0], C, win, win)

        # 窗口还原 & 去 pad
        out = window_reverse(out_win, win, Hp, Wp)
        out = _unpad_hw(out, pad_h, pad_w)

        # 融合
        out = res * (1.0 - self.attention_strength) + out * self.attention_strength
        return out

class ProgressivePyramidPooling(nn.Module):
    """递进式金字塔池化 - 用于MSCF阶段"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 多尺度池化
        self.pool1 = nn.AdaptiveAvgPool2d(1)      # 全局池化
        self.pool2 = nn.AdaptiveAvgPool2d(2)      # 2x2池化
        self.pool3 = nn.AdaptiveAvgPool2d(4)      # 4x4池化
        
        # 投影层
        self.proj1 = nn.Conv2d(channels, channels, 1)
        self.proj2 = nn.Conv2d(channels, channels, 1)
        self.proj3 = nn.Conv2d(channels, channels, 1)
        
        # 融合层
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 多尺度池化
        pool1 = self.pool1(x)  # B, C, 1, 1
        pool2 = self.pool2(x)  # B, C, 2, 2
        pool3 = self.pool3(x)  # B, C, 4, 4
        
        # 投影
        proj1 = self.proj1(pool1)
        proj2 = self.proj2(pool2)
        proj3 = self.proj3(pool3)
        
        # 上采样到原始尺寸
        proj1 = F.interpolate(proj1, size=(H, W), mode='nearest')
        proj2 = F.interpolate(proj2, size=(H, W), mode='nearest')
        proj3 = F.interpolate(proj3, size=(H, W), mode='nearest')
        
        # 融合
        fused = torch.cat([proj1, proj2, proj3], dim=1)
        output = self.fusion(fused)
        
        return output

class MSCF(nn.Module):
    """递进式MSCF模块 - 第四阶段：最终融合"""
    
    def __init__(self, c1, c2, num_heads=2, use_attention=True):
        """
        - c1: 第一分支通道数，同时作为 fusion_dim 使用
        - num_heads: 传入强交叉注意力
        - use_attention: 占位参数（本模块始终使用交叉注意力，不改动原功能）
        """
        super(MSCF, self).__init__()
        self.modal1_dim = c1
        self.modal2_dim = c1
        self.fusion_dim = c1
        self.use_attention = use_attention  # 占位，不改变原有逻辑
        
        # 特征投影层（各模态 -> fusion_dim）
        self.modal1_proj = nn.Sequential(
            nn.Conv2d(self.modal1_dim, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU()
        )
        self.modal2_proj = nn.Sequential(
            nn.Conv2d(self.modal2_dim, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU()
        )

        # 强交叉注意力机制（双向）
        self.cross_attention1 = StrongCrossModalAttention(self.fusion_dim, self.fusion_dim, num_heads)
        self.cross_attention2 = StrongCrossModalAttention(self.fusion_dim, self.fusion_dim, num_heads)

        # 递进式金字塔池化
        self.pyramid_pooling = ProgressivePyramidPooling(self.fusion_dim)

        # 复杂融合网络（保持尺寸：dilation=2 对应 padding=2）
        self.complex_fusion = nn.Sequential(
            nn.Conv2d(self.fusion_dim * 2, self.fusion_dim, 3, padding=1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU(),
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU(),
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 1),
            nn.BatchNorm2d(self.fusion_dim)
        )
        
        # 残差连接（concat原始两模态 → 1×1 对齐到 fusion_dim）
        in_sc = self.modal1_dim + self.modal2_dim
        self.shortcut = nn.Conv2d(in_sc, self.fusion_dim, 1) if in_sc != self.fusion_dim else nn.Identity()

        # 输出处理
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.fusion_dim, self.fusion_dim, 3, padding=1),
            nn.BatchNorm2d(self.fusion_dim),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        与HCMT一致：x 为 [x1_feat, x2_feat]
        """
        x1_feat, x2_feat = x[0], x[1]
        B, C1, H, W = x1_feat.shape
        B2, C2, H2, W2 = x2_feat.shape
        # （可选）断言两模态空间一致
        # assert H == H2 and W == W2, "Two modalities must have the same H and W"

        # 特征投影
        modal1_proj = self.modal1_proj(x1_feat)
        modal2_proj = self.modal2_proj(x2_feat)

        # 强交叉注意力（双向）
        modal1_attended = self.cross_attention1(modal1_proj, modal2_proj)
        modal2_attended = self.cross_attention2(modal2_proj, modal1_proj)

        # 特征融合
        combined_feat = torch.cat([modal1_attended, modal2_attended], dim=1)
        fused_feat = self.complex_fusion(combined_feat)

        # 递进式金字塔池化并相加
        pyramid_feat = self.pyramid_pooling(fused_feat)
        final_feat = fused_feat + pyramid_feat

        # 残差连接（来自原始输入）
        shortcut = self.shortcut(torch.cat([x1_feat, x2_feat], dim=1))
        final_feat = final_feat + shortcut

        # 输出处理
        output = self.output_proj(final_feat)
        return output

class EnhancedMSCF(nn.Module):
    """增强版递进式MSCF模块"""

    def __init__(self, c1, c2, num_heads=2, use_attention=True):
        """
        与 HCMT 对齐的接口：
        - c1: 第一分支通道数，同时作为 fusion_dim
        - num_heads: 传递给内部 MSCF
        - use_attention: 与 HCMT 接口保持一致（本模块增强路径不依赖该开关）
        """
        super(EnhancedMSCF, self).__init__()
        self.fusion_dim = c1
        self.use_attention = use_attention  # 占位

        # 基础MSCF模块（已改为HCMT风格：__init__(c1,c2,...) & forward(x)）
        self.base_mscf = MSCF(
            c1, c2, num_heads=num_heads, use_attention=use_attention
        )

        # 增强模块（保持尺寸：dilation=3 ⇒ padding=3）
        self.enhancement = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 3, padding=3, dilation=3),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 1),
            nn.BatchNorm2d(c1)
        )

        # 注意力门控（SE风格）
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, c1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [x1_feat, x2_feat]，与 HCMT 保持一致
        """
        # 基础 MSCF 融合（直接按 HCMT 风格传入）
        base_feat = self.base_mscf(x)

        # 增强处理
        enhanced_feat = self.enhancement(base_feat)

        # 注意力门控
        gate = self.attention_gate(enhanced_feat)
        gated_feat = enhanced_feat * gate

        # 残余连接
        output = base_feat + gated_feat
        return output


