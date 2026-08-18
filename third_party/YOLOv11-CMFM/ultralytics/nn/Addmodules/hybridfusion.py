import torch
import torch.nn as nn

class HybridAdd(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, c1, c2,weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        x_left = x[0]
        x_right = x[1]
        return x[0] * self.w + x[1] * (1 - self.w)

class HybridMulti(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, c1, c2,weight=0.5):
        super(HybridMulti, self).__init__()

    def forward(self, x):
        x_left = x[0]
        x_right = x[1]
        return x[0] * x[1] 
    
class HybridGFU(nn.Module):
    def __init__(self, c1, c2):
        super(HybridGFU, self).__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1),
            nn.Sigmoid()  # 输出 gate ∈ (0, 1)
        )

    def forward(self, x):
        feat_vis = x[0]
        feat_inf = x[1]

        # 拼接两个模态特征
        fused = torch.cat([feat_vis, feat_inf], dim=1)  # shape: [B, 2C, H, W]
        # 生成门控权重
        gate = self.gate_conv(fused)  # shape: [B, C, H, W], 每个像素一个 gate 值
        out = gate * feat_vis + (1 - gate) * feat_inf
        return out

# input1 = torch.randn(1, 6, 256, 256)
# input2 = torch.randn(1, 6, 256, 256)
# HybridGFU = HybridGFU(12,6)s
# output = HybridGFU([input1,input2])
# print(output.shape) #[1, 3, 256, 256]