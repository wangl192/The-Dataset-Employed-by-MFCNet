import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        # 初始化代码
        super().__init__()
        # 二维卷积，输入通道为2，输出通道为1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()  # 将输出映射到0-1之间

    def forward(self, x):
        # 获取最大池化特征和平均池化特征
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化 [B,C,H,W]-->[B,1,H,W]
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 平均池化 [B,C,H,W]-->[B,1,H,W]
        result = torch.cat([max_result, avg_result], 1)  # 将最大池化特征和平均池化特征拼接 [B,2,H,W]
        output = self.conv(result)  # 卷积 [B,2,H,W]-->[B,1,H,W]
        output = self.sigmoid(output)  # 将输出映射到0-1之间
        return output

return_attention = True

class DDJA(nn.Module):
    def __init__(self, in_channels, act_ratio=0.5, act_fn=nn.GELU):
        super().__init__()

        reduce_channels = int(in_channels * act_ratio)
        self.spatial1 = SpatialAttention()

        self.norm = nn.LayerNorm(in_channels)

        self.channel_reduce = nn.Linear(in_channels, reduce_channels)
        self.channel_up = nn.Linear(reduce_channels, in_channels)

        self.spatial_reduce = nn.Linear(in_channels, reduce_channels)
        self.spatial_select = nn.Linear(reduce_channels, 1)
        # 激活函数
        self.act_fn = act_fn()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attention=False):  # 添加 return_attention 参数，默认值为 False
        B, C, H, W = x.shape
        ori_x = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        copy = x
        x_avg = F.avg_pool2d(copy, x.shape[2], x.shape[3])
        x_max = F.max_pool2d(copy, x.shape[2], x.shape[3])
        x_global = (x_max + x_avg).permute(0, 2, 3, 1)
        x_global = self.act_fn(self.channel_reduce(x_global))
        channel_out_1 = self.sigmoid(self.channel_up(x_global))
        channel_out_1 = channel_out_1.permute(0, 3, 1, 2)
        channel_out = channel_out_1 * ori_x

        channel_out = channel_out.permute(0, 2, 3, 1)
        x_local = self.act_fn(self.spatial_reduce(channel_out))
        x_global = x_global.expand(-1, H, W, -1)
        x_mid = x_global * x_local

        x_mid_temp = x_mid.permute(0, 3, 1, 2)
        spatial_oup_1 = self.spatial1(x_mid_temp).permute(0, 2, 3, 1)  # 形状: (B, H, W, 1)

        out = spatial_oup_1.expand_as(channel_out) * channel_out
        out = out.permute(0, 3, 1, 2)

        if return_attention:
            # 返回注意力权重（形状: (B, H, W)，去掉最后一维）
            return out, spatial_oup_1.squeeze(-1)  # squeeze 后形状: (B, H, W)
        else:
            return out



if __name__ == "__main__":
    # 假设输入通道数为 16

    in_channels = 16
    # 创建 BiAttn 模块
    bi_attn = DDJA(in_channels)
    # 创建一个输入张量，形状为 (batch_size, channels, height, width)
    input_tensor = torch.randn(4, in_channels, 32, 32)
    # 前向传播
    output_tensor = bi_attn(input_tensor)

    # 打印输出张量的形状
    print("Output tensor shape:", output_tensor.shape)
    # 检查输出张量是否有 NaN 或 Inf 值
    assert not torch.isnan(output_tensor).any(), "Output contains NaN"
    assert not torch.isinf(output_tensor).any(), "Output contains Inf"