import torch
import torch.nn as nn
from My_attention2 import DDJA

class L2NormDense(nn.Module):

    def __init__(self):

        super(L2NormDense, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)
        return x


class BasicBlock1(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True 节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x))


class Mul_OSnet(nn.Module):

    def __init__(self):
        # block: 基本模块类型，可以是 BasicBlock 或其他自定
        super().__init__()
        self.basic_block1 = BasicBlock1(in_channels=32, out_channels=32)
        self.basic_block2 = BasicBlock1(in_channels=64, out_channels=64)
        self.basic_block3 = BasicBlock1(in_channels=128, out_channels=128)
        self.cs_att1 = DDJA(32)
        self.cs_att2 = DDJA(64)
        self.cs_att3 = DDJA(128)

        # self.CBAMBlock1 = CBAMBlock(32)
        # self.CBAMBlock2 = CBAMBlock(64)
        # self.CBAMBlock3 = CBAMBlock(128)
        #
        # self.se_Block1 = SE_Block(32)
        # self.se_Block2 = SE_Block(64)
        # self.se_Block3 = SE_Block(128)


        # 定义第一个卷积层，包括卷积、批归一化和 ReLU 激活
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),  # 输入通道为 1，输出通道为 32
            nn.BatchNorm2d(32),  # 批归一化，用于加速训练和提高稳定性
            nn.ReLU(inplace=True)  # ReLU 激活函数，inplace=True 节省内存
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 批归一化，用于加速训练和提高稳定性
            nn.ReLU(inplace=True)  # ReLU 激活函数，inplace=True 节省内存
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),  # 批归一化，用于加速训练和提高稳定性
            nn.ReLU(inplace=True)  # ReLU 激活函数，inplace=True 节省内存
        )
        # 定义一个卷积层，使用 dropout 和批归一化
        self.conv4 = nn.Sequential(
            nn.Dropout(0.1),  # Dropout 防止过拟合，丢弃率为 0.1
            nn.Conv2d(224, 9, kernel_size=1, padding=1, bias=False),  # 卷积，将通道数变为 9
            nn.BatchNorm2d(9)  # 批归一化
        )

        # 定义一个 1x1 卷积层，用于通道压缩和融合
        self.conv_cat = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),  # 1x1 卷积，通道数从 96 变为 64
            nn.BatchNorm2d(64),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU 激活函数
        )


    def input_norm(self, x):
        """
        对输入张量 x 进行归一化处理
        归一化公式: (x - 均值) / 标准差
        参数:
            x: 输入张量，形状为 (batch_size, channels, height, width)
        返回:
            归一化后的张量
        """
        # 将输入张量展平为二维 (batch_size, num_features)
        flat = x.view(x.size(0), -1)
        # 计算每个样本的均值 (沿着 feature 维度)
        mp = torch.mean(flat, dim=1)
        # 计算每个样本的标准差，并加入一个小的常数防止除以零
        sp = torch.std(flat, dim=1) + 1e-7
        # 归一化公式
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).expand_as(x)

    def forward(self, x , visualize_attn=True):
        """
        定义前向传播的逻辑
        参数:
            x: 输入张量，形状为 (batch_size, 1, height, width)
        返回:
            模型的最终输出张量
        """
        # 第一个卷积层，输入经过归一化处理
        output1 = self.conv1(self.input_norm(x))
        output2 = self.basic_block1(output1)
        output3_1 = output2 + output1
        if visualize_attn:
            output_cs2, attn1 = self.cs_att1(output2, return_attention=True)  # 获取注意力权重
        else:
            output_cs2 = self.cs_att1(output2)  # 普通前向传播
        output3 = self.conv2(output3_1)
        output4 = self.basic_block2(output3)
        output4_1 = output4 + output3
        if visualize_attn:
            output_cs4, attn2 = self.cs_att2(output4, return_attention=True)
        else:
            output_cs4 = self.cs_att2(output4)

        output5 = self.conv3(output4_1)
        output6 = self.basic_block3(output5)
        if visualize_attn:
            output_cs6, attn3 = self.cs_att3(output6, return_attention=True)
        else:
            output_cs6 = self.cs_att3(output6)

            # 拼接特征（根据是否可视化决定是否使用带注意力的特征）
        if visualize_attn:
            # 返回注意力权重（attn1, attn2, attn3 形状均为 (B, H, W)）
            return torch.cat([output_cs2, output_cs4, output_cs6], 1), [attn1, attn2, attn3]
        else:
            return self.conv4(torch.cat([output_cs2, output_cs4, output_cs6], 1))

        # output = self.conv4(torch.cat([output2, output4, output6], 1))

        # 加
        # output_cs2 = self.CBAMBlock1(output2)
        # output_cs4 = self.CBAMBlock2(output4)
        # output_cs6 = self.CBAMBlock3(output6)
        # output = self.conv4(torch.cat([output_cs2, output_cs4, output_cs6], 1))
        # 加
        # output_cs2 = self.cs_att1(output2)
        # output_cs4 = self.cs_att2(output4)
        # output_cs6 = self.cs_att3(output6)
        # output = self.conv4(torch.cat([output_cs2, output_cs4, output_cs6], 1))
        # 加SE
        # output_cs2 = self.se_Block1(output2)
        # output_cs4 = self.se_Block2(output4)
        # output_cs6 = self.se_Block3(output6)
        # output = self.conv4(torch.cat([output_cs2, output_cs4, output_cs6], 1))
        # return output


class Mul_OSnetPseudo(nn.Module):

    def __init__(self):
        super(Mul_OSnetPseudo, self).__init__()
        self.ResNet_Opt = Mul_OSnet()
        self.ResNet_Sar = Mul_OSnet()

    def input_norm(self, x):
        """
        对输入张量 x 进行归一化处理
        归一化公式: (x - 均值) / 标准差

        参数:
            x: 输入张量，形状为 (batch_size, channels, height, width)
        返回:
            归一化后的张量
        """
        # 将输入张量展平为二维 (batch_size, num_features)
        flat = x.view(x.size(0), -1)
        # 计算每个样本的均值 (沿着 feature 维度)
        mp = torch.mean(flat, dim=1)
        # 计算每个样本的标准差，并加入一个小的常数防止除以零
        sp = torch.std(flat, dim=1) + 1e-7
        # 归一化公式
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).expand_as(x)

    def forward(self, input_opt, input_sar,visualize_attn=True):
        """
        定义模型的前向传播逻辑
        参数:
            input_opt: Optical 数据输入张量，形状为 (batch_size, channels, height, width)
            input_sar: SAR 数据输入张量，形状为 (batch_size, channels, height, width)
        返回:
            经过 L2NormDense 归一化的 Optical 和 SAR 特征
        """
        # 对 Optical 和 SAR 数据分别进行归一化
        input_opt = self.input_norm(input_opt)
        input_sar = self.input_norm(input_sar)

        if visualize_attn:
            # 提取光学分支的特征和注意力权重
            feat_opt, attn_opt = self.ResNet_Opt(input_opt, visualize_attn=True)
            # 提取 SAR 分支的特征和注意力权重
            feat_sar, attn_sar = self.ResNet_Sar(input_sar, visualize_attn=True)
            return feat_opt, feat_sar, attn_opt, attn_sar  # 返回特征和注意力权重
        else:
            feat_opt = self.ResNet_Opt(input_opt)
            feat_sar = self.ResNet_Sar(input_sar)
            return L2NormDense()(feat_opt), L2NormDense()(feat_sar)


        # # 将 Optical 数据输入到 Optical 子网络，提取 Optical 特征
        # features_opt = self.ResNet_Opt(input_opt)
        # # 将 SAR 数据输入到 SAR 子网络，提取 SAR 特征
        # features_sar = self.ResNet_Sar(input_sar)
        #
        # # 使用 L2NormDense 模块对 Optical 和 SAR 特征进行 L2 归一化
        # return L2NormDense()(features_opt), L2NormDense()(features_sar)

