import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        max_out = self.mlp(self.max_pool(x))  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        x = self.conv1(x)  # 通过卷积层得到注意力权重:(B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)  # 通过通道注意力机制得到的特征图,x:(B,C,H,W),ca(x):(B,C,1,1),out:(B,C,H,W)
        result = out * self.sa(out)  # 通过空间注意力机制得到的特征图,out:(B,C,H,W),sa(out):(B,1,H,W),result:(B,C,H,W)
        return result


if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)
    b, c, h, w = x.shape
    net = CBAM(c)
    y = net(x)
    print(y.size())
