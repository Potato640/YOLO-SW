import torch
from torch import nn
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, mode="*"):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv 71.7
        self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)  # dilation conv 72.6 our
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  # dilation conv 72.4
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=5, dilation=5)  # dilation conv 71.2  640-500nwp88.5
        self.f = nn.Linear(dim, 6 * dim)
        self.act = nn.GELU()
        self.g = nn.Linear(3 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else 1.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        tmp = x
        x = x.permute(0, 2, 3, 1)
        input = x
        x = self.norm(x)
        # x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.diconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.f(x)
        B, H, W, C = x.size()
        x1, x2 = x.reshape(B, H, W, 2, int(C // 2)).unbind(3)
        x = self.act(x1) + x2 if self.mode == "sum" else self.act(x1) * x2
        x = self.g(x)
        x = input + self.drop_path(self.gamma * x) if self.mode == "sum" else  input * self.drop_path(self.gamma * x)
        x = x.permute(0, 3, 1, 2)
        x = tmp + x
        return x


if __name__ == "__main__":
    # #########################测试数据 ################################
    x = torch.ones(3, 512, 32, 32)    # print(model)
    model = Block(512)
    out = model(x)
    print(out.shape)

    # ##################################################################