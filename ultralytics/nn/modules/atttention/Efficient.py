import torch
import torch.nn as nn


class ContextLayer(nn.Module):
    def __init__(self, in_dim, context_act=nn.GELU,
                 context_f=True, context_g=True):
        # channel last
        super().__init__()
        self.f = nn.Linear(in_dim, in_dim // 2) if context_f else nn.Identity()
        self.g = nn.Linear(in_dim // 2, in_dim) if context_g else nn.Identity()
        self.act = context_act() if context_act else nn.Identity()

        self.conv = nn.Conv2d(in_dim // 2, in_dim // 2 , 5, stride=1, padding=5 // 2)


    def forward(self, x):
        tmp = x
        x = x.permute(0, 2, 3, 1)
        x = self.f(x)
        tmp2 = self.g(x)

        tmp2 = tmp2.permute(0, 3, 2, 1)
        # print("tmp2--", tmp2.shape)
        # print("1--",x.shape)
        out = 0

        x = x.permute(0, 3, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        # print("2--",x.shape)
        x = x.permute(0, 3, 2, 1)
        # print("mid--", x.shape)
        ctx = self.g(x)
        ctx = ctx.permute(0, 3, 2, 1)
        # print("3--",ctx.shape)

        out = tmp2 * ctx
        # print("out--",out.shape)

        return out + tmp

if __name__ == '__main__':
    input=torch.randn(1,16,3,3)
    gc = ContextLayer(16)
    output=gc(input)
    print(output.shape)