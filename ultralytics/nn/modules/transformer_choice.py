import torch
import torch.nn as nn

class Tc(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.wq = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim // 2, bias=qkv_bias)


        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  
        self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  



        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        tmp = x

        a1 = self.diconv(x)
        a2 = self.dwconv(x)

        a1 = a1.permute(0, 3, 2, 1)
        a1 = self.wq(a1)
        a1 = a1.permute(0, 3, 2, 1)


        a2 = a2.permute(0, 3, 2, 1)
        a2 = self.wk(a2)
        a2 = a2.permute(0, 3, 2, 1)


        attn = torch.cat([a1, a2], dim=1)


        avg = torch.mean(attn, dim=1, keepdim=True)
        avg = avg.softmax(dim=-1)

        max, _ = torch.max(attn, dim=1, keepdim=True)
        max = max.softmax(dim=-1)


        n_a1 = avg * a1
        n_a2 = max * a2


        n_attn = torch.cat([n_a1, n_a2], dim=1)


        x = self.proj(n_attn)

        x = tmp * x
        return x



if __name__ == '__main__':
    model = Tc(32)
    print(model)
    X = torch.ones(32, 32, 64, 64)
    Y = model(X)
    print(Y.shape)