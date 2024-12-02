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

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  #Sigmoid70.9  softmax mid72.5
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #Sigmoid72.5
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #Sigmoid69.4
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #softmax 75.2  mid68.9
        self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #softmax 72.2  mid68.6
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #softmax 68.6 mid65.5
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #softmax 69.7 mid70.1
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)  #softmax 67.5 mid71.5
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #softmax 71.9 mid67.4
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=dim)  #softmax 74.6 mid72
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim)  #softmax mid68.2
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # softmax mid72.4
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  #

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim)  # softmax mid73
        # self.diconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  #

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        tmp = x

        a1 = self.diconv(x)
        a2 = self.dwconv(x)

        a1 = a1.permute(0, 3, 2, 1)
        a1 = self.wq(a1)
        a1 = a1.permute(0, 3, 2, 1)
        # print("a1", a1.shape)

        a2 = a2.permute(0, 3, 2, 1)
        a2 = self.wk(a2)
        a2 = a2.permute(0, 3, 2, 1)
        # print("a2", a2.shape)

        attn = torch.cat([a1, a2], dim=1)
        # print(attn.shape)

        avg = torch.mean(attn, dim=1, keepdim=True)
        avg = avg.softmax(dim=-1)
        # avg = self.sigmoid(avg)
        # print("avg",avg.shape)
        max, _ = torch.max(attn, dim=1, keepdim=True)
        max = max.softmax(dim=-1)
        # max = self.sigmoid(max)
        # print("max", max.shape)

        n_a1 = avg * a1
        n_a2 = max * a2
        # print(n_a2.shape)

        n_attn = torch.cat([n_a1, n_a2], dim=1)
        # print(n_attn.shape)

        x = self.proj(n_attn)

        x = tmp * x
        return x



if __name__ == '__main__':
    model = Tc(32)
    print(model)
    X = torch.ones(32, 32, 64, 64)
    Y = model(X)
    print(Y.shape)