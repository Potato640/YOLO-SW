import torch
import torch.nn as nn

class CAM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim * 3)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.diconv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)

        self.avg = nn.AvgPool2d(7, 1, 3)
        self.max = nn.MaxPool2d(7, 1, 3)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim) #72.5
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) #71.1
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim) # reverse avg and max 67.6

        self.proj = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        x1 = self.diconv(self.conv2(self.act(self.norm(self.conv1(self.avg(x))))))
        # x1 = x * x1
        # x1 = x + x1
        # print(x1.shape)
        x2 = self.sig(self.conv(self.dwconv(self.max(self.conv(x)))))
        # x2 = x * x2 #65
        # x2 = x + x2 #
        # print(x2.shape)
        x3 = torch.cat((x1, x2), 1)
        return self.proj(x3)

if __name__ == '__main__':
    model = CAM(32)
    print(model)
    X = torch.ones(32, 32, 64, 64)
    Y = model(X)
    print(Y.shape)