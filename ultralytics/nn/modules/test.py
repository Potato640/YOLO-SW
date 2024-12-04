import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()

        self.soft = nn.Softmax()

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)

        attn = torch.mean(x, dim=1, keepdim=True)
        avg = attn.softmax(dim=-1)
        return x + (x * avg)

if __name__ == '__main__':
    model = SE(32)
    print(model)
    X = torch.ones(32, 32, 64, 64)
    Y = model(X)
    print(Y.shape)