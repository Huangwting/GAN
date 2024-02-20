import torch
from torch import nn

data_channels = 3 #RGB face

# 鉴别器
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminator = nn.Sequential(nn.Conv2d(in_channels=data_channels, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Linear(256 * 2 * 2, 1)
        self.out_ac = nn.Sigmoid()

    def forward(self, image):
        out_d = self.discriminator(image)
        out_d = out_d.view(-1, 256 * 2 * 2)
        return self.out_ac(self.linear(out_d))

    @staticmethod
    def weights_init(layer):
        layer_class_name = layer.__class__.__name__
        if 'Conv' in layer_class_name:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in layer_class_name:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.normal_(layer.bias.data, 0.)