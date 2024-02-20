from torch import nn

data_channels = 3 #RGB face
z_dim = 100

# 处理生成器
class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection_layer = nn.Linear(z_dim, 4*4*1024)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=data_channels, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.Tanh(),
        )
    def forward(self, latent_Z):
        z = self.projection_layer(latent_Z)
        z_projected = z.view(-1, 1024, 4, 4)
        return self.generator(z_projected)

    @staticmethod
    def weights_init(layer):
        layer_class_name = layer.__class__.__name__
        if 'Conv' in layer_class_name:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in layer_class_name:
            nn.init.normal_(layer.weight.data, 1.0,0.02)
            nn.init.normal_(layer.bias.data,0.)
