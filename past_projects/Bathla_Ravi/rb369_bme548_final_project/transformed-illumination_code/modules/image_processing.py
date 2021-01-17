import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    Just a plain old-cnn + MLP
    """

    def __init__(self, image_size=28, mlp_size=64, channels_in=1):
        super().__init__()
        self.mlp_size = mlp_size
        # following a simple conv conv MP, conv conv MP, MLP --> out
        self.convolutional_layers = nn.Sequential(
            # 28 x 28 x 1
            nn.Conv2d(in_channels=channels_in, out_channels=16, kernel_size=3, padding=0),
            nn.ReLU(),
            # 26 x 26 x 16
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0),
            nn.ReLU(),
            # 24 x 24 x 16
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=3),
            # 8 x 8 x 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            # 6 x 6 x 32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            # 4 x 4 x 32
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2),
            # 2 x 2 x 32
        )
        self.out_features = int(((image_size - 2 - 2) // 3 - 2 - 2) // 2) ** 2 * 32
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=self.out_features, out_features=self.mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=self.mlp_size, out_features=self.mlp_size),
            # nn.ReLU()
        )

    def forward(self, x):
        # x is input image: (B x H x W)
        # output is: (B x mlp_size)
        batch_size = x.shape[0]
        conv_out = self.convolutional_layers(x)
        mlp_in = conv_out.view(-1, self.out_features)
        mlp_out = self.fully_connected_layers(mlp_in)
        return mlp_out.view(batch_size, 1, -1)
