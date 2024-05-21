import torch
from torch import nn


class TimeEmbd(nn.Module):
    """
    Do the time embedding changing from a scalar value to a vector of size C
    (the amount of channels). The resulting vector will be added across the
    CHANNELS of the representation of the image.

    Input:
    - channels: integer stating the amount of channels in x.

    Output:
    - vector in R^C represented in [1, C, 1, 1] for convenience.
    """

    def __init__(self, channels):
        super().__init__()
        self.C = channels

    def forward(self, t):
        omega = torch.arange(1, self.C // 2 + 1)
        t = t.unsqueeze(1)
        p1 = (omega * t).sin()
        p2 = (omega * t).cos()
        t_emb = torch.cat((p1, p2), 1)
        return t_emb.unsqueeze(-1).unsqueeze(-1)


class GenreEmbd(nn.Module):
    """
    Do the genre embedding for the network. Contrary to TimeEmbd, this embedding
    IS learnable.
    """

    def __init__(self, genres, channels):
        super().__init__()
        self.model = nn.Embedding(genres, channels)

    def forward(self, genre):
        # Reshaped for convenience
        return self.model(genre)[:, ..., None, None]


class UNet(nn.Module):
    """
    Implementation of the UNet class for prediction of reconstruction of x0.
    Substitutes RecoverX0 written before. Takes xt as input and tries to
    recover x0.
    """

    def __init__(self):
        super().__init__()

        # Define model blocks
        self.fatten = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(64, 128, 80, 2048)
        self.down2 = Down(128, 256, 40, 1024)
        self.down3 = Down(256, 512, 20, 512)
        self.up1 = Up(512, 256, 10, 256)
        self.up2 = Up(256, 128, 20, 512)
        self.up3 = Up(128, 64, 40, 1024)
        self.thin = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # Define time embeddings
        self.t1 = TimeEmbd(64)
        self.t2 = TimeEmbd(128)
        self.t3 = TimeEmbd(256)
        self.t4 = TimeEmbd(512)

        # Define genre embedding
        self.g1 = GenreEmbd(8, 64)
        self.g2 = GenreEmbd(8, 128)
        self.g3 = GenreEmbd(8, 256)
        self.g4 = GenreEmbd(8, 512)

        # Apply He initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, genre, t):
        x = self.fatten(x)
        d1 = self.down1(x + self.t1(t) + self.g1(genre))
        d2 = self.down2(d1 + self.t2(t) + self.g2(genre))
        d3 = self.down3(d2 + self.t3(t) + self.g3(genre))
        u = self.up1(d3 + self.t4(t) + self.g4(genre))
        u = self.up2(u + d2 + self.t3(t) + self.g3(genre))
        u = self.up3(u + d1 + self.t2(t) + self.g2(genre))
        u = self.thin(u + self.t1(t) + self.g1(genre))
        return u


class Residual(nn.Module):
    """
    Single residual block inside a UNet Block. Its input is x, and its output
    is the processing of x through model(x) plus x itself.
    """

    def __init__(self, in_ch, out_ch, height, width):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm((height, width)),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        # Apply He initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.model(x) + x


class _UNetBlock(nn.Module):
    """
    Parent class for either the downwards or upwards UNet block.
    The children classes Up and Down are a fixed instantiation of this parent
    class that either increase dimensions and reduce channels (Up), or reduce
    dimensions and increase channels (Down).

    Whether it is an Up or Down block is determined by the value self.up
    set at instantiation.
    """

    def __init__(self, up: bool):
        super().__init__()
        self.up = up

    def _generate_model(self, in_ch, out_ch, height, width):
        """
        Generate the appropriate model depending on the value of the up flag.
        """
        if self.up:
            model = nn.Sequential(
                Residual(in_ch, in_ch, height, width),
                Residual(in_ch, in_ch, height, width),
                Residual(in_ch, in_ch, height, width),
                nn.LayerNorm((height, width)),
                nn.Upsample([height * 2, width * 2], mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm((height * 2, width * 2)),
            )
        else:
            model = nn.Sequential(
                Residual(in_ch, in_ch, height, width),
                Residual(in_ch, in_ch, height, width),
                Residual(in_ch, in_ch, height, width),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            )
        return model

    def forward(self, x):
        pass


class Down(_UNetBlock):
    """
    Downsampling version of the UNet Block
    """

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__(up=False)
        self.model = super()._generate_model(in_channels, out_channels, height, width)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.model(x)
        return x


class Up(_UNetBlock):
    """
    Upsampling version of the UNet Block.
    """

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__(up=True)
        self.model = super()._generate_model(in_channels, out_channels, height, width)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.model(x)
        return x
