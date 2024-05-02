import torch
from torch import nn
import torch.nn.functional as F

def pad_to_power_of_two(x, target_dim=2048):
    _, _, h, w = x.size()
    pad_h = target_dim - h if h < target_dim else 0
    pad_w = target_dim - w if w < target_dim else 0
    # Asegúrate de que el padding sea simétrico
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(x, padding, "constant", 0)  # padding (left, right, top, bottom)

# Ejemplo de uso
x = torch.randn(16, 1, 80, 1723)
x_padded = pad_to_power_of_two(x, 2048)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.down = down
        self.use_dropout = use_dropout
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def forward(self, x):
        if self.down:
            x = self.conv(x)
            if self.use_dropout:
                x = self.dropout(x)
        else:
            x = self.up(x)
            x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UNetBlock(1, 64)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512, use_dropout=True)

        self.up1 = UNetBlock(512, 256, down=False)
        self.up2 = UNetBlock(256, 128, down=False)
        self.up3 = UNetBlock(128, 64, down=False)
        self.up4 = UNetBlock(64, 32, down=False)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        pad_to_power_of_two(x)
        x1 = self.down1(x);print(x1.shape)
        x2 = self.down2(x1);print(x2.shape)
        x3 = self.down3(x2);print(x3.shape)
        x4 = self.down4(x3);print(x4.shape)

        x = self.up1(x4);print(x.shape)
        x = self.up2(x);print(x.shape)
        x = self.up3(x);print(x.shape)
        x = self.up4(x);print(x.shape)
        x = self.final_conv(x);print(x.shape)
        return x

