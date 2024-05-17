import torch
from torch import nn
from torch.nn import functional as F


################################################################################
################################################################################
################################################################################
### EMBEDDINGS

### I feel like this could have been a function instead of a class but whatever
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
        omega = torch.arange(1, self.C//2+1)
        t = t.unsqueeze(1)
        p1 = (omega * t).sin()
        p2 = (omega * t).cos()
        t_emb = torch.concatenate((p1, p2), 1)
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
        ### As above, reshaped for convenience
        tensor = self.model(genre)[:, ..., None, None]
        return self.model(genre)[:, ..., None, None]



################################################################################
################################################################################
################################################################################
### SUBARCHITECTURES


class UNet(nn.Module):
    """
    Implementation of the UNet class for prediction of reconstruction of x0.
    Substitutes RecoverX0 written before. Takes xt as input and tries to
    recover x0.
    """
    def __init__(self):
        super().__init__()

        ### Define model blocks
        self.fatten = nn.Conv2d(16, 64, kernel_size=3,
                                stride=1, padding=1)
        self.down1 = Down(64, 128, 80, 2048)
        self.down2 = Down(128, 256, 40, 1024)
        self.down3 = Down(256, 512, 20, 512)
        self.up1 = Up(512, 256, 10, 256)
        self.up2 = Up(256, 128, 20, 512)
        self.up3 = Up(128, 64, 40, 1024)
        self.thin = nn.Conv2d(64, 1, kernel_size=3,
                              stride=1, padding=1)

        ### Define time embeddings
        self.t1 = TimeEmbd(64)
        self.t2 = TimeEmbd(128)
        self.t3 = TimeEmbd(256)
        self.t4 = TimeEmbd(512)

        ### Define genre embedding
        self.g1 = GenreEmbd(8, 64)
        self.g2 = GenreEmbd(8, 128)
        self.g3 = GenreEmbd(8, 256)
        self.g4 = GenreEmbd(8, 512)

    def forward(self, x, genre, t):
        x = self.fatten(x)
        d1 = self.down1(x + self.t1(t) + self.g1(genre))
        #print(f'\tD1: {d1.shape}')
        d2 = self.down2(d1 + self.t2(t) + self.g2(genre))
        #print(f'\tD2: {d2.shape}')
        d3 = self.down3(d2 + self.t3(t) + self.g3(genre))
        #print(f'\tD3: {d3.shape}')
        u = self.up1(d3 + self.t4(t) + self.g4(genre))
        #print(f'\tU1: {u.shape}')
        u = self.up2(u + d2 + self.t3(t) + self.g3(genre))
        #print(f'\tU2: {u.shape}')
        u = self.up3(u + d1 + self.t2(t) + self.g2(genre))
        #print(f'\tU3: {u.shape}')
        u = self.thin(u + self.t1(t) + self.g1(genre))
        #print(f'\tTh: {u.shape}')
        return u


class Residual(nn.Module):
    """
    Single residual block inside a UNet Block. Its input is x, and its output
    is the processing of x through model(x) plus x itself.
    """
    def __init__(self, in_ch, out_ch, height, width):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm((height, width)),
                                   nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                             stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                             stride=1, padding=1),
                                   )

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
    def __init__(self, up:bool):
        super().__init__()
        self.up = up

    def _generate_model(self, in_ch, out_ch, height, width):
        """
        Generate the appropriate model depending on the value of the up
        flag.
        """
        if self.up:
            model = nn.Sequential(Residual(in_ch, in_ch, height, width),
                                  Residual(in_ch, in_ch, height, width),
                                  Residual(in_ch, in_ch, height, width),
                                  nn.LayerNorm((height, width)),
                                  nn.Upsample([height*2, width*2],
                                              mode='nearest'),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                            stride=1, padding=1),
                                  nn.LayerNorm((height*2, width*2))
                                  )


        else:
            model = nn.Sequential(Residual(in_ch, in_ch, height, width),
                                  Residual(in_ch, in_ch, height, width),
                                  Residual(in_ch, in_ch, height, width),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                            stride=2, padding=1),
                                  )
        return model

    def forward(self, x):
        pass


class Down(_UNetBlock):
    """
    Upsampling version of the UNet Block.
    """
    def __init__(self, in_channels, out_channels, height, width):
        super().__init__(up=False)
        self.model = super()._generate_model(in_channels, out_channels,
                                             height, width)

    def forward(self, x):
        x = self.model(x)
        return x


class Up(_UNetBlock):
    """
    Upsampling version of the UNet Block.
    """
    def __init__(self, in_channels, out_channels, height, width):
        super().__init__(up=True)
        self.model = super()._generate_model(in_channels, out_channels,
                                             height, width)

    def forward(self, x):
        x = self.model(x)
        return x



################################################################################
################################################################################
################################################################################
### DIFFUSION

class Diffusion(nn.Module):
    """
    Doc...
    """
    def __init__(self, steps=100, epsilon=0.01):
        super().__init__()
        self.steps = steps
        self.tspan = torch.linspace(0, 1, steps)
        self.epsilon = torch.tensor(epsilon)
        #self.up = nn.Sequential([Up(...) for ii in range(depth)])
        self.up = UNet()

    def noise(self, x):
        """
        The whole forward steps for the noise.
        """
        z = torch.randn_like(x)
        #mu = torch.prod(torch.tensor([self.mu(t) for t in self.tspan]))
        mu = self.alpha_bar()
        #var = torch.prod(torch.tensor([self.var(t) for t in self.tspan]))
        var = 1 - self.alpha_bar()
        x_noisy = self.mu(1)*x + self.var(1)*z
        return x_noisy

    def alpha(self, t):
        return 1 - 0.9*t
        #return torch.cos(torch.acos(torch.sqrt(self.epsilon))*t)**2

    def alpha_bar(self, end=None):
        if end is None:
            end = self.tspan[-1] + 1
        return torch.tensor([self.alpha(t) for t in self.tspan
                             if t <= end]).prod()

    def mu(self, t_final):
        return (torch.sqrt(self.epsilon).arccos() * t_final).cos() ** 2
        #return self.alpha(t_final).sqrt()

    def var(self, t_final):
        return (1 - t_final + (self.epsilon.pow(2))).sqrt()
        #return (1.01 - self.alpha(t_final)).sqrt()

    def forward(self, x, y):
        #genre = torch.rand_like(x)[:, 1, :, :]
        #genre.copy_(x[:, 1, :, :])

        ### All the way down in one step for the noisy
        x_noisy = self.noise(x)
        print(f'\t\tNoisy mean: {x_noisy.mean().item()}')
        L_T = KS(x_noisy, torch.randn_like(x_noisy))
        L = L_T

        ### Step by step up for the denoising
        #print(self.tspan)
        for t in torch.linspace(1, 0, self.steps)[:-1]:
            print(f'\n\tDepth: {t}')
            pred = self.up(x_noisy, y, t)
            e = torch.randn_like(pred)
            loss = (pred - e).mean()**2
            print(f'\t\tLoss: {loss.item()}')
            L += loss
            mu = self.mu_up(x_noisy, pred, t)
            var = self.var_up(t)
            x_noisy = mu + torch.randn_like(x_noisy)*var
            print(f'\t\tNoisy mean: {x_noisy.mean().item()}')

        loss = (x - x_noisy).mean()**2
        print(loss)
        L += loss
        return x_noisy, L

    def mu_up(self, xt, x0, t):
        #alpha_bar = self.alpha_bar(t)
        #alpha_bar_prev = self.alpha_bar(t-1)
        #alpha = self.alpha(t)
        alpha_bar = self.mu(t).pow(2)
        alpha_bar_prev = self.mu(t - 1).pow(2)
        alpha = alpha_bar / alpha_bar_prev
        summand1 = ((alpha.sqrt() * (1 - alpha_bar_prev) * xt) /
                    (1 - alpha_bar))
        summand2 = ((alpha_bar_prev.sqrt() * (1 - alpha) * x0) /
                    (1 - alpha_bar))
        print(f'\t\t\tmu: {(summand1 + summand2).mean()}')
        return summand1 + summand2

    def var_up(self, t):
        #alpha = self.alpha(t)
        #alpha_bar_prev = self.alpha_bar(t-1)
        #alpha_bar = self.alpha_bar(t)
        alpha_bar = self.mu(t).pow(2)
        alpha_bar_prev = self.mu(t - 1).pow(2)
        alpha = alpha_bar / alpha_bar_prev
        #print(f'alpha_bar: {alpha_bar}\nalpha_bar_p: {alpha_bar_prev}\nalpha: {alpha}')
        var_sq = (1 - alpha)*(1 - alpha_bar_prev) / (1 - alpha_bar)
        print(f'\t\t\tvar: {var_sq}')
        return var_sq


def KS(a, b):
    """
    Function...
    """
    return 1





################################################################################
################################################################################
################################################################################
### IMPLEMENTATION

if __name__ == '__main__':
    #model = GenreEmbd(3, 2)
    #print(model(torch.tensor([1, 0, 2])).shape)
    #print(model(torch.tensor([1])).shape)
    #exit()

    x = torch.randn(4, 1, 80, 2048)
    y = torch.randint(0, 7, [4, ])
    print(y)
    print(f'In: {x.shape}')
    #model = UNet()
    #out = model(x, y, torch.tensor(0.5))
    #print(f'Out: {out.shape}')

    D = Diffusion(9)
    desc = 'Model with {} parameters and {} trainable parameters'
    print(desc.format(sum(p.numel() for p in D.parameters()),
                      sum(p.numel() for p in D.parameters() if p.requires_grad)))
    out = D(x, y)
    print(f'Final: {out[0].shape} with error {out[1]}')

    #print(x)
    #print(out[0])
