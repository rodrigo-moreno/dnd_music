import torch
from torch import nn


# EMBEDDINGS


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
        """
        Forward pass for the time embedding.

        Args:
        - t (torch.Tensor): Scalar time value.

        Returns:
        - torch.Tensor: Time embedding of shape [1, C, 1, 1].
        """
        omega = torch.arange(1, self.C // 2 + 1).to(t.device)
        t = t.unsqueeze(0)
        p1 = (omega * t).sin()
        p2 = (omega * t).cos()
        t_emb = torch.cat((p1, p2), 0)
        return t_emb.unsqueeze(-1).unsqueeze(-1)


class GenreEmbd(nn.Module):
    """
    Do the genre embedding for the network. Contrary to TimeEmbd, this embedding
    IS learnable.

    Args:
    - genres (int): Number of genres.
    - channels (int): Number of channels for embedding.
    """

    def __init__(self, genres, channels):
        super().__init__()
        self.model = nn.Embedding(genres, channels)

    def forward(self, genre):
        """
        Forward pass for the genre embedding.

        Args:
        - genre (torch.Tensor): Genre label.

        Returns:
        - torch.Tensor: Genre embedding of shape [1, C, 1, 1].
        """
        return self.model(genre)[:, ..., None, None]


# SUBARCHITECTURES


class UNet(nn.Module):
    """
    Implementation of the UNet class for prediction of reconstruction of x0.
    Substitutes RecoverX0 written before. Takes xt as input and tries to
    recover x0.
    """

    def __init__(self):
        super().__init__()

        # Define model blocks
        self.fatten = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
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

        # Apply He initialization of parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for convolutional layers.

        Args:
        - module (nn.Module): Layer module.
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, genre, t):
        """
        Forward pass for the UNet model.

        Args:
        - x (torch.Tensor): Input tensor.
        - genre (torch.Tensor): Genre embedding tensor.
        - t (torch.Tensor): Time embedding tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
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

    Args:
    - in_ch (int): Number of input channels.
    - out_ch (int): Number of output channels.
    - height (int): Height of the input tensor.
    - width (int): Width of the input tensor.
    """

    def __init__(self, in_ch, out_ch, height, width):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm((height, width)),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Forward pass for the Residual block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.model(x) + x


class _UNetBlock(nn.Module):
    """
    Parent class for either the downwards or upwards UNet block.
    The children classes Up and Down are a fixed instantiation of this parent
    class that either increase dimensions and reduce channels (Up), or reduce
    dimensions and increase channels (Down).

    Whether it is an Up or Down block is determined by the value self.up
    set at instantiation.

    Args:
    - up (bool): Boolean flag to determine the type of block.
    """

    def __init__(self, up: bool):
        super().__init__()
        self.up = up

    def _generate_model(self, in_ch, out_ch, height, width):
        """
        Generate the appropriate model depending on the value of the up
        flag.

        Args:
        - in_ch (int): Number of input channels.
        - out_ch (int): Number of output channels.
        - height (int): Height of the input tensor.
        - width (int): Width of the input tensor.

        Returns:
        - nn.Sequential: Generated model.
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
    Downsampling version of the UNet Block.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - height (int): Height of the input tensor.
    - width (int): Width of the input tensor.
    """

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__(up=False)
        self.model = super()._generate_model(in_channels, out_channels, height, width)

    def forward(self, x):
        """
        Forward pass for the Down block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.model(x)
        return x


class Up(_UNetBlock):
    """
    Upsampling version of the UNet Block.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - height (int): Height of the input tensor.
    - width (int): Width of the input tensor.
    """

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__(up=True)
        self.model = super()._generate_model(in_channels, out_channels, height, width)

    def forward(self, x):
        """
        Forward pass for the Up block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.model(x)
        return x


# DIFFUSION


class Diffusion(nn.Module):
    """
    Implementation of Diffusion model.

    Args:
    - steps (int): Number of diffusion steps.
    - epsilon (float): Epsilon value for noise.
    """

    def __init__(self, steps=100, epsilon=0.01):
        super().__init__()
        self.steps = steps
        self.tspan = torch.linspace(0, 1, steps)
        self.epsilon = torch.tensor(epsilon)
        self.up = UNet()
        self.loss = nn.MSELoss()

    def noise(self, x):
        """
        The whole forward steps for the noise.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Noisy tensor.
        """
        z = torch.randn_like(x)
        x_noisy = self._mu(1) * x + self._var(1) * z
        return x_noisy

    def alpha(self, t):
        """
        Alpha value for a given time step.

        Args:
        - t (float): Time step.

        Returns:
        - float: Alpha value.
        """
        return 1 - 0.9 * t

    def alpha_bar(self, end=None):
        """
        Alpha bar value for a given end time step.

        Args:
        - end (float): End time step.

        Returns:
        - float: Alpha bar value.
        """
        if end is None:
            end = self.tspan[-1] + 1
        return torch.tensor([self.alpha(t) for t in self.tspan if t <= end]).prod()

    def _mu(self, t_final):
        """
        Mu value for a given final time step.

        Args:
        - t_final (float): Final time step.

        Returns:
        - float: Mu value.
        """
        return (torch.sqrt(self.epsilon).arccos() * t_final).cos() ** 2

    def _var(self, t_final):
        """
        Variance value for a given final time step.

        Args:
        - t_final (float): Final time step.

        Returns:
        - float: Variance value.
        """
        return (1 - t_final + (self.epsilon.pow(2))).sqrt()

    def forward(self, x, y):
        """
        Forward pass for the Diffusion model.

        Args:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Genre embedding tensor.

        Returns:
        - torch.Tensor: Noisy tensor.
        - torch.Tensor: Loss value.
        """
        x_noisy = self.noise(x)
        L = None

        for t in torch.linspace(1, 0, self.steps)[:-1]:
            pred = self.up(x_noisy, y, t)
            loss = self.loss(pred, x)
            if L is None:
                L = loss
            else:
                L += loss
            mu = self.mu_up(x_noisy, pred, t)
            var = self.var_up(t)
            x_noisy = mu + torch.randn_like(x_noisy) * var

        return x_noisy, L

    def mu_up(self, xt, x0, t):
        """
        Mu update step for the diffusion process.

        Args:
        - xt (torch.Tensor): Noisy tensor.
        - x0 (torch.Tensor): Original tensor.
        - t (float): Time step.

        Returns:
        - torch.Tensor: Updated mu value.
        """
        alpha_bar = self._mu(t).pow(2)
        alpha_bar_prev = self._mu(t - 1).pow(2)
        alpha = alpha_bar / alpha_bar_prev
        summand1 = (alpha.sqrt() * (1 - alpha_bar_prev) * xt) / (1 - alpha_bar)
        summand2 = (alpha_bar_prev.sqrt() * (1 - alpha) * x0) / (1 - alpha_bar)

        return summand1 + summand2

    def var_up(self, t):
        """
        Variance update step for the diffusion process.

        Args:
        - t (float): Time step.

        Returns:
        - float: Updated variance value.
        """
        alpha_bar = self._mu(t).pow(2)
        alpha_bar_prev = self._mu(t - 1).pow(2)
        alpha = alpha_bar / alpha_bar_prev
        var_sq = (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)

        return var_sq

    @torch.no_grad()
    def sample(self, x, g):
        """
        Sampling function for the Diffusion model. Takes a random array and a
        genre determiner, and returns the final image.

        Args:
        - x (torch.Tensor): Input tensor.
        - g (int): Genre label.

        Returns:
        - torch.Tensor: Output tensor.
        """
        if not isinstance(g, torch.Tensor):
            g = torch.tensor([g])
        if 80 not in x.shape or 2048 not in x.shape:
            raise ValueError("Input tensor must have shape [1, 1, 80, 2048].")
        if len(x.shape) != 4:
            x = x.view(1, 1, 80, 2048)

        for t in torch.linspace(1, 0, self.steps).to(x.device)[:-1]:
            pred = self.up(x, g, t)
            mu = self.mu_up(x, pred, t)
            var = self.var_up(t)
            x_noisy = mu + torch.randn_like(x) * var
        return x_noisy


# IMPLEMENTATION
# This block covers the different tests performed during development

if __name__ == "__main__":
    # Check if Genre Embedding works
    if False:
        model = GenreEmbd(3, 2)
        g1 = torch.tensor([1])
        print(g1.shape)
        print(model(g1).shape)
        g2 = torch.tensor([0, 1, 2, 2])
        print(g2.shape)
        print(model(g2).shape)
        exit()

    # Check if Time Embedding works
    if False:
        model = TimeEmbd(3)
        t = torch.linspace(0, 1, 90)
        ts = t[3]
        print(ts.shape)
        print(model(ts).shape)
        exit()

    # Check if UNet works
    if False:
        x = torch.randn(4, 1, 80, 2048)
        y = torch.randint(0,7,[4,],)
        print(y)
        print(f"In: {x.shape}")
        model = UNet()
        out = model(x, y, torch.tensor(0.5))
        print(f"Out: {out.shape}")
        exit()

    # Check if Diffusion works
    if False:
        x = torch.randn(4, 1, 80, 2048)
        y = torch.randint(0,7,[4,],)
        print(y)
        print(f"In: {x.shape}")
        D = Diffusion(3)
        desc = "Model with {} parameters and {} trainable parameters"
        print(
            desc.format(
                sum(p.numel() for p in D.parameters()),
                sum(p.numel() for p in D.parameters() if p.requires_grad),
            )
        )
        out, error = D(x, y)
        print(f"Final: {out.shape} with error {error}")
        error.backward()
        print(next(iter(D.parameters())).grad.mean())
        exit()

    # Check if sampling works
    if True:
        model = Diffusion(10)
        if False:
            # Check if it works with the desired type of input
            x = torch.randn(1, 1, 80, 2048)
            g = torch.tensor([1])
            pred = model.sample(x, g)
            print(pred.shape)
            exit()
        if True:
            # Check if it works with weird inputs
            x = torch.randn(80, 2048)
            g = 1
            out = model.sample(x, g)
            print(out.mean())
