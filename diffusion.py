import torch
from tqdm import tqdm
class DiffusionModel:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
        """
        Initialize the Diffusion Model.

        Args:
            beta_start (float): The starting value for beta.
            beta_end (float): The ending value for beta.
            num_timesteps (int): The number of timesteps in the diffusion process.
        """
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, axis=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]], axis=0)

    def q_sample(self, x_start, t, noise):
        """
        Sample from the forward diffusion process.

        Args:
            x_start (Tensor): The initial image tensor.
            t (Tensor): The current timestep.
            noise (Tensor): The noise to be added to the image.

        Returns:
            Tensor: The noisy image at timestep t.
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1).to(x_start.device)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1, 1).to(x_start.device)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def p_sample(self, model, x, t, t_index, genre):
        """
        Sample from the reverse diffusion process.

        Args:
            model (nn.Module): The neural network model (U-Net) predicting the noise.
            x (Tensor): The current image tensor.
            t (Tensor): The current timestep tensor.
            t_index (int): The index of the current timestep.

        Returns:
            Tensor: The denoised image at timestep t-1.
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t_index]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t_index]).view(-1, 1, 1, 1)
        posterior_variance = self.beta[t_index] * (1 - self.alpha_cumprod_prev[t_index]) / (1 - self.alpha_cumprod[t_index])

        # Calculate the predicted mean by the model
        model_mean = (x - sqrt_one_minus_alpha_cumprod * model(x, genre, t)) / sqrt_alpha_cumprod

        # Generate noise if it is not the final timestep
        noise = torch.randn_like(x) if t_index > 0 else torch.zeros_like(x)

        # Sample x_{t-1} from the Gaussian distribution
        return model_mean + torch.sqrt(posterior_variance).view(-1, 1, 1, 1) * noise

    def sample(self,model, shape, genre, timesteps=1000):
        x = torch.randn(shape)
        for t_index in tqdm(reversed(range(timesteps)), desc=f"Generating sample with {timesteps} diffusion steps"):
            t = torch.tensor([t_index])
            x = self.p_sample(model,x, t, t_index, genre)
        return x
