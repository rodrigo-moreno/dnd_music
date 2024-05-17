"""
Script to train our diffusion model
"""

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import wandb
from model import UNet
from dataset import MelSpectrogramDataset
from diffusion import DiffusionModel
from tqdm import tqdm


def diffusion_loss(model, x_0, t, noise, diffusion):
    """
    Calculate the loss between the true added and the predicted noise.

    Args:
        model: The neural network (U-Net) predicting the noise.
        x_0: The original image tensor.
        t: The timestep index tensor.
        noise: The noise added to the image.
        diffusion: The DiffusionModel instance.

    Returns:
        loss: The calculated loss.
    """
    x_t = diffusion.q_sample(x_0, t, noise)

    # Predict the noise using the model
    predicted_noise = model(x_t, t)

    # Calculate the loss (MSE between real and predicted noise)
    loss = torch.mean((noise - predicted_noise) ** 2)

    return loss


def train_model():
    image_dir = "data"
    num_epochs = 5
    batch_size = 16
    learning_rate = 1e-4
    num_timesteps = 1000

    # in_channels = 2       => parámetro a definir ¿?

    dataset = MelSpectrogramDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)  # modelo a definir

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    diffusion = DiffusionModel(num_timesteps=num_timesteps)

    # Initialize WandB
    wandb.init(project="dnd_music diffusion model")

    for epoch in range(num_epochs):
        epoch_loss = 0
        ii = 0
        for images, labels in tqdm(dataloader):
            ii += 1
            # print(f'iteration {ii} in epoch {epoch + 1}')
            images = images.to(device)
            t = torch.randint(0, num_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)

            # Calculate the diffusion loss using the DiffusionModel instance
            loss = diffusion_loss(model, images, t, noise, diffusion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Log the loss to WandB
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})
        wandb.watch(model, log="all")

        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                sample_images = torch.randn((batch_size, 1, 128, 128), device=device)
                for i in range(num_timesteps - 1, -1, -1):
                    t = torch.tensor([i], device=device).long()
                    sample_images = diffusion.p_sample(model, sample_images, t, i)
                save_image(sample_images, f'sample_{epoch + 1}.png')

                # Log generated images to WandB
                wandb.log({"sample_images": wandb.Image(f'sample_{epoch + 1}.png')})

    torch.save(model.state_dict(), "unet_diffusion_model.pth")


def main():
    # Path to personal wandb key
    path_to_key = "../../wandb_key.txt"
    key = open(path_to_key, "r").read().split("\n")[0]

    wandb.login(key=key)
    train_model()


if __name__ == "__main__":
    main()
