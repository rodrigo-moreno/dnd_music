"""
Previous training script. Even if could not get reliable results, it helped in finding training params.
"""
import os
import torch
import wandb
from torch.utils.data import DataLoader
from model import UNet
from dataset import MelSpectrogramDataset
from diffusion_previous import DiffusionModel
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def train_model(device, learning_rate, weight_decay):
    """
    Train the UNet model with the specified device.

    Args:
        device (torch.device): The device to use for training.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

    Returns:
        list: List of residual images.
        UNet: Trained UNet model.
        DiffusionModel: Diffusion model used during training.
    """
    image_dir = "data"
    num_epochs = 15
    batch_size = 16
    num_timesteps = 1000

    # Load the dataset and create a dataloader
    dataset = MelSpectrogramDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, optimizer, scheduler, and diffusion model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    diffusion = DiffusionModel(num_timesteps=num_timesteps)

    # Initialize wandb
    wandb.init(project="dnd_music diffusion model")

    residual_images = []

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, label in tqdm(dataloader):
            if images.size(0) != batch_size:
                residual_images.append(images)
                print(f"Skipping batch with size {images.size(0)}")
                continue

            images = images.to(device)

            t = torch.randint(0, num_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images).to(device)

            optimizer.zero_grad()
            x_t = diffusion.q_sample(images, t, noise)
            predicted_noise = model(x_t, label.int(), t)
            loss = torch.mean((noise - predicted_noise) ** 2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), f"steps/model_{epoch + 1}.15.pth")

        # Log the loss to wandb
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})
        wandb.watch(model, log="all")

        # Step the learning rate scheduler
        scheduler.step()

    torch.save(model.state_dict(), f"diffusion_model_{learning_rate}{weight_decay}.pth")


def sweep_train():
    """
    Function to train the model with parameters from wandb sweep.
    """
    with wandb.init() as run:
        config = run.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_model(device, config.learning_rate, config.weight_decay)


def main():
    """
    Main function to set up the environment and start training and testing.
    """
    path_to_key = "../../wandb_key.txt"
    key = open(path_to_key, "r").read().split("\n")[0]
    wandb.login(key=key)

    os.makedirs("steps", exist_ok=True)

    # Configure wandb sweep
    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {"values": [1e-4, 1e-5]},
            "weight_decay": {"values": [1e-5, 1e-6]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="dnd_music diffusion model")
    wandb.agent(sweep_id, function=sweep_train)

    lr = 1e-3
    wd = 1e-5
    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(global_device, lr, wd)


if __name__ == "__main__":
    main()
