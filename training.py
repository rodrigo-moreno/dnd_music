"""
Script to train the diffusion model.
Parameters are adjusted in an ideal case. See report for actual training parameters.
"""
import os
import torch
import wandb
from torch.utils.data import DataLoader
from model import Diffusion
from dataset import MelSpectrogramDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def train_model(device, learning_rate, weight_decay):
    """
    Train the UNet model with the specified device.

    Args:
        device (torch.device): The device to use for training.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
    """
    image_dir = "data"
    num_epochs = 15
    batch_size = 16
    num_timesteps = 1000

    # Load the dataset and create a dataloader
    dataset = MelSpectrogramDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, optimizer, scheduler, and diffusion model
    model = Diffusion(num_timesteps).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # LR scheduler to improve training results
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize wandb for logging
    wandb.init(project="dnd_music diffusion model")

    residual_images = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, label in tqdm(dataloader):
            if images.shape[0] != batch_size:
                residual_images.append(images)
                print(f"Skipping batch with size {images.shape[0]}")
                continue

            images = images.to(device)
            label = label.to(device)

            # Forward pass
            out, loss = model(images, label)
            optimizer.zero_grad()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        # Save the model checkpoint
        torch.save(model.state_dict(), f"steps/model_{epoch+1}_R.pth")

        # Log the loss to wandb
        wandb.log({"loss": epoch_loss})
        wandb.watch(model, log="all")

        # Step the scheduler
        scheduler.step()

    # Save the final model state
    torch.save(model.state_dict(), f"final_state_R.pth")


def main():
    """
    Main function to set up directories, parameters, and initiate training.
    Parameters were set during previous assessment.
    """
    os.makedirs("steps", exist_ok=True)

    lr = 1e-3
    wd = 1e-5
    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(global_device, lr, wd)


if __name__ == "__main__":
    main()
