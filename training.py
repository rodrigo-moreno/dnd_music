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

    Returns:
        list: List of residual images.
        UNet: Trained UNet model.
        DiffusionModel: Diffusion model used during training.
    """
    image_dir = "data"
    num_epochs = 15
    batch_size = 4
    num_timesteps = 10

    # Load the dataset and create a dataloader
    dataset = MelSpectrogramDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4)

    # Initialize the model, optimizer, scheduler, and diffusion model
    model = Diffusion(num_timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay
                                )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize wandb
    wandb.init(project="dnd_music diffusion model")

    residual_images = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, label in tqdm(dataloader):
            if images.shape[0] != batch_size:
                residual_images.append(images)
                print(f'Skipping batch with size {images.shape[0]}')
                continue

            images = images.to(device)

            ### This model is now directly the diffusion model
            out, loss = model(images, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        torch.save(model.state_dict(), f'steps/model_{epoch+1}_R.pth')

        wandb.log({'loss': epoch_loss})
        wandb.watch(model, log='all')

        scheduler.step()
    torch.save(model.state_dict(), f'final_state_R.')


def main():
    os.makedirs('steps', exist_ok=True)

    lr = 1e-3
    wd = 1e-5
    global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(global_device, lr, wd)


if __name__ == '__main__':
    main()
