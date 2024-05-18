import os
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import UNet
from dataset import MelSpectrogramDataset
from diffusion import DiffusionModel
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# Global variables for model and device
global_model = None
global_device = None


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
    num_epochs = 10
    batch_size = 16
    num_timesteps = 1000

    # Load the dataset and create a dataloader
    dataset = MelSpectrogramDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, optimizer, scheduler, and diffusion model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        torch.save(model.state_dict(), f'steps/model_{epoch + 1}.pth')

        # Log the loss to wandb
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})
        wandb.watch(model, log="all")

        # Step the learning rate scheduler
        scheduler.step()

    torch.save(model.state_dict(), "diffusion_model.pth")
    return residual_images, model, diffusion


def evaluate_model(images, model, diffusion, device):
    """
    Evaluate the model on a set of images.

    Args:
        images (torch.Tensor): Batch of images.
        model (UNet): Trained UNet model.
        diffusion (DiffusionModel): Diffusion model.
        device (torch.device): The device to use for evaluation.

    Returns:
        float: Evaluation loss.
    """
    images = images.to(device)
    genre = images.genre

    t = torch.randint(0, diffusion.num_timesteps, (images.size(0),), device=device).long()
    noise = torch.randn_like(images).to(device)

    x_t = diffusion.q_sample(images, t, noise)
    predicted_noise = model(x_t, t, genre)
    loss = torch.mean((noise - predicted_noise) ** 2)

    return loss.item()


def test_residual_images(residual_images, model, device, num_timesteps=1000):
    """
    Test the model using residual images obtained during training.

    Args:
        residual_images (list): List of residual images.
        model (UNet): Trained UNet model.
        device (torch.device): The device to use for testing.
        num_timesteps (int): Number of timesteps for the diffusion model.
    """
    diffusion = DiffusionModel(num_timesteps=num_timesteps)
    model.eval()

    total_loss = 0.0
    num_batches = len(residual_images)

    with torch.no_grad():
        for i, sample_images in enumerate(residual_images):
            sample_images = sample_images.to(device)
            genre = sample_images.genre
            for t in range(diffusion.num_timesteps - 1, -1, -1):
                sample_images = diffusion.p_sample(model, sample_images, t, genre)
            save_image(sample_images, f'images/residual_sample_{i}.png')
            wandb.log({"residual_sample_images": wandb.Image(f'images/residual_sample_{i}.png')})

            loss = evaluate_model(sample_images, model, diffusion, device)
            total_loss += loss
            print(f'Batch {i + 1}/{num_batches}, Loss: {loss:.4f}')

    avg_loss = total_loss / num_batches
    print(f'Average Evaluation Loss: {avg_loss:.4f}')
    wandb.log({"avg_evaluation_loss": avg_loss})


def test_model_with_genre(genre, num_timesteps=1000):
    """
    Test the model by generating images from noise, conditioned on a given genre.

    Args:
        genre (int): Genre label to condition the generation.
        num_timesteps (int): Number of timesteps for the diffusion model.
    """
    global global_model, global_device
    diffusion = DiffusionModel(num_timesteps=num_timesteps)
    global_model.eval()
    with torch.no_grad():
        sample_images = torch.randn(1, 1, 256, 256).to(global_device)
        for t in range(diffusion.num_timesteps - 1, -1, -1):
            sample_images = diffusion.p_sample(global_model, sample_images, t, genre)
        save_image(sample_images, 'images/generated_sample.png')
        wandb.log({"generated_sample_images": wandb.Image('images/generated_sample.png')})


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
    global global_model, global_device

    # Path to personal wandb key
    path_to_key = "../../wandb_key.txt"
    key = open(path_to_key, "r").read().split("\n")[0]
    wandb.login(key=key)

    os.makedirs('images', exist_ok=True)
    os.makedirs('steps', exist_ok=True)

    # Configure wandb sweep
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'learning_rate': {
                'values': [1e-3, 1e-4, 1e-5]
            },
            'weight_decay': {
                'values': [1e-4, 1e-5, 1e-6]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="dnd_music diffusion model")
    wandb.agent(sweep_id, function=sweep_train)

    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_images, model, diffusion = train_model(global_device)

    # Set the global model after training
    global_model = model

    # Test with residual images obtained during training
    test_residual_images(residual_images, model, global_device, 1000)

    # Test with a specific genre
    genre = 1
    test_model_with_genre(genre, 1000)


if __name__ == "__main__":
    main()
