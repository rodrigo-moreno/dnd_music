"""
    Script to train out diffusion model
"""
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import torch
from model import UNet, pad_to_power_of_two
from dataset import MelSpectrogramDataset
from matplotlib import pyplot as plt


def mae_loss(pred_mel, target_mel):
    """
    Function to compute mean absolute error on the predicted mel diagram
    """
    return torch.mean(torch.abs(pred_mel - target_mel))


model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=16)

dataset = MelSpectrogramDataset("data")
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

NUM_EPOCHS = 5
train_loss = []
bad_ones = []
for epoch in range(NUM_EPOCHS):
    JJ = 0
    for mel_spectrogram in data_loader:
        JJ += 1
        mel_spectrogram = pad_to_power_of_two(mel_spectrogram, 2048)
        name = dataset.file_paths[16 * epoch + JJ]
        print(f"mel espectogram number: {JJ} in epoch {epoch+1}")
        print(f"shape: {mel_spectrogram.shape}")
        print(name)
        if mel_spectrogram.shape[0] != 16:
            print(name + "is weird!!")
            bad_ones.append(name)
        else:
            noise_shape = mel_spectrogram.shape
            noise = torch.randn(noise_shape, dtype=torch.float)
            noisy_mel = scheduler.add_noise(
                mel_spectrogram, noise.float(), timesteps=scheduler.timesteps
            )
            optimizer.zero_grad()
            predicted_mel = model(noisy_mel)
            loss = mae_loss(predicted_mel, mel_spectrogram)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if JJ % 20 == 0:
                torch.save(
                    model.state_dict(), f".train_results/steps/mel_spectrogram{JJ}_{epoch}.pth"
                )

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "train_results/mel_spectrogram_model.pth")

weird_shapes = open(".train_results/weird_shapes.txt", "w")
for bad in bad_ones:
    weird_shapes.write(bad+'\n')

plt.figure()
plt.title("Training Loss")
plt.plot(train_loss)
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.show()
