from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import torch
from model import UNet
from dataset import MelSpectrogramDataset


def mae_loss(predicted_mel, target_mel):
    return torch.mean(torch.abs(predicted_mel - target_mel))


model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=16)

dataset = MelSpectrogramDataset("data")
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_epochs = 500
for epoch in range(num_epochs):
    for mel_spectrogram in data_loader:
        noise_shape = mel_spectrogram.shape

        noise = torch.randn(noise_shape, dtype=torch.float)
        noisy_mel= scheduler.add_noise(mel_spectrogram, noise.float(), timesteps=scheduler.timesteps)

        optimizer.zero_grad()
        predicted_mel = model(noisy_mel)
        loss = mae_loss(predicted_mel, mel_spectrogram)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "mel_spectrogram_model.pth")