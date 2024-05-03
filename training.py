from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import torch
from model import UNet, pad_to_power_of_two
from dataset import MelSpectrogramDataset
from matplotlib import pyplot as plt
import os
import numpy as np
def mae_loss(predicted_mel, target_mel):
    return torch.mean(torch.abs(predicted_mel - target_mel))


model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=16)

dataset = MelSpectrogramDataset("data")
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_epochs = 5
patience = 2
train_loss = []

for epoch in range(num_epochs):
    ii = 0
    jj = 0
    for mel_spectrogram in data_loader:
        jj+=1
        print(f'mel espectogram number: {jj} in epoch {epoch}')
        mel_spectrogram = pad_to_power_of_two(mel_spectrogram, 2048)
        noise_shape = mel_spectrogram.shape
        noise = torch.randn(noise_shape, dtype=torch.float)
        noisy_mel = scheduler.add_noise(mel_spectrogram, noise.float(), timesteps=scheduler.timesteps)

        optimizer.zero_grad()
        predicted_mel = model(noisy_mel)
        loss = mae_loss(predicted_mel, mel_spectrogram)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if epoch == 5:
            out_name = f'/generated' + f'_{ii + 1}.npy'
            print(f'Output file {out_name}')
            np.save(out_name, predicted_mel)

    print(f"Epoch {epoch}, Loss: {loss.item()}")

fig, axs = plt.subplots(1, 1)

axs[0].set_title('Training Loss')
axs[0].plot(train_loss)
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('MSE Loss')

plt.show()

torch.save(model.state_dict(), "mel_spectrogram_model.pth")