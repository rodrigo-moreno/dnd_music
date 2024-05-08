import os

from model import (
    UNet,
)
from dataset import MelSpectrogramDataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

model = UNet()
model.load_state_dict(torch.load("train_results/mel_spectrogram_model.pth"))
model.eval()
mood = "romance"
dataset = MelSpectrogramDataset(f"data/{mood}")
test_loader = DataLoader(dataset, batch_size=16, shuffle=True)
i = 0
j = 0
for test_data in test_loader:
    i += 1
    print(f"testing batch {i}")
    test_data = test_data.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        predictions = model(test_data)


os.makedirs("./test_results/generated_npy/", exist_ok=True)
os.makedirs("./test_results/generated_image/", exist_ok=True)

for mel in predictions:
    predictions_np = predictions.squeeze().cpu().numpy()

    filename = f"./test_results/generated_npy/mel_spectogram_{mood}_{j}.npy"
    np.save(filename, predictions_np)
    j += 1
    mel = mel.squeeze()
    mel = mel - mel.min()
    mel = mel / mel.max() * 255
    mel = mel.byte()
    image = to_pil_image(mel)
    image.save(f"./test_results/generated_image/mel_spectogram_{mood}_{j}.png")
