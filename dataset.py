from torchvision import transforms
import numpy as np
import glob
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = self.find_npy_files(root_dir)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0], std=[1]),
            ]
        )

    def find_npy_files(self, root_dir):
        return glob.glob(root_dir + "/**/*.npy", recursive=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel_spectrogram, genre = self.load_mel_spectrogram(self.file_paths[idx])
        mel_spectrogram = self.transform(mel_spectrogram)
        genre = genre[1, 1]
        return mel_spectrogram[0, :, :].unsqueeze(0), int(genre)

    def load_mel_spectrogram(self, file_path):
        spectrogram = np.load(file_path)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram[0, :, :], spectrogram[1, :, :]
