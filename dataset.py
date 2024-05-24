from torchvision import transforms
import numpy as np
import glob
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    """
    Custom Dataset for loading mel spectrograms and their associated genres.

    Args:
    - root_dir (str): Root directory containing the .npy files.
    """

    def __init__(self, root_dir):
        self.file_paths = self.find_npy_files(root_dir)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0], std=[1]),
            ]
        )

    def find_npy_files(self, root_dir):
        """
        Find all .npy files in the root directory recursively.

        Args:
        - root_dir (str): Root directory containing the .npy files.

        Returns:
        - list: List of file paths to .npy files.
        """
        return glob.glob(root_dir + "/**/*.npy", recursive=True)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: Tuple containing the mel spectrogram tensor and genre label.
        """
        mel_spectrogram, genre = self.load_mel_spectrogram(self.file_paths[idx])
        mel_spectrogram = self.transform(mel_spectrogram)
        genre = genre[1, 1]
        return mel_spectrogram[0, :, :].unsqueeze(0), int(genre)

    def load_mel_spectrogram(self, file_path):
        """
        Load a mel spectrogram from a file.

        Args:
        - file_path (str): Path to the .npy file.

        Returns:
        - tuple: Tuple containing the mel spectrogram and genre label.
        """
        spectrogram = np.load(file_path)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram[0, :, :], spectrogram[1, :, :]
