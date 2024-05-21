import os
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT


class AudioWindow:
    """
    Generator class to obtain the time-windows of the audio.

    Inputs:
    - audio: audio object read from TorchAudio. Array.
    - size: the size (in elements) of the window.
    - overlap: the amount (in elements) of common items between consecutive
      windows.
    """

    def __init__(self, audio, size, overlap=0):
        self.audio = audio
        self.size = size
        self.overlap = overlap
        self.pos = 0

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        # print(f'\tPosition: {self.pos}')
        while self.pos < len(self.audio):
            out = self.audio[self.pos : self.pos + self.size]
            self.pos += self.size - self.overlap
            return out
        raise StopIteration()


def iterate_through_tree(path, mel_pars):
    """
    A function that looks at all the files in a given path, and makes the
    piece-wise Mel-Spectrogram for each of them.
    """
    files = next(os.walk(path))
    for file in files[-1]:
        ext = os.path.splitext(file)[-1]
        if ext != ".mp3":
            continue
        spec = make_spectrogram(path + "/" + file, mel_pars)


def make_spectrogram(filename, arguments):
    """
    For each .mp3 file in the directory, reads the file and makes the Mel
    Spectrogram. Adds a second, constant-valued channel, that represents the
    genre of the music, determined by the path that it was read from.

    Input:
    - filename: filename of the audio to generate spectrograms.
    - arguments: specific arguments for MelSpectrogram.
    """
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1, 1)
    mel_spec_transform = TT.MelSpectrogram(**arguments)
    genre_embd = {
        "epic": 0,
        "festival": 1,
        "fight": 2,
        "mysterious": 3,
        "romance": 4,
        "sad": 5,
        "tavern": 6,
        "town": 7,
    }
    genre = genre_embd[filename.split("/")[1]]

    step = 256 * (2047 + 0)
    overlap = 0
    windows = AudioWindow(audio, step, overlap)

    ### For each window, calcualte spectrogram and add layer corresponding
    ### to the genre
    for ii, window in enumerate(windows):
        if len(window) < step:
            break
        spec = mel_spec_transform(window)
        spec = torch.unsqueeze(spec, 0)
        genre = torch.ones_like(spec) * genre
        out = torch.cat((spec, genre), 0)
        if out.ndim != 3:
            continue
        out_name = os.path.splitext(filename)[0] + f"_{ii+1}.npy"

        np.save(out_name, out)


mel_pars = {
    "sample_rate": 22050,
    "win_length": 256 * 4,
    "hop_length": 256,
    "n_fft": 1024,
    "f_min": 20.0,
    "f_max": 22050 / 2.0,
    "n_mels": 80,
    "power": 1.0,
    "normalized": True,
}

if __name__ == "__main__":
    genres = [g.path for g in os.scandir() if g.is_dir()]
    genres.sort()
    print(genres)
    for genre in genres:
        iterate_through_tree(genre, mel_pars)
