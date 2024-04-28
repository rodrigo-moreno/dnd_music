import os
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT


class AudioWindow():
    """
    Generator class to obtain the time-windows of the audio.

    Inputs:
    - audio: audio object read from TorchAudio. Array.
    - size: the size (in elements) of the window.
    - overlap: the amount (in elements) of common items between consecutive
      windows.
    """
    def __init__(self, audio, size, overlap = 0):
        self.audio = audio
        self.size = size
        self.overlap = overlap
        self.pos = 0

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        #print(f'\tPosition: {self.pos}')
        while self.pos < len(self.audio):
            out = self.audio[self.pos : self.pos+self.size]
            self.pos += self.size - self.overlap
            return out
        raise StopIteration()


def iterate_through_tree(path, mel_pars):
    """
    A function that looks at all the files in a given path, and makes the
    piece-wise Mel-Spectrogram for each of them.
    """
    print(f'Standing on path {path}')
    files = next(os.walk(path))
    #print(files[-1])
    for file in files[-1]:
        ext = os.path.splitext(file)[-1]
        if ext != '.mp3':
            #print(f'f\tIgnoring file {file}')
            continue
        print(f'\tGonna transform {path + "/" + file}')
        make_spectrogram(path + '/' + file, mel_pars)


def make_spectrogram(filename, arguments):
    """
    Read file, break it into small pieces, make Mel-Spectrogram, and write the
    file with the necessary information in the filename.

    Input:
    - filename: filename of the audio to generate spectrograms.
    - arguments: specific arguments for MelSpectrogram.
    """
    ### Read file and setup spectrogram conditions
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1, 1)
    mel_spec_transform = TT.MelSpectrogram(**arguments)

    step = len(audio) // 10
    step_time = 10   ### We want this amount of seconds of audio
    overlap_time = 0   ### How much time overlap between windows
    step = sr * step_time   ### Convert to cycles
    overlap = sr * overlap_time   ### Likewise
    windows = AudioWindow(audio, step, overlap)
    for ii, window in enumerate(windows):
        if len(window) < step:
            break
        spec = mel_spec_transform(window)
        out_name = os.path.splitext(filename)[0] + f'_{ii+1}.npy'
        print(f'Output file {out_name}')
        np.save(out_name, spec)


mel_pars = {'sample_rate': 22050,
            'win_length': 256*4,
            'hop_length': 256,
            'n_fft': 1024,
            'f_min': 20.,
            'f_max': 22050 / 2.,
            'n_mels': 80,
            'power': 1.,
            'normalized': True,
            }

if __name__ == '__main__':
    genres = [g.path for g in os.scandir() if g.is_dir()]
    print(genres)
    for genre in genres:
        iterate_through_tree(genre, mel_pars)

