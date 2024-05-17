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
    #print(f'Standing on path {path}')
    files = next(os.walk(path))
    #print(files[-1])
    for file in files[-1]:
        ext = os.path.splitext(file)[-1]
        if ext != '.mp3':
            #print(f'f\tIgnoring file {file}')
            continue
        #print(f'\tGonna transform {path + "/" + file}')
        spec = make_spectrogram(path + '/' + file, mel_pars)


def make_spectrogram(filename, arguments):
    """
    For each .mp3 file in the directory, reads the file and makes the Mel
    Spectrogram. Adds a second, constant-valued channel, that represents the
    genre of the music, determined by the path that it was read from.

    Input:
    - filename: filename of the audio to generate spectrograms.
    - arguments: specific arguments for MelSpectrogram.
    """
    ### Read file and setup spectrogram conditions
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1, 1)
    mel_spec_transform = TT.MelSpectrogram(**arguments)
    genre_embd = {'epic': 1,
                  'festival': 2,
                  'fight': 3,
                  'mysterious': 4,
                  'romance': 5,
                  'sad': 6,
                  'tavern': 7,
                  'town': 8,
                  }
    genre = genre_embd[filename.split('/')[1]]

    ### We're ignoring this now. It was interesting for other resons
    #step_time = 10                ### We want this amount of seconds of audio
    #overlap_time = 0              ### How much time overlap between windows
    #step = sr * step_time         ### Convert to cycles
    #overlap = sr * overlap_time   ### Likewise

    step = 256*(2047 + 0)
    overlap = 0
    windows = AudioWindow(audio, step, overlap)

    ### For each window, calcualte spectrogram and add layer corresponding
    ### to the genre
    for ii, window in enumerate(windows):
        if len(window) < step:
            break
        #print(f'Dims of window: {window.shape}')
        spec = mel_spec_transform(window)
        spec = torch.unsqueeze(spec, 0)
        #print(f'Dims of spec: {spec.shape}')
        genre = torch.ones_like(spec) * genre
        #print(f'Dims of genre: {genre.shape}')
        out = torch.cat((spec, genre), 0)
        #print(f'Dims of out: {out.shape}')
        out_name = os.path.splitext(filename)[0] + f'_{ii+1}.npy'
        #print(f'Output file {out_name}')
        np.save(out_name, out)


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
    genres.sort()
    print(genres)
    for genre in genres:
        iterate_through_tree(genre, mel_pars)

