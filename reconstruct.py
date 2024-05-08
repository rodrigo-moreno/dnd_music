import numpy as np
import librosa
import soundfile as sf
import os


def spectrogram_to_audio(
    filename, sr=22050, n_fft=1024, hop_length=256, win_length=256 * 4
):
    """
    Convert a Mel Spectrogram back to an audio waveform using the Griffin-Lim algorithm.

    Input:
    - filename: Path to the .npy file containing the Mel Spectrogram.
    - sr: Sample rate of the original audio.
    - n_fft: FFT window size.
    - hop_length: Number of samples between successive frames.
    - win_length: Each frame of audio is windowed by window() of length win_length.
    - n_mels: Number of Mel bands to generate.
    """

    S = np.load(filename)
    S = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=n_fft, power=2.0)
    audio = librosa.griffinlim(
        S, n_iter=32, hop_length=hop_length, win_length=win_length
    )

    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace("mel_spectogram_", "")
    audio_name = f"./test_results/generated_wav/{name}.wav"
    sf.write(audio_name, np.ravel(audio), sr)
    print(f"Reconstructed audio saved as {audio_name}")


os.makedirs("./test_results/generated_wav/",exist_ok=True)
path = "./test_results/generated_npy"
files = next(os.walk(path))[-1]
for spec in files:
    print(f"\tGonna transform {spec}")
    spec = path + "/" + spec
    spectrogram_to_audio(spec)
