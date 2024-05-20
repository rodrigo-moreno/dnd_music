import os
from model import (
    UNet,
)
from diffusion import DiffusionModel
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf


def test_model_with_genre(genre, num_timesteps=1000):
    """
    Test the model by generating images from noise, conditioned on a given genre.

    Args:
        genre (int): Genre label to condition the generation.
        num_timesteps (int): Number of timesteps for the diffusion model.
    """
    diffusion = DiffusionModel(num_timesteps=num_timesteps)
    model = UNet()
    model.load_state_dict(torch.load(f'diffusion_model_1.pth'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genre = torch.tensor([genre], dtype=torch.long).to(device)

    with torch.no_grad():
        # Generar un nuevo espectrograma
        generated_spectrogram = diffusion.sample(model, (1, 16, 80, 2048), genre, num_timesteps)

    # Convertir espectrograma generado a audio
    generated_spectrogram = generated_spectrogram.squeeze().cpu().detach().numpy()
    return spectrogram_to_audio(generated_spectrogram)




def spectrogram_to_audio(
    spec, sr=22050, n_fft=1024, hop_length=256, win_length=256 * 4
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

    S = librosa.feature.inverse.mel_to_stft(spec, sr=sr, n_fft=n_fft, power=2.0)
    audio = librosa.griffinlim(
        S, n_iter=32, hop_length=hop_length, win_length=win_length
    )
    audio_name = f"test_results/generated_wav/generated_audio.wav"
    sf.write(audio_name, np.ravel(audio), sr)
    return audio_name


def spec_image(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)

    # Convertir el audio a un espectrograma usando STFT
    n_fft = 1024
    hop_length = 256
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(D)
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    output_dir = 'test_results/generated_image'
    # Ruta para guardar la imagen
    image_path = os.path.join(output_dir, 'generated_spectrogram.png')

    # Visualizar y guardar el espectrograma
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Generated Audio')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(image_path)
    plt.close()


def main(genre):
    os.makedirs("test_results/generated_npy/", exist_ok=True)
    os.makedirs("test_results/generated_image/", exist_ok=True)
    os.makedirs("test_results/generated_wav/", exist_ok=True)
    audio_path = test_model_with_genre(genre,10)
    spec_image(audio_path)
    print("done!")


if __name__ == "__main__":
    main()