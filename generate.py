"""
Script to generate the spectrograms and sound with the trained model
"""
import os
import sys
import subprocess
import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from model import Diffusion


def spectrogram_to_audio(
        filename, sr=22050, n_fft=1024, hop_length=256, win_length=1024
):
    """
    Convert a Mel Spectrogram back to an audio waveform using the Griffin-Lim algorithm.

    Args:
        filename (string): path to Mel Spectrogram array.
        sr (int): Sample rate of the original audio.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Window length used for each frame of audio.

    Return:
          audio_name(string): location of the audio file.
    """
    # Load the Mel Spectrogram from file
    audio = np.load(filename)
    # Convert Mel Spectrogram to STFT
    S = librosa.feature.inverse.mel_to_stft(audio, sr=sr, n_fft=n_fft, power=2.0)
    # Apply Griffin-Lim algorithm to get the audio waveform
    audio = librosa.griffinlim(
        S, n_iter=32, hop_length=hop_length, win_length=win_length
    )
    # Save the generated audio to a file
    audio_name = "test_results/generated_wav/generated_audio.wav"
    sf.write(audio_name, np.ravel(audio), sr)
    return audio_name


def spec_image(audio_path):
    """
    Generate and save a spectrogram image from an audio file.

    Args:
        audio_path (str): Path to the audio file.
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=22050)
    n_fft = 1024
    hop_length = 256
    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(D)
    # Convert amplitude to dB
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Define the output directory for the image
    output_dir = "test_results/generated_image"
    image_path = os.path.join(output_dir, "generated_spectrogram.png")
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram of Generated Audio")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # Save the spectrogram image
    plt.savefig(image_path)
    plt.close()


def generate_sample(genre, params_file, num_timesteps=10):
    """
    Test the model by generating images from noise, conditioned on a specific genre.

    Args:
        genre (int): Genre label to condition the generation.
        params_file (str): path to parameter file.
        num_timesteps (int): Number of timesteps for the diffusion model.

    """
    # Initialize the diffusion model
    diffusion = Diffusion(num_timesteps)
    diffusion.load_state_dict(torch.load(params_file))
    diffusion.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = diffusion.to(device)
    genre_tensor = torch.tensor([genre], dtype=torch.long).to(device)
    # Generate initial noise
    noise = torch.randn(1, 1, 80, 2048).to(device)
    # Generate the spectrogram conditioned on the genre
    generated_spectrogram = diffusion.sample(noise, genre_tensor)
    generated_spectrogram = generated_spectrogram.squeeze().cpu().detach().numpy()

    # Save the generated spectrogram to a file
    filename = "test_results/generated_npy/generated_spectrogram.npy"
    np.save(filename, generated_spectrogram)
    print("Converting...")
    # Convert the spectrogram to audio and return the path to the audio file
    return spectrogram_to_audio(filename)


def main():
    """
    Main function to orchestrate the generation of audio from a conditioned model,
    and subsequently create a spectrogram image.
    """
    # Create necessary directories
    os.makedirs("test_results/generated_image/", exist_ok=True)
    os.makedirs("test_results/generated_wav/", exist_ok=True)
    os.makedirs("test_results/generated_npy/", exist_ok=True)
    subprocess.run(["python", "gui.py"])

    params_file = sys.argv[1]
    print(f"Trained model: {params_file}")
    with open("genre_embd.txt", "r") as genre_file:
        genre = genre_file.read()
    # Generate the audio sample
    print("Generating music...")
    audio_path = generate_sample(int(genre), params_file)
    # Generate the spectrogram image from the audio
    spec_image(audio_path)

    print("Done!")
    print("You can find the generated sample in: test_results/generated_wav")
    print("You can find the generated spectrogram in: test_results/generated_image")


if __name__ == "__main__":
    main()
