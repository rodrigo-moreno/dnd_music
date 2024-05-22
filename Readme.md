# DnD Music

The goal of the project is to generate mood-specific ambient music for role-playing purposes. We use a diffusion model to generate Mel spectrograms from which we reconstruct an audio signal.

## How to try?
Just run generate.py on your local machine.

e.g: python3 generate.py

## How to train?
Just run train.py on your local machine.

e.g: python3 train.py

NB: remember to include a .txt file on your repository containing your wandb key!
# Components
## Data Preparation

The split_train.py script can be used to process the files inside of the data folder to create the spectrograms used for the model's training.
The dataset.py file contains the MelSpectrogramDataset class that is used to handle the loading of the spectrograms in the dataset.

## Model

The model.py file contains the model's definition along with all its components.

## Model Training

The model can be trained by running the training.py script with a complete dataset of spectrograms. The logs will be saved using Weights and Biases.
The diffusion.py script contains sampling tools for the forward and reverse diffusion process. 

## Testing

The performance can be assessed after training by running the generate.py script, the test will generate some audio samples that will be saved in the test_results/generated_wav directory along with spectrograms in the test_results/generated_image directory.
The test will also run the user interface.
Due to some NaN values possibly appearing in the generated samples, we use the interpolate.py script to tackle this issue and fix the values.

## GUI

The gui.py script provides a simple graphic user interface to use the trained model to generate samples conditionally on the selected genre.




