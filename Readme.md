# DnD Music

The goal of the project is to generate mood-specific ambient music for role-playing purposes. We use a diffusion model to generate Mel spectrograms from which we reconstruct an audio signal.

## How to try?
Just run generate.py on your local machine including the path to your model.

e.g: python3 generate.py <PATH_TO_MODEL>

PD: all required libraries are provided in requirements.txt

## How to train?
Just run training.py on your local machine.

e.g: python3 training.py

NB: remember to loging in WandB!
# Components
## Data Preparation

The split_train.py script can be used to process the files inside of the data folder to create the spectrograms used for the model's training.
The dataset.py file contains the MelSpectrogramDataset class that is used to handle the loading of the spectrograms in the dataset.

## Model

The model.py file contains the model's definition along with all its components and the Diffusion processes.

## Model Training

The model can be trained by running the training.py script with a complete dataset of spectrograms. The logs will be saved using Weights and Biases.

## Testing

The performance can be assessed after training by running the generate.py script, the test will generate some audio samples that will be saved in the test_results/generated_wav directory along with spectrograms in the test_results/generated_image directory.
The test will also run the user interface.

## GUI

The gui.py script provides a simple graphic user interface to use the trained model to generate samples conditionally on the selected genre.

# OTHER COMMENTS
You may find our first approach on the model architecture in the previous_try/ folder.

We also include the report (report.pdf) with more in-depth information of our work.


