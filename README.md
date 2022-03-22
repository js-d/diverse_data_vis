# diverse_data_vis

After cloning, move inside this repository and:
* create a conda environment using `conda env create -f environment.yml`
* add a `checkpoints` folder containing the checkpoints of the different models
* add an `out` folder which will contain the visualizations.

Inside `main.py`, specify a list of model names, a list of images, and list of layer names. 

`python main.py` (with access to CUDA) computes the corresponding [caricatures](https://github.com/tensorflow/lucid/issues/121) and stores them in `out` in `.npy` and `.png` format. 

`activations.py` and `objectives.py` contain functions used by `main.py`.