# diverse_data_vis

After cloning, move inside this repository and:
* create a conda environment using `conda env create -f environment.yml`
* add a `checkpoints` folder containing the checkpoints of the different models
* add an `out` folder which will contain the visualizations.

Inside `main.py`, specify a list of model names, a list of images, and list of layer names. 

`python main.py` (with access to CUDA) computes the corresponding [caricatures](https://github.com/tensorflow/lucid/issues/121) and stores them in `out`.

In `out`, the visualizations are stored in `.npy` and `.png` formats. They are named in the form `<model_name>_<layer_name>_<image_name>.npy` and `<model_name>_<layer_name>_<image_name>.png`.

`activations.py` and `objectives.py` contain functions used by `main.py`.

## preliminary results

To start out, I computed visualizations for:
* 4 models: `resnet_50_imagenet_200k`, `resnet_50_single_camera`, `resnet_50_single_texture`, and the torchvision `pretrained` resnet50
* 3 images: [flowers](https://distill.pub/2018/building-blocks/examples/input_images/flowers.jpeg), [dog_cat](https://distill.pub/2018/building-blocks/examples/input_images/dog_cat.jpeg), [chain](https://distill.pub/2018/building-blocks/examples/input_images/chain.jpeg)
* 7 layers: `layer2_2_conv1`, `layer3_0_conv3`, `layer3_2_conv2`, `layer3_3`, `layer3_4_conv3`, `layer4_0_conv3`, `layer4_2_conv3`


You can find the visualizations by unzipping `preliminary_results.zip`. A few observations:
* Unsurprisingly, the caricatures look much better for `resnet_50_imagenet_200k` and `pretrained`, both of which were trained on 1000 classes rather than 40.
* Comparing the caricatures of `resnet_50_single_camera` and `resnet_50_single_texture` is difficult because they don’t look great, but I would say they look different. 
* The caricatures by residual blocks (here `layer3_3`) look better than the others, and they also look quite different between both models. In particular: 
    * The colours are consistenly different, and are more uniform for `resnet_50_single_texture`.
    * The visualisations of `resnet_50_single_texture` have small circular patterns that the visualizations of `resnet_50_single_camera` lack.
