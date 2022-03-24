# diverse_data_vis

### To display the visualizations

After cloning, move inside this repository and:

- install `streamlit`
- download and unzip `pretrain_results.zip` ([link](https://drive.google.com/file/d/1az7xllu2cJz6tM7QZfO-vBqQSnWw8eaM/view?usp=sharing)) and `ft_results.zip` ([link](https://drive.google.com/file/d/1p2crA0_j5A359TmIb9_m5STm53wjhppr/view?usp=sharing)), and move the folders inside this repository
- run the streamlit app: `streamlit run display.py`

### To obtain visualizations

After cloning, move inside this repository and:

* create a conda environment using `conda env create -f environment.yml` (requires GPU support)
* add a `checkpoints` folder containing the checkpoints of the different models you would like to visualize
* add an `out` folder which will contain the visualizations.

Inside `main.py`, specify a list of model names, a list of images, and list of layer names.

`python main.py` (with access to CUDA) computes the corresponding [caricatures](https://github.com/tensorflow/lucid/issues/121) and stores them in `out`. Intuitively, a caricature describes what a layer `l` "sees" in an input `x` by visualizing the activation vector `f_l(x)`.

In `out`, the visualizations are stored in `.npy` and `.png` formats. They are named in the form `<model_name>_<layer_name>_<image_name>.npy` and `<model_name>_<layer_name>_<image_name>.png`.

`activations.py` and `objectives.py` contain functions used by `main.py`.

## preliminary results

To start out, I computed visualizations for:

* 4 models: `resnet_50_imagenet_200k`, `resnet_50_single_camera`, `resnet_50_single_texture`, and the torchvision `pretrained` resnet50
* 3 images: [flowers](https://distill.pub/2018/building-blocks/examples/input_images/flowers.jpeg), [dog_cat](https://distill.pub/2018/building-blocks/examples/input_images/dog_cat.jpeg), [chain](https://distill.pub/2018/building-blocks/examples/input_images/chain.jpeg)
* 7 layers: `layer2_2_conv1`, `layer3_0_conv3`, `layer3_2_conv2`, `layer3_3`, `layer3_4_conv3`, `layer4_0_conv3`, `layer4_2_conv3`

You can find the visualizations by unzipping `preliminary_results.zip` ([link](https://drive.google.com/file/d/1TpodrJn6ts_xvUhRfV-JgGeG1AtCjebQ/view?usp=sharing)).

A few observations:

* Unsurprisingly, the caricatures look much better for `resnet_50_imagenet_200k` and `pretrained`, both of which were trained on 1000 classes rather than 40.
* Comparing the caricatures of `resnet_50_single_camera` and `resnet_50_single_texture` is difficult because they don’t look great, but I would say they look different.
* The caricatures by residual blocks (here `layer3_3`) look better than the others, and they also look quite different between both models. In particular:
  * The colours are consistenly different, and are more uniform for `resnet_50_single_texture`.
  * The visualisations of `resnet_50_single_texture` have small circular patterns that the visualizations of `resnet_50_single_camera` lack.

## pretrain results

Next, I computed visualizations for:

* 21 models: the 20 models trained on different variants of the dataset, as well as `resnet_50_imagenet_200k`
* 6 images: [flowers](https://distill.pub/2018/building-blocks/examples/input_images/flowers.jpeg), [dog_cat](https://distill.pub/2018/building-blocks/examples/input_images/dog_cat.jpeg), [chain](https://distill.pub/2018/building-blocks/examples/input_images/chain.jpeg), as well as images of a piano, a Jeep, a Chrysler car, a chest of drawers
* 8 layers: `layer2_1_conv2`,`layer3_1`,`layer3_2_conv2`,`layer3_3`,`layer3_4_conv3`,`layer3_5`,`layer4_1_conv2`,`layer4_2`, which I chose to visualize different parts of the network, and to have both convolutional layers and residual block ends

You can find the visualizations by unzipping `pretrain_results.zip` ([link](https://drive.google.com/file/d/1az7xllu2cJz6tM7QZfO-vBqQSnWw8eaM/view?usp=sharing)).

## fine-tune results

I computed visualizations for:

* 14 models: `single_environment`, `single_texture`, `train_00`, `imagenet_200k`, `single_camera`, `single_model`, `train_12`, `train_11`, `train_10`, `train_09`, `train_08`, `train_07`, `train_06`, `train_05`
* 6 images: [flowers](https://distill.pub/2018/building-blocks/examples/input_images/flowers.jpeg), [dog_cat](https://distill.pub/2018/building-blocks/examples/input_images/dog_cat.jpeg), [chain](https://distill.pub/2018/building-blocks/examples/input_images/chain.jpeg), as well as images of a piano, a Jeep, a Chrysler car, a chest of drawers
* 8 layers: `layer2_1_conv2`,`layer3_1`,`layer3_2_conv2`,`layer3_3`,`layer3_4_conv3`,`layer3_5`,`layer4_1_conv2`,`layer4_2`, which I chose to visualize different parts of the network, and to have both convolutional layers and residual block ends

You can find the visualizations by unzipping `ft_results.zip` ([link](https://drive.google.com/file/d/1p2crA0_j5A359TmIb9_m5STm53wjhppr/view?usp=sharing)).
