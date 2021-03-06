#!~/.conda/envs/train_imagenet/bin/python
from time import time

t0 = time()

import argparse
import os
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

from lucent.optvis import render
from lucent.modelzoo.util import get_model_layers

from objectives import caricature_obj
from activations import single_layer_acts


def get_model(model_name, device, ft=False):
    if model_name == "pretrained":
        model = models.resnet50(pretrained=True)
    else:
        if ft == True:
            chk_name = f"ft_checkpoints/{model_name}.pt"
            num_classes = 37 # pets dataset
        else:
            chk_name = f"pretrain_checkpoints/{model_name}.pt"
            num_classes = 40 # auto3d dataset
        state_dict = torch.load(chk_name)["state_dict"]
        model = models.resnet50()
        if model_name != "resnet_50_imagenet_200k":
            model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


# to get all the layers, including residual layers (not used here)
def get_resnet_layers(num_stages=4, num_blocks=[3, 4, 6, 3], num_conv=3):
    res_list = []
    for stage in range(1, num_stages + 1):
        for block in range(num_blocks[stage - 1]):
            for conv in range(1, num_conv + 1):
                res_list.append(f"layer{stage}_{block}_conv{conv}")
            res_list.append(f"layer{stage}_{block}")
        res_list.append(f"layer{stage}")
    return res_list


def get_img_tens(img_name, device, norm=None):
    img_path = f"images/{img_name}.png"
    img = Image.open(img_path).convert("RGB")
    img_tens = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # normalize image
    if norm == None:
        # for all models but pretrained
        transform_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
    elif norm == "pretrained":
        # for pretrained model
        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    norm_img_tens = transform_normalize(img_tens).to(device)
    return norm_img_tens


def get_vis(model, norm_img_tens, layer_name, num_steps):
    # get objective
    with torch.no_grad():
        direction = single_layer_acts(model, norm_img_tens, layer_name)[0, :, :, :]
    obj = caricature_obj(layer_name, direction)

    # get explanation
    fvis = render.render_vis(
        model,
        obj,
        thresholds=(num_steps,),
        show_image=False,
        verbose=True,
        progress=False,
    )
    fvis_arr = fvis[0][0, :, :, :]

    return fvis_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # models
    # pretraining
    # list_model_names = [
    #     m_name[:-3]
    #     for m_name in os.listdir("pretrain_checkpoints")
    #     if m_name.startswith("resnet_50_single")
    #     or m_name.startswith("resnet_50_train")
    #     or m_name == "resnet_50_imagenet_200k"
    # ]
    # list_model_names.remove("resnet_50_train_00_redo")

    # fine-tuning
    list_model_names = [m_name[:-3] for m_name in os.listdir("ft_checkpoints")]

    # images
    assert args.img_name in [
        "flowers",
        "chain",
        "new_jeep",
        "new_piano",
        "new_chrysler",
        "new_furniture",
    ]
    list_img_names = [args.img_name]

    # layers
    list_layer_names = [
        "layer2_1_conv2",
        "layer3_1",
        "layer3_2_conv2",
        "layer3_3",
        "layer3_4_conv3",
        "layer3_5",
        "layer4_1_conv2",
        "layer4_2",
    ]

    # get the visualizations
    list_times = []
    for model_name in list_model_names:
        # pretraining
        # model = get_model(model_name, device)
        # finetuning
        model = get_model(model_name, device, ft=True)

        for img_name in list_img_names:
            if model_name == "pretrained":
                norm_img_tens = get_img_tens(img_name, device, norm="pretrained").to(
                    device
                )
            else:
                norm_img_tens = get_img_tens(img_name, device).to(device)

            for layer_name in list_layer_names:
                if layer_name.startswith("layer4"):
                    num_steps = 1024
                else:
                    num_steps = 512
                print(model_name)
                print(img_name)
                print(layer_name)

                t1 = time()

                # get visualization
                fvis_arr = get_vis(model, norm_img_tens, layer_name, num_steps)

                # count visualization time
                vis_time = time() - t1
                print("visualization time", vis_time)
                list_times.append(vis_time)

                # save npy
                np.save(f"out/{model_name}_{layer_name}_{img_name}.npy", fvis_arr)

                # save png
                im = Image.fromarray((fvis_arr * 255).astype("uint8"), "RGB")
                im.save(f"out/{model_name}_{layer_name}_{img_name}.png")

                print()

    print()
    print(list_times)
    print("total time", time() - t0)
    print("number of visualizations", len(list_times))
    print("sum visualization times", sum(list_times))
    print("mean visualization times", sum(list_times) / len(list_times))

    # to get the names of the layers of the model
    # list_layers = get_model_layers(model)[:20]

    # if neuron vis
    # param_f = lambda: param.image(224, fft=True, decorrelate=True) if neuron
    # vis = render.render_vis(model, layer_name, param_f=param_f, show_image=False)

