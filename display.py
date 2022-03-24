import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


st.write("# Visualizations")

# list of models, images, and layers
list_pretrain_model_names = [
    m_name[:-3]
    for m_name in os.listdir("pretrain_checkpoints")
    if m_name.startswith("resnet_50_single")
    or m_name.startswith("resnet_50_train")
    or m_name == "resnet_50_imagenet_200k.pt"
]
list_pretrain_model_names.remove("resnet_50_train_00_redo")

list_ft_model_names = [m_name[:-3] for m_name in os.listdir("ft_checkpoints")]

list_img_names = [
    "flowers",
    "chain",
    "new_jeep",
    "new_piano",
    "new_chrysler",
    "new_furniture",
]

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

ft1 = st.sidebar.selectbox("Model 1: finetuned?", ["Fine-tuned", "Pretrained"])
if ft1 == "Fine-tuned":
    model1_name = st.sidebar.selectbox("Model 1: model?", list_ft_model_names)
else:
    model1_name = st.sidebar.selectbox("Model 1: model?", list_pretrain_model_names)

ft2 = st.sidebar.selectbox("Model 2: finetuned?", ["Fine-tuned", "Pretrained"])
if ft2 == "Fine-tuned":
    model2_name = st.sidebar.selectbox("Model 2: model?", list_ft_model_names)
else:
    model2_name = st.sidebar.selectbox("Model 2: model?", list_pretrain_model_names)

layer_name = st.sidebar.selectbox("Layer", list_layer_names)

img_name = st.sidebar.selectbox("Image", list_img_names)
st.sidebar.image(Image.open(Path(f"images/{img_name}.png")))


fn1_core = f"{model1_name}_{layer_name}_{img_name}"
fn2_core = f"{model2_name}_{layer_name}_{img_name}"

if ft1 == "Fine-tuned":
    path1 = Path(f"ft_results/{fn1_core}.npy")
else:
    path1 = Path(f"pretrain_results/{fn1_core}.npy")
if ft2 == "Fine-tuned":
    path2 = Path(f"ft_results/{fn2_core}.npy")
else:
    path2 = Path(f"pretrain_results/{fn2_core}.npy")

img1 = Image.fromarray(np.uint8(np.load(path1) * 255)).convert("RGB")
img2 = Image.fromarray(np.uint8(np.load(path2) * 255)).convert("RGB")

st.image(
    [img1, img2],
    width=320,
    caption=[f"Model 1: {model1_name}", f"Model 2: {model2_name}"],
)

frame_text = st.empty()
image = st.empty()
