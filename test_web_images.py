#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Test the model

Usage:
  test.py

Options:
  -h --help     Show this help.
  <dataset>     Dataset folder
  <ckpt>        Path to the checkpoints to restore
"""

from docopt import docopt
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import pickle
import os

from model import ModelTrafficSign
from data_handler import get_data

def test_web_images(ckpt):
    """
        Test images located into the "from_web" folder.
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """

    # Load name of id
    with open("signnames.csv", "r") as f:
        signnames = f.read()
    id_to_name = { int(line.split(",")[0]):line.split(",")[1] for line in signnames.split("\n")[1:] if len(line) > 0}

    images = []

    # Read all image into the folder
    for filename in os.listdir("from_web"):
        img = Image.open(os.path.join("from_web", filename))
        img = img.resize((32, 32))
        img = np.array(img) / 255
        images.append(img)

    # Load the model
    model = ModelTrafficSign("TrafficSign", output_folder=None)
    model.load(ckpt)

    # Get the prediction
    predictions = model.predict(images)

    # Plot the result
    fig, axs = plt.subplots(10, 2, figsize=(10, 25))
    axs = axs.ravel()
    for i in range(20):
        if i%2 == 0:
            axs[i].axis('off')
            axs[i].imshow(images[i // 2])
            axs[i].set_title("Prediction: %s" % id_to_name[np.argmax(predictions[i // 2])])
        else:
            axs[i].bar(np.arange(43), predictions[i // 2])
            axs[i].set_ylabel("Softmax")
            axs[i].set_xlabel("Labels")

    plt.show()


if __name__ == '__main__':
    test_web_images("/media/thanhtung/DATA/traffic-sign-data/temp/capsnet/outputs/checkpoints/c1s_9_c1n_256_c2s_6_c2n_64_c2d_0.7_c1vl_16_c1s_5_c1nf_16_c2vl_32_lr_0.0001_rs_1--TrafficSign--1516048778.566972")
