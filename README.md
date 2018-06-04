# Capsnet
Traffic sign detection use capsules network and HSV colorspace.

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-80%25-blue.svg?style=plastic)

Dataset: https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

## Requirements
- Python 3
- NumPy 1.13
- Tensorflow 1.4
- Keras
- OpenCV 3
- docopt 0.6.2
- Sklearn: 0.18.1
- Matplotlib

## Train

    $> python train_capsnet.py -h
    $> python train_capsnet.py dataset/

## Run

    $> python main.py
