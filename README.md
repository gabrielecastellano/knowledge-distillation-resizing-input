# Knowledge Distillation without Upsampling

This repository contains code to train image classification models on the CIFAR-100 dataset, 
starting from models that have been pretrained on the bigger dataset ImageNet 2012. The goal of this project is to 
train the models in the new dataset input size (32x32) rather than the classic approach of upsampling image sizes to
those of the dataset that the network was trained from.

### Credits

Models 


## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101
- tensorboard 2.2.2(optional)


## Usage

### 1. enter directory
```bash
$ cd knowledge-distillation-resizing-input
```

### 2. run tensorbard (optional)
Install tensorboard
```bash
$ pip install tensorboard
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 3. train

```bash
# train resnet50 on gpu
$ python kd_train.py -net resnet50 -gpu
```


### 4. test

```bash
$ python test.py -net resnet50 -weights {path/to/saved/weights/for/Resnet50.pt}
```

## Training Details

The practical approach used to improve the accuracy is as follows: knowledge is distilled from the pretrained model 
by gradually using smaller sized images, starting from 224 (the size
original network was trained) to 32 (the size of pictures in the target CIFAR-100 dataset). 
The size of the input images is reduced by 32 pixels after a long period of not-improving accuracy occurs. The learning rate is
reduced as well with an higher frequency, and restored every time the input size is reduced.

## Results
---
