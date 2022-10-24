# Machine Percpeption Project 1

## Objectives
1. Build a Deep Convolutional Neural Network, train it on CIFAR-10 training set and test it on CIFAR-10 testing You can use any architectures learned in class or come up with your own architecture.
2. Attack the DCNN using three perturbations of your choice and show how the performance is affected. 
3. Provide a defense for one of the noise models and show the performance improvement.


&nbsp;

## Dataset:
The  CIFAR-10  dataset  consists  of  60000 32×32  color  images  of  10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. However, the training batches in total contain exactly 5000 images from each class.

&nbsp;

### Model used:
GoogleNet


### Attack methods
1. Fast Gradient Sign Method (FGSM)


### Defense Method
1. Defensive Distillation 