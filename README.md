# EN.520.665 Machine Percpeption
Project 1
Team Member: Danny (Iou-Sheng) Chang, Juo-Tung Chen 

## Objectives
1. Build a Deep Convolutional Neural Network, train it on CIFAR-10 training set and test it on CIFAR-10 testing set.
2. Attack the DCNN using three perturbations of our choice and show how the performance is affected. 
3. Provide a defense for one of the noise models and show the performance improvement.


&nbsp;

## Dataset:
The CIFAR-10 Dataset consists of 60000 32×32 color images of 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches in total contain exactly 5000 images from each class.

Reference: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009, https://www.cs.toronto.edu/~kriz/cifar.html.

&nbsp;

## Choice of Deep Convolutional Neural Network
GoogLeNet

## Adversarial Attacks Methods
1. Fast Gradient Sign Method (FGSM)
2. Noise Attack
3. Semantic Attack
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/Attacked_image.png?raw=true)


## Defense Method
The team selected Defensive Distillation as the defense method against FGSM attack.

Defensive distillation is a defense method to adversarial perturbations against DCNN. The method was introduced by Papernot et al. in Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks.

Reference: Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami. Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks. arXiv preprint arXiv:1511.04508v2, 2016.

### Defense Results
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/denfense_comparison(FGSM).png?raw=true)
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/class_accuracy_no_defense.png?raw=true)
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/class_accuracy_defense.png?raw=true)