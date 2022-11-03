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
GoogLeNet


### Attack methods
1. Fast Gradient Sign Method (FGSM)
2. noise attack
3. semantic attack
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/Attacked_image.png?raw=true)


### Defense Method
Defensive Distillation was used to deal with the FGSM attack.


- It is an adversarial training technique that adds flexibility to an algorithm’s classification process so the model is less susceptible to exploitation. In distillation training, one model is trained to predict the output probabilities of another model that was trained on an earlier, baseline standard to emphasize accuracy.

- The first model is trained with “hard” labels to achieve maximum accuracy, the first model then provides “soft” labels to train the second model. This uncertainty is used to train the second model to act as an additional filter. Since now there’s an element of randomness to gaining a perfect match, the second or “distilled” algorithm is far more robust and can spot spoofing attempts easier. It’s now far more difficult to attack the model with adversarial examples.

### Defense Results
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/denfense_comparison(FGSM).png?raw=true)
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/class_accuracy_no_defense.png?raw=true)
![alt text](https://github.com/JuoTungChen/MP_project1/blob/main/class_accuracy_defense.png?raw=true)




