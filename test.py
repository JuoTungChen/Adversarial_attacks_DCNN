"""
instructions:
1. modify main to call desired functions:
    (i) if you want to show regular testing results from the pretrained model 
        -> call solver.run() in main and uncomment the line with testing batch size = 100
    (ii) if you want to show testing results under attack 
        -> call solver.run_attack() in main and uncomment the line with testing batch size = 1

2. using different neural network models
    (i) training with different prebuild NN model
        in the load_model function within the Solver class
        -> comment out self.model = torch.load('model_1.pth')
        -> uncomment corresponding models such as: self.model = AlexNet().to(self.device)
    (ii) save trained model
        in the run function in Solver class
        -> uncomment the last line: self.save()
    (iii) loading pretrained model
        in the load_model function within the Solver class
        -> modify the path in self.model = torch.load('model_1.pth')

3. changing parameters for the attack methods
    -> for FGSM and noise attack, just change the epsilons in the run_attack function

written by: Juo-Tung Chen
"""

import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from torchvision import transforms as transforms
import numpy as np

import argparse

from models import *
from misc import progress_bar

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    # parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--trainBatchSize', default=50, type=int, help='training batch size')
## original testing -> test size = 100 (uncomment this line to show regular testing results)
    # parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')

## run attack -> test size = 1 (uncomment this line to show attack results)
    parser.add_argument('--testBatchSize', default=1, type=int, help='testing batch size')

    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)

    # solver.run()
    solver.run_attack()
    # solver.defense()
    # solver.test_distillation()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.Temp = 100

    def load_data(self):
        """
            load the image data from CIFAR-10
        """
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        """
            load perbuild NN models or load pretrained model 
        """
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = LeNet().to(self.device)
        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        # self.model = WideResNet(depth=28, num_classes=10).to(self.device)

        self.model = torch.load('model_1.pth')      #load previous trained model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)

        self.model_defense = torch.load('modelF1.pth') 
        self.optimizer_d = optim.Adam(self.model_defense.parameters(), lr=self.lr)
        self.scheduler_d = optim.lr_scheduler.MultiStepLR(self.optimizer_d, milestones=[75, 150], gamma=0.5)


        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.criterion = nn.NLLLoss()().to(self.device)



    def train(self):
        """
            train the NN model 
        """
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total



    def test(self):
        """
            regular testing function using the testing set from the CIFAR-10
            (can display the accuracy for each class of objects)
        """
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():           
            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in CLASSES}
            total_pred = {classname: 0 for classname in CLASSES} 

            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
            # test different class
                for label, predict in zip(target, prediction[1]):
                    if label == predict:
                        correct_pred[CLASSES[label]] += 1
                    total_pred[CLASSES[label]] += 1    
        # accuracy for different class 
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        return test_loss, test_correct / total


# ===================================================== FGSM Attack ===============================================================================

    ## fast gradient sign method (fgsm) attack
    def fgsm_attack(self, x, eps, data_grad):
        """
            fgsm attack -> Create the perturbed image by adjusting each pixel of the input image
                        based on the element-wise sign of the data gradient

            x: torch.Tensor (The input image)
            eps: noise magnitude
            data_grad: the gradient of the input data
        """

        pert_out = x + eps * data_grad.sign()
        pert_out = torch.clamp(pert_out, 0, 1)           # Adding clipping to maintain [0,1] range

        return pert_out




# ===================================================== noise attack ===============================================================================

    def noise_attack(self, x, eps):
        """
            noise attack -> create a tensor of the same shape as the input image x
                            and then make it a uniform distribution between -eps and  +eps

            x: torch.Tensor (The input image)
            eps: noise magnitude
        """
        eta = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(self.device)
        adv_x = x + eta

        return adv_x

# ===================================================== semantic attack ===============================================================================

    def semantic_attack(self, x):
        """
            semantic attack -> returns the negated image using the max value subtracting the image
            x: torch.Tensor (The input image)
        """
        return torch.max(x) - x
        


# ===================================================== defense ===============================================================================

    def defense(self):
        """
            implementing the defensive distillation to counter the FGSM attack

            defensive distillation: using the output of the originally trained model
                                    to train another Neural Network model
        
        """
        # creating a new model and its optimizer and scheduler
        modelF1 = GoogLeNet().to(self.device)
        optimizerF1 = optim.Adam(modelF1.parameters(), lr=self.lr)
        schedulerF1 = optim.lr_scheduler.MultiStepLR(optimizerF1, milestones=[75, 150], gamma=0.5)

        # train distilled Network
        accuracy = 0
        F1_epoch = 100
        for epoch in range(1, F1_epoch + 1):
            schedulerF1.step(epoch)
            print("\n===> epoch: %d/100" % epoch)
            train_result = self.train_F1(modelF1, optimizerF1, F1_epi)
            test_result = self.test_attack_F1(modelF1, 0.1, "fgsm")

        # save the trained defense model
        model_out_path = "modelF1_1.pth"
        torch.save(modelF1, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))



    def train_F1(self, model, optimizer, epsilon):
        """
            training the defense model
        """
        self.load_data()
        self.load_model()
        print("train:")
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)             
            output = self.model(data)
            output = output/self.Temp
            optimizer.zero_grad()
            loss = self.criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].detach().cpu().numpy() == target.detach().cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total




# ===================================================== test attacks ===============================================================================

    def run_attack(self):
        """
            evaluate the model with 3 different attacks
            for FGSM, a denfensive distillation model was used to deal with the attack
        """ 
        self.load_data()
        self.load_model()
        # can change the epsilons to desired values
        epsilons = [0,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64]


        # run through all of the attcks
        for attack in ("fgsm","noise","semantic"):
        # for attack in ("semantic", "noise"):
            accuracies = []
            accuracies_d = []
            examples = []

            if attack == "fgsm":
                 # test attack on model without defense
                print("\nresults for original GoogleNet:")
                for eps in epsilons:
                    acc, ex = self.test_attack(eps, attack)      
                    accuracies.append(acc)

                 # test attack on defense network
                print("\nresults for GoogleNet with defense:")
                for eps in epsilons:
                    acc, ex = self.test_attack_F1(eps, attack)     
                    accuracies_d.append(acc)
            
                # plot the results
                plt.figure(figsize=(5,5))
                plt.plot(epsilons, accuracies, "*-", label = 'GoogleNet w/o defense')
                plt.plot(epsilons, accuracies_d, "*-", label = 'GoogleNet w/ defensive-distillation')
            
                plt.title("FGSM attack")
                plt.legend(loc='lower right')
                plt.xlabel("Epsilon")
                plt.ylabel("Accuracy")
                plt.show() 

            elif attack == "noise":
                for eps in epsilons:
                    acc, ex = self.test_attack(eps, attack)       # test attack on regular NN
                    accuracies.append(acc)

            elif attack == "semantic":
                acc, ex = self.test_attack(0, attack)       # test attack on regular NN
                accuracies.append(acc)




    def test_attack(self, epsilon, attack):
        """
            test attack on the model
        """
        self.model.eval()
        correct = 0
        adv_examples = []
        correct_pred = {classname: 0 for classname in CLASSES}
        total_pred = {classname: 0 for classname in CLASSES} 
        for num, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)

            init_pred = output.max(1, keepdim=True)[1] 
            if init_pred.item() != target.item():
                continue
            loss = self.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            if attack == "fgsm":
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            elif attack == "noise":
                perturbed_data = self.noise_attack(data, epsilon)
            elif attack == "semantic":
                perturbed_data = self.semantic_attack(data)
        
            output = self.model(perturbed_data)         
            # prediction = torch.max(output, 1)

            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                    # adv_examples.append(adv_ex)

            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                    # adv_examples.append(adv_ex)

            # test different class
            for label, predict in zip(target, final_pred):
                if label == predict:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1    

        final_acc = correct/float(len(self.test_loader))
        if attack == "fgsm" or attack == "noise":
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(self.test_loader), final_acc))
        else:
            print("semantic: \tTest Accuracy = {} / {} = {}".format(correct, len(self.test_loader), final_acc))
        
        # accuracy for different class 
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
        plt.figure(figsize=(5,5))
        x, y = zip(*correct_pred.items())
        # print(len())
        # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
        plt.bar(np.arange(len(x)), y)
        plt.xticks(np.arange(len(x)), x)
        plt.title("Accuracy for each class (epsilon = {})".format(epsilon))
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.show()

        return final_acc, adv_examples


    def test_attack_F1(self, epsilon, attack):
        """
            test attack on the defense model
        """
        self.model_defense.eval()
        correct = 0
        adv_examples = []
        correct_pred = {classname: 0 for classname in CLASSES}
        total_pred = {classname: 0 for classname in CLASSES} 
        for num, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model_defense(data)

            init_pred = output.max(1, keepdim=True)[1] 
            if init_pred.item() != target.item():
                continue
            loss = self.criterion(output, target)
            self.model_defense.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            if attack == "fgsm":
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            elif attack == "noise":
                perturbed_data = self.noise_attack(data, epsilon)
            elif attack == "semantic":
                perturbed_data = self.semantic_attack(data)
        
            output = self.model_defense(perturbed_data)
            # prediction = torch.max(output, 1)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

            # test different class
            for label, predict in zip(target, final_pred):
                if label == predict:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1    

        final_acc = correct/float(len(self.test_loader))
        if attack == "fgsm" or attack == "noise":
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(self.test_loader), final_acc))
        else:
            print("semantic: \tTest Accuracy = {} / {} = {}".format(correct, len(self.test_loader), final_acc))

        # accuracy for different class 
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        return final_acc, adv_examples



# ======================================================= save and run =======================================================================================

    def save(self):
        """
            save the trained model to designated path
        """
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        """
            train the model and run the trained model with the regular testing function 
        """
        self.load_data()
        self.load_model()
        accuracy = 0
        #for epoch in range(1, self.epochs + 1):
        #self.scheduler.step(epoch)
        #print("\n===> epoch: %d/200" % epoch)
        #train_result = self.train()
        #print(train_result)
        test_result = self.test()
        accuracy = max(accuracy, test_result[1])
        print(accuracy)
        #if epoch == self.epochs:
            #print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
            #self.save()


if __name__ == '__main__':
    main()




