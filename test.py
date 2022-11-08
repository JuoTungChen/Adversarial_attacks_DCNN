"""
EN.520.665 Machine Perception
Project 1
Team Member: Danny (Iou-Sheng) Chang, Juo-Tung Chen

Code Instructions:
# 
1. Modify main to call desired functions:
    (i) if you want to show regular testing results from the pretrained model 
        -> call solver.run() in main and uncomment the line with testing batch size = 100
    (ii) if you want to show testing results under attack 
        -> call solver.run_attack() in main and uncomment the line with testing batch size = 1

2. Saving and Loading Model
    (i) save trained model
        in the run function in Solver class
        -> uncomment the last line: self.save()
    (ii) loading pretrained model
        in the load_model function within the Solver class
        -> modify the path in self.model = torch.load('model_1.pth')

3. changing parameters for the attack methods
    -> for FGSM and noise attack, just change the epsilons in the run_attack function
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
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs to train for')
    ## original testing -> test size = 100 (uncomment this line to show regular testing results)
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    
    ## run attack -> test size = 1 (uncomment this line to show attack results)
    parser.add_argument('--trainBatchSize', default=50, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=1, type=int, help='testing batch size')

    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    ## Solver Section
    solver = Solver(args)
    # solver.run()
    # solver.img_attack_show()
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
            load GoogLeNet model or load pretrained model 
        """
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        ## Uncommand if one wish to retrain the network
        # self.model = GoogLeNet().to(self.device)

        ## Load previously trained model
        self.model = torch.load('model_1.pth')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)

        self.model_defense = torch.load('modelF1.pth') 
        self.optimizer_d = optim.Adam(self.model_defense.parameters(), lr=self.lr)
        self.scheduler_d = optim.lr_scheduler.MultiStepLR(self.optimizer_d, milestones=[75, 150], gamma=0.5)

        self.criterion = nn.CrossEntropyLoss().to(self.device)



    def train(self):
        """
            train the GoogLeNet model 
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

            # train_correct incremented by one if predicted corrected
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total



    def test(self):
        """
            test function using the testing set from the CIFAR-10
            display the accuracy for each class of objects
        """
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():           
            # prepare to count predictions for each classes
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
            # test different classes
                for label, predict in zip(target, prediction[1]):
                    if label == predict:
                        correct_pred[CLASSES[label]] += 1
                    total_pred[CLASSES[label]] += 1    
        # accuracy for different classes
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        return test_loss, test_correct / total



# ===================================================== FGSM Attack ===============================================================================

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
        


# ===================================================== show attack images ===============================================================================

    def img_attack_show(self):
        # get some random training images
        self.load_data()
        self.load_model()

        dataiter = iter(self.train_loader)
        data, labels = next(dataiter)
        data = data.to(self.device)
        data = data[0:4]
        # create attacked images
        epsilon = 0.3
        noise_img = self.noise_attack(data, epsilon)
        semantic_img = self.semantic_attack(data)

        rows = 2
        columns = 2
        # show images
        grid = torchvision.utils.make_grid(data[0:4])
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()

        fig = plt.figure()
        fig.suptitle(r'Original CIFAR-10 Images and Adversarial Examples', fontsize=18, y=0.95)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r"Original Image from CIFAR-10")

        grid = torchvision.utils.make_grid(noise_img)
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()
        fig.add_subplot(rows, columns, 2)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r'Image under Noise Attack ($\varepsilon$ = {})'.format(epsilon))

        grid = torchvision.utils.make_grid(semantic_img)
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()
        fig.add_subplot(rows, columns, 3)

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r"Image under Semantic Attack")
 
        self.model.eval()
        for data, target in self.train_loader:

            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)

            loss = self.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        grid = torchvision.utils.make_grid(perturbed_data[0:4])
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()
        fig.add_subplot(rows, columns, 4)

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r'Image under FGSM Attack ($\varepsilon$ = {})'.format(epsilon))
        plt.show()



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
            train_result = self.train_F1(modelF1, optimizerF1)
            # test_result = self.test_attack_F1(modelF1, 0.1, "fgsm", cnt)

        # save the trained defense model
        model_out_path = "modelF1_1.pth"
        torch.save(modelF1, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))



    def train_F1(self, model, optimizer):
        """
            training the new defense network model
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

            # train_correct incremented by one if predicted correctly
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
        # epsilons = [0.1, 0.2]


        ## run through all of the attcks
        # for attack in ("fgsm","noise","semantic"):
        for attack in ("fgsm","semantic"):
            accuracies = []
            accuracies_d = []
            examples = []

            if attack == "fgsm":
                 # test attack on model without defense
                print("\nresults for original GoogleNet:")
                plt.figure()
                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(r'Accuracy for Each Class (without defense) under FGSM Attack', fontsize=18, y=0.95)
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                cnt = 0
                for eps in epsilons:
                    acc, ex = self.test_attack(eps, attack, cnt)      
                    accuracies.append(acc)
                    cnt+=1

                 # test attack on defense network
                print("\nresults for GoogleNet with defense:")
                plt.figure()
                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(r'Accuracy for Each Class (with defense) under FGSM Attack', fontsize=18, y=0.95)
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                cnt = 0
                for eps in epsilons:
                    acc, ex = self.test_attack_F1(eps, attack, cnt)     
                    accuracies_d.append(acc)
                    cnt+=1

            
                # plot the results
                plt.figure(figsize=(5,5))
                plt.plot(epsilons, accuracies, "*-", label = r'GoogleNet w/o defense')
                plt.plot(epsilons, accuracies_d, "*-", label = r'GoogleNet w/ defensive-distillation')
            
                plt.title(r"FGSM Attack Accuracy Comparison with and without Defense")
                plt.legend(loc='center right')
                plt.xlabel(r"Epsilon")
                plt.ylabel(r"Accuracy")
                plt.show() 

            elif attack == "noise":
                plt.figure()
                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(r'Accuracy for Each Class (without defense) under Noise Attack', fontsize=18, y=0.95)
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                cnt = 0
                for eps in epsilons:
                    acc, ex = self.test_attack(eps, attack, cnt)       # test attack on regular NN
                    accuracies.append(acc)
                    cnt+=1
                # plt.show() 

            elif attack == "semantic":
                plt.figure()
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                cnt = 0
                acc, ex = self.test_attack(0, attack, cnt)       # test attack on regular NN
                accuracies.append(acc)
                cnt+=1
                # plt.show() 





    def test_attack(self, epsilon, attack, cnt):
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
            x, y = zip(*correct_pred.items())
            ax = plt.subplot(3, 3, cnt + 1)
            # print(len())
            # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
            ax.bar(np.arange(len(x)), y)
            ax.set_xticks(np.arange(len(x)), x)
            ax.set_title(r'$\varepsilon$ = {} (Total accuracy = {})'.format(epsilon, final_acc))
            ax.set_ylim([0, 1100])
            ax.set_xlabel(r"Class")
            ax.set_ylabel(r"Accuracy")
            # plt.show(block=False)
        else:
            print("Semantic: \tTest Accuracy = {} / {} = {}".format(correct, len(self.test_loader), final_acc))
            x, y = zip(*correct_pred.items())
            # print(len())
            # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
            plt.bar(np.arange(len(x)), y)
            plt.xticks(np.arange(len(x)), x)
            plt.title(r"Accuracy for Each Class (without defense) under Semantic Attack (Total accuracy = {})".format(final_acc), fontsize=18, y=0.95)
            plt.ylim([0, 1100])
            plt.xlabel(r"Class")
            plt.ylabel(r"Accuracy")
            # plt.show(block=False)
        
        # accuracy for different class 
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
        

        return final_acc, adv_examples


    def test_attack_F1(self, epsilon, attack, cnt):
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

        x, y = zip(*correct_pred.items())
        ax = plt.subplot(3, 3, cnt + 1)
        # print(len())
        # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
        ax.bar(np.arange(len(x)), y)
        ax.set_xticks(np.arange(len(x)), x)
        ax.set_title(r'$\varepsilon$ = {} (Total accuracy = {})'.format(epsilon, final_acc))
        ax.set_ylim([0, 1100])
        ax.set_xlabel(r"Class")
        ax.set_ylabel(r"Accuracy")

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