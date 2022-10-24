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
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')

## original testing -> test size = 100 (uncomment this line to show regular testing results)
    # parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')

## run attack -> test size = 1 (uncomment this line to show attack results)
    parser.add_argument('--testBatchSize', default=1, type=int, help='testing batch size')

    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)

    # solver.run()
    solver.run_attack()





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

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
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
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
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
    def fgsm_attack(self, input, epsilon, data_grad):
        # Create the perturbed image by adjusting each pixel of the input image based on the element-wise sign of the data gradient
        pert_out = input + epsilon * data_grad.sign()
        # Adding clipping to maintain [0,1] range
        pert_out = torch.clamp(pert_out, 0, 1)
        # Return the perturbed image
        return pert_out

    def test_attack(self, epsilon, attack):
        correct = 0
        adv_examples = []
        for num, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1] 
            if init_pred.item() != target.item():
                continue
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            if attack == "fgsm":
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            # elif attack == "ifgsm":
            #     perturbed_data = ifgsm_attack(data,epsilon,data_grad)
            # elif attack == "mifgsm":
            #     perturbed_data = mifgsm_attack(data,epsilon,data_grad)
        
            output = self.model(perturbed_data)
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

        final_acc = correct/float(len(self.test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(self.test_loader), final_acc))

        return final_acc, adv_examples

    def run_attack(self):
        self.load_data()
        self.load_model()
        epsilons = [0,0.007,0.01,0.02,0.03,0.05,0.1,0.2,0.3]
        # for attack in ("fgsm","ifgsm","mifgsm"):
        attack = "fgsm"
        accuracies = []
        examples = []
        for eps in epsilons:
            acc, ex = self.test_attack(eps, attack)       # call the test_attack function with various epsilons
            accuracies.append(acc)
            examples.append(ex)
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.title(attack)
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()

        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons),len(examples[0]),cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig,adv,ex = examples[i][j]
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex)
        plt.tight_layout()
        plt.show()


# ====================================================== DeepFool Attack =======================================================================================

    def DeepFool(image,model,num_classes = 10,maxiter = 50,min_val = -1,max_val = 1,true_label = 0):
        '''   Our VGG model accepts images with dimension [B,C,H,W] and also we have trained the model with the images normalized with mean and std.
                Therefore the image input to this function is mean ans std normalized.
                min_val and max_val is used to clamp the final image
        '''
        image = torch.from_numpy(image)

        is_cuda = torch.cuda.is_available()
        
        if is_cuda:
            model = model.cuda()
            image = image.cuda()

        model.train(False)
        f = model(Variable(image[None,:,:,:],requires_grad = True)).data.cpu().numpy().flatten()
        I = (np.array(f)).argsort()[::-1]
        label = I[0]  # Image label

        input_shape = image.cpu().numpy().shape

        if(label != true_label): # Find adversarial noise only when Predicted label = True label
            print("Predicted label is not the same as True Label ")
            return np.zeros(input_shape),0,-1,0, np.zeros(input_shape)
        
        
        pert_image = image.cpu().numpy().copy()   # pert_image stores the perturbed image
        w = np.zeros(input_shape)                # 
        r_tot = np.zeros(input_shape)   # r_tot stores the total perturbation
        
        pert_image = torch.from_numpy(pert_image)
        
        if is_cuda:
            pert_image = pert_image.cuda()
            
        x = Variable(pert_image[None,:,:,:],requires_grad = True)
        fs = model(x)
        fs_list = [ fs[0,I[k]] for k in range(num_classes) ]
        
        k_i = label  # k_i stores the label of the ith iteration
        loop_i = 0
        
        
        while loop_i < maxiter and k_i == label:
            pert = np.inf
            fs[0,I[0]].backward(retain_graph = True)  
            grad_khat_x0 = x.grad.data.cpu().numpy()  # Gradient wrt to the predicted label
            
            for k in range(1,num_classes):
            if x.grad is not None:
                x.grad.data.fill_(0)
            fs[0,I[k]].backward(retain_graph = True)
            grad_k = x.grad.data.cpu().numpy()
            
            w_k = grad_k - grad_khat_x0
            f_k = (fs[0,I[k]] - fs[0,I[0]]).data.cpu().numpy()
            
            pert_k = abs(f_k)/(np.linalg.norm(w_k.flatten()))
            
            if pert_k < pert:
                pert = pert_k
                w = w_k

            r_i = (pert)*(w/np.linalg.norm(w.flatten()))
            r_tot = np.float32(r_tot +  r_i.squeeze())

            if is_cuda:
            pert_image += (torch.from_numpy(r_tot)).cuda()
            else:
            pert_image += toch.from_numpy(r_tot)
            
            x = Variable(pert_image,requires_grad = True)
            fs = model(x[None,:,:,:])
            k_i = np.argmax(fs.data.cpu().numpy().flatten())
            
            
            loop_i += 1
        pert_image = torch.clamp(pert_image,min_val,max_val) 
        pert_image = pert_image.data.cpu().numpy()
    
        return r_tot,loop_i,label,k_i,pert_image



# =============================== Visualisation =======================================================================================

    mean = np.array([0.485, 0.456, 0.406])  # Mean and std of the data
    mean = mean[:,None,None]
    std = np.array([0.229, 0.224, 0.225])
    std = std[:,None,None]

    def show_image(img1,img2,mean,std):
    img1 = img1*std + mean   # unnormalize
    img2 = img2*std + mean
    img1 = img1.clip(0,1)
    img2 = img2.clip(0,1)
    
    img1 = np.transpose(img1,(1,2,0))
    img2 = np.transpose(img2,(1,2,0))
    
    
    noise = img2- img1
    noise = abs(noise)
    noise =noise.clip(0,1)

    disp_im = np.concatenate((img1,img2,2*noise),axis = 1)
    plt.imshow(disp_im)


    idx = np.random.randint(0,10000)

    sample_image = test_images[idx]
    img_label = test_label[idx]
    # since the image input to the DeepFool function in normalized according to the above mean and std therefore the min_val and max_val is not {0,1} but {-2.117,2.64}
    r_tot,loop_i,label,k_i,pert_image = DeepFool(sample_image,model,num_classes = 10,maxiter = 50,min_val = -2.117,max_val = 2.64,true_label = img_label)

    print("Clean Label: " ,classes[label]," Adversarial Label:" ,classes[k_i])
    show_image(sample_image,pert_image,mean,std)







    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
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
        #if epoch == self.epochs:
            #print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
            #self.save()


if __name__ == '__main__':
    main()




