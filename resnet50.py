import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy as acc
import os
import pickle
import numpy as np
import time

class brick(nn.Module):

    expansion = 4

    def __init__(self, in_channel, planes, stride = 1, downsample = None):
        super(brick, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=planes ,kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(in_channels=planes, out_channels=planes,kernel_size=3,stride= stride , padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion ,kernel_size=1,stride=1, bias=False, padding=0),
            nn.BatchNorm2d(planes * 4),

        )      
        self.relu = nn.ReLU()  
        self.downsample = downsample


    def forward(self, x):
        identity = x

        x = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = x+identity

        x= self.relu(x) 

        return x
    

class Resnet50(nn.Module):
    def __init__(self, num_classess = 10):
        super(Resnet50 , self).__init__()
        self.in_con = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.block1 = self.make_block( brick, 64, 64, 2, 3)
        self.block2 = self.make_block( brick, 256, 128, 2, 3)
        self.block3 = self.make_block( brick, 512, 256, 2, 3)
        self.block4 = self.make_block( brick, 1024, 512, 2, 3)
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_layer = nn.Linear(in_features=2048, out_features=num_classess, )

    def make_block(self, brick,in_layer,  layers, stride = 1, blocks=1):

        out_layer = layers * brick.expansion

        downsamples = None
        if stride != 1 or in_layer != layers:
            downsamples=nn.Sequential(
                nn.Conv2d(in_channels=in_layer, out_channels=layers *4 , kernel_size=1, stride=stride, bias=False ),
                nn.BatchNorm2d(layers * 4)
            )
        x=[]
        x.append(brick(in_layer ,layers,stride, downsamples))

        for i in range(1,blocks):
            x.append(brick(out_layer,layers,1))

        return nn.Sequential(*x)

    def forward(self, x):
        x= self.in_con(x)
        x= self.pool1(x)  
        x= self.block1(x)  
        x= self.block2(x)
        x= self.block3(x)  
        x= self.block4(x)  
        x= self.pool2(x)  
        x= torch.flatten(x, 1)
        x= self.fc_layer(x)
        return x 




class my_data(torch.utils.data.Dataset):
    def __init__(self, input_data, labels):
        self.input_data=input_data
        self.labels=labels

    def __len__(self):
        return len(self.input_data)
    

    def __getitem__(self, index):
        return self.input_data[index],self.labels[index]
    

def open_file(path):
    with open(path,"rb") as f:
        temp = pickle.load(f,encoding='latin1')
    x=[]
    serial_data = temp['data']
    labels = temp['labels']
    for i in range(10000):
        x.append(np.reshape(serial_data[i],(3,32,32)))
    return x, labels


if __name__ == '__main__':
    t1 = time.time()

    model = Resnet50()
    model

    path = r"D:\Cifar-benchmark\cifar-10-batches-py"

    target_files = {f"data_batch_{v}" for v in [1,2,3,4,5] }

    input_data = []
    output_data = []

    for name in os.listdir(path):
        if name in target_files:
            a,b = open_file(os.path.join(path,name))  
            input_data.append(a)
            output_data.append(b)

    input_data = np.concatenate(input_data,axis=0)
    output_data= np.concatenate(output_data,axis=0)

    input_data = torch.tensor(input_data).float() 
    output_data = torch.tensor(output_data).long()

    input_data =  my_data(input_data, output_data)
    input_generator = torch.utils.data.DataLoader(input_data, batch_size= 256, shuffle=True, num_workers= 4, pin_memory=True)
    optimiser = optim.Adam(model.parameters(),lr=0.01)
    loss_metric = nn.CrossEntropyLoss()
    device = 'cuda'




    model.to(device)


    for epochs in range(50):
        running_loss = 0.0
        for input, target in input_generator:
            input = input/255.0

            
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            optimiser.zero_grad()
            loss = loss_metric(output , target)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * input.size(0)

        epoch_loss = running_loss / len(input_data)

        print(f"Epoch {epochs+1} loss: {epoch_loss:.4f}")
    t2 = time.time()
    print((t2-t1))
        

    model.eval()

    from torchmetrics import Accuracy as acc
    accuracy_metric = acc('multiclass',num_classes=10)
    with open(r"D:\Cifar-benchmark\cifar-10-batches-py\test_batch", "rb") as f:

        test_data = pickle.load(f, encoding="latin1")

    serial_test_data = test_data['data']

    test_labels = test_data['labels']
    t_x=[]

    for i in range(10000):

        t_x.append(np.reshape(serial_test_data[i],(3,32,32)))



    test_data = my_data(t_x, test_labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)

    model.eval()
    accuracy = accuracy_metric.to(device)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float() / 255.0  # normalize the inputs
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            accuracy.update(outputs, targets)
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    final_accuracy = accuracy.compute()
    print(f'Test Accuracy: {final_accuracy:.4f}')