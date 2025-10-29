import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
import pickle
from torchmetrics import Accuracy as acc


class inception(nn.Module):
    def __init__(self, in_channels , _1x1, _3x3r, _3x3, _5x5r, _5x5 ,pool):
        super(inception, self).__init__()
        
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=_1x1, bias=False, kernel_size=1),
            nn.BatchNorm2d(_1x1),
            nn.ReLU()
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=_3x3r, bias=False, kernel_size=1),
            nn.BatchNorm2d(_3x3r),
            nn.ReLU(),
            nn.Conv2d(in_channels=_3x3r, out_channels=_3x3, bias=False, kernel_size=3, padding= 1),
            nn.BatchNorm2d(_3x3),
            nn.ReLU()
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=_5x5r, bias=False, kernel_size=1),
            nn.BatchNorm2d(_5x5r),
            nn.ReLU(),
            nn.Conv2d(in_channels=_5x5r, out_channels=_5x5, bias=False, kernel_size=5, padding= 2),
            nn.BatchNorm2d(_5x5),
            nn.ReLU()
        )

        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool,kernel_size=1, bias=False ),
            nn.BatchNorm2d(pool),
            nn.ReLU()
        )



    def forward(self, x):

        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.p3(x)
        x4 = self.p4(x)

        x = torch.cat((x1,x2,x3,x4), dim=1)

        return x
    


class InceptionNet(nn.Module):
    def __init__(self , block = inception, inchannel = 3 , num_classes = 10):
        super(InceptionNet, self).__init__()

        self.inblock = nn.Sequential( nn.Conv2d(in_channels=inchannel, out_channels= 192, kernel_size= 3,stride= 1,  bias=False, padding=1),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(),
                                    #  nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                                    #  nn.Conv2d(in_channels=64, out_channels=64, bias=False, kernel_size=1),
                                    #  nn.BatchNorm2d(64),
                                    #  nn.ReLU(),
                                    #  nn.Conv2d(in_channels=64, out_channels=192, bias=False, kernel_size=3, padding= 1),
                                    #  nn.BatchNorm2d(192),
                                    #  nn.ReLU(),

                                
        )
        self.pool1 = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1)
        self.inc3A = block(192,64,96,128,16,32,32)
        self.inc3B = block(256,128,128,192,32,96,64)
        self.pool2 = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1)
        self.inc4A = block(480,192,96,208,16,48,64)
        self.inc4B = block(512,160,112,224,24,64,64)
        self.inc4C = block(512,128,128,256,24,64,64)
        self.inc4D = block(512,112,144,288,32,64,64)
        self.inc4E = block(528,256,160,320,32,128,128)
        self.pool3 = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1)
        self.inc5A = block(832,256,160,320,32,128,128 )
        self.inc5B = block(832,384,192,384,48,128,128 )
        self.pool4 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(0.4)
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_classes),
        )

        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(4*4*128, 1024),
            nn.Linear(2*2*128, 1024),  #for cifar-10 data
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024,num_classes)

        )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(4*4*128, 1024),
            nn.Linear(2*2*128, 1024),  #for cifar-10 data
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024,num_classes)

        )
    def forward(self, x):
        x=self.inblock(x)
        x=self.pool1(x)
        x=self.inc3A(x)
        x=self.inc3B(x)
        x=self.pool2(x)
        x=self.inc4A(x)
        if self.training is True:
            x1 = self.aux1(x)
        x=self.inc4B(x)
        x=self.inc4C(x)
        x=self.inc4D(x)
        if self.training is True:
            x2 = self.aux2(x)
        x=self.inc4E(x)
        x=self.pool3(x) 
        x=self.inc5A(x)
        x=self.inc5B(x) 
        x=self.pool4(x)
        x=torch.flatten(x,1)
        x=self.out(x)
        if self.training is True:
            return x,x1,x2
        else:
            return x
        

class my_data(torch.utils.data.Dataset):
    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.input_data[index], self.labels[index]
    
    def __len__(self):
        return len(self.input_data)
    


def load_file(path):
    with open(path, "rb") as f:
        temp = pickle.load(f, encoding='latin1')
    x = []
    serial_data = temp['data']
    labels = temp['labels']
    x = serial_data.reshape(10000,3,32,32)
    return x, labels

if __name__ == '__main__':
    t1 = time.time()

    model = InceptionNet()
    model.train()

    path = r"D:\Cifar-benchmark\cifar-10-batches-py"

    target_files = {f"data_batch_{v}" for v in [1,2,3,4,5] }

    input_data = []
    output_data = []

    for name in os.listdir(path):
        if name in target_files:
            a,b = load_file(os.path.join(path,name))  
            input_data.append(a)
            output_data.append(b)

    input_data = np.concatenate(input_data,axis=0)
    output_data= np.concatenate(output_data,axis=0)

    input_data = torch.tensor(input_data).float() 
    output_data = torch.tensor(output_data).long()

    input_data =  my_data(input_data, output_data)
    input_generator = torch.utils.data.DataLoader(input_data, batch_size= 256, shuffle=True, num_workers= 4, pin_memory=True)
    lr = 0.001
    optimiser = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimiser, step_size= 8, gamma=0.96)
    loss_metric = nn.CrossEntropyLoss()
    device = 'cuda'
    model.to(device)

    for epochs in range(100):
        running_loss = 0.0
        for input,target in input_generator:
            input = input/255.0
            input = input.to(device)
            target = target.to(device)


            optimiser.zero_grad()
            output1, output2, output3 = model(input)

            loss1 = loss_metric(output1, target)
            loss2 = loss_metric(output2, target)
            loss3 = loss_metric(output3, target)
            loss = loss1+ 0.3 * loss2+ 0.3 * loss3
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * input.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(input_data)
        print(f"Epoch {epochs+1} loss: {epoch_loss:.4f}")
    t2 = time.time()
    print((t2-t1))


    save_path = "inception_cifar10.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    model.eval()
    accuracy_metric = acc('multiclass',num_classes= 10)
    accuracy = accuracy_metric.to(device)

    with open(r"D:\Cifar-benchmark\cifar-10-batches-py\test_batch", "rb") as f:

        test_data = pickle.load(f, encoding="latin1")

    serial_test_data = test_data['data']

    test_labels = test_data['labels']
    t_x=[]

    for i in range(10000):

        t_x.append(np.reshape(serial_test_data[i],(3,32,32)))



    test_data = my_data(t_x, test_labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float() / 255.0 
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
    
            accuracy.update(outputs, targets)

    final_accuracy = accuracy.compute()
    print(f'Test Accuracy: {final_accuracy:.4f}')
