""" This is a python script for simple_net a simple convolutional neural network for image classification task. The architecture is pretty obvious. This model 
has been trained on CIFAR10 dataset and have achieved an accuracy of 77%. Not good enough but feel free to edit as you like and learn"""


import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import numpy as np
from torchmetrics import Accuracy as acc



num_channels = 3
height=32
width=32


class simple_net(nn.Module):

    def __init__(self):

        super(simple_net,self).__init__()

        self.feature_extractor_1=nn.Sequential(

            nn.Conv2d(in_channels=num_channels,out_channels=32,kernel_size=3,stride=1,padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(stride=2,kernel_size=2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)

            )
        

        self.fc_layer=nn.Sequential(

            nn.Flatten(),

            nn.Linear(128*8*8,2048),
            nn.BatchNorm1d(2048),

            nn.ReLU(),

            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),

            nn.ReLU(),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,10)

            )


    def forward(self,x):

        x=self.feature_extractor_1(x)
        x=self.fc_layer(x)

        return x


model=simple_net()
optimiser = optim.Adam(model.parameters(), lr=0.01)
lf = nn.CrossEntropyLoss()

""" i have not used inbuilt torch vision dataset. I have included the code for creating your own torch dataset."""

class my_data(torch.utils.data.Dataset):


    def __init__(self,inputs, labels):

        self.labels=labels

        self.inputs=inputs


    def __len__(self):

        return len(self.inputs)


    def __getitem__(self, index):

        x = self.inputs[index]

        y=self.labels[index]



        return x,y
    

def open_file(file):

    with open(file, "rb") as f:

        train_data = pickle.load(f, encoding="latin1")

    serial_data = train_data['data']

    labels = train_data['labels']

    x=[]

    for i in range(10000):

        x.append(np.reshape(serial_data[i],(3,32,32)))


    return x,labels


x=[]
labels=[]

ver = (1,2,3,4,5)

target_files = {f"data_batch_{v}" for v in ver}
for name in os.listdir((r"**********************************")):                        # do change the directory for the cifar files. this is the path to the folder containing pickle files.
    if os.path.basename(name) in target_files :
        a,b= open_file(os.path.join(r"*********************************",name))         #this is the path to the folder containing pickle files.
        x.append(a)
        labels.append(b)
        
x=np.array(x)
x=np.concatenate(x,axis=0)

labels=np.array(labels)
labels=np.concatenate(labels,axis=0)


input_data = my_data(x,labels)

input_generator = torch.utils.data.DataLoader(input_data, batch_size=128 , shuffle= True)


model.train()
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)     
    model.to(device)



for epoch in range(50):

    running_loss = 0.0

    for inputs, targets in input_generator:
        inputs = inputs.float() / 255.0
        targets = targets.long()
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = lf(outputs, targets)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(input_data)
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")


model.eval()

accuracy_metric = acc('multiclass',num_classes=10)

with open(r"************************", "rb") as f:    #again path to the cifar10 data folder

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




