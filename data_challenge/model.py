import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models  
from torchvision.models import  googlenet
import torch.optim as optim

nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class GoogleNetModel(nn.Module):
    def __init__(self):
        super(GoogleNetModel, self).__init__()

        #backbone
        model = googlenet(pretrained=True)
        
        modules = list(model.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        #adding the fully connected layers
        nb_features = model.fc.in_features
        self.fc1 = nn.Linear(nb_features , 1024)
        self.fc2 = nn.Linear(1024, nclasses)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.2)
    

        # defining different learning rates for different parts of the model
        self.optimizer = optim.SGD([
        {'params': self.backbone.parameters(), 'lr': 0.001},  #  for the backbone
        {'params': self.fc1.parameters(), 'lr': 0.01},         # for fc1
        {'params': self.fc2.parameters(), 'lr': 0.01}          # for fc2
    ], momentum = 0.9) 
    def forward(self, x):

        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        return x


# This was the model implemented with freezing the first 10 layers of googlenet

# class GoogleNetModel_freeze(nn.Module):
#     def __init__(self):
#         super(GoogleNetModel, self).__init__()

#         model = googlenet(pretrained=True)

        #freeze only the 10 first layers
#         for idx, child in enumerate(model.children()):
#             if idx < 10:
#                 for param in child.parameters():
#                     param.requires_grad = False
#             else:
#                 break 

#         modules = list(model.children())[:-1]
#         self.backbone = nn.Sequential(*modules)

#         nb_features = model.fc.in_features
#         self.fc1 = nn.Linear(nb_features , 1024)
#         self.fc2 = nn.Linear(1024, nclasses)
#         self.dropout1 = nn.Dropout(p=0.5)
#         self.dropout2 = nn.Dropout(p=0.2)

#  
#         self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
#     def forward(self, x):
#         # Forward pass through the backbone
#         x = self.backbone(x)
#         x = x.view(x.size(0), -1)

#         x = F.sigmoid(self.fc1(x))
#         x = self.dropout1(x) 
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)

#         return x

