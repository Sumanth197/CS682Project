import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 10


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
#         # Modification BaseNet
#         self.basenet = nn.Sequential(
#             nn.Conv2d(1, 64, 5),
#             nn.Conv2d(64, 32, 3),
#             nn.Conv2d(32, 16, 3),
#             nn.MaxPool2d(2, 2)

#         )
#         self.bs_classifier = nn.Sequential(
#             nn.Linear(16 * 10 * 10, 120),
#             nn.Linear(120, 256),
#             nn.Linear(256, NUM_CLASSES)
#         )
        
        # Modification BatchNorm
#         self.modNet = nn.Sequential(
#             nn.Conv2d(1, 64, 3),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(inplace = True),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, 3),
#             nn.BatchNorm2d(num_features = 128),
#             nn.ReLU(inplace = True),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(128, 256, 3),
#             nn.ReLU(inplace = True)

#         )
#         self.md_classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(256 * 3 * 3,  256),
#             nn.ReLU(inplace = True),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace = True),
#             nn.Dropout(0.1),
#             nn.Linear(128, NUM_CLASSES)
#             #nn.Softmax(dim = 1)
#         )
        
        
        # My Model
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 24, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, NUM_CLASSES)



    def forward(self, x):
        
        
#         # BaseNet Forward
#         output = self.basenet(x)
#         output = torch.flatten(output, 1)
#         output = self.bs_classifier(output)
        
#         # Modified Network Forward
#         output = self.modNet(x)
#         output = torch.flatten(output, 1)
#         output = self.md_classifier(output)
#         output = F.log_softmax(output, dim=1)
#         return output
        
        # My Model Forward
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        output = F.log_softmax(self.fc2(x), dim=1)
        return output
