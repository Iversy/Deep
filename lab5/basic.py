import torch
import torchvision
import torch.nn as nn

from ock import inti
class BasicCNN(nn.Module):
    def __init__(self, num_classes=3, init_type=None):
        super(BasicCNN, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        
        self.relu = nn.ReLU()
        
        if init_type:
            self.apply(lambda m: inti(m))
            
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        
        x = x.view(x.size(0), -1)
        
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x