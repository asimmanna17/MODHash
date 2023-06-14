import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 64)
        

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x, indices1 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = F.relu(self.conv2(x))
        x, indices2 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x, indices3 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x_fc6 = x
        x = F.dropout(x)
        x = F.relu(self.fc2(x))#x
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = F.relu(self.fc4(x))
        return x  # ,indices1,indices2,indices3


class Modality(nn.Module):
    def __init__(self,nModality, code):
        super(Modality, self).__init__()
        self.l1 = nn.Linear(64, nModality)
        self.softmax = nn.LogSoftmax(dim=1)
        self.l2 = nn.Linear(64, code)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x_c = self.l1(input)
        x_c = self.softmax(x_c)
        #print(x_c)
        x_h = self.tanh(self.l2(input))
        return x_c, x_h   
    
class Organ(nn.Module):
    def __init__(self,nOrgan, code):
        super(Organ, self).__init__()
        self.l1 = nn.Linear(64, nOrgan)
        self.softmax = nn.LogSoftmax(dim=1)
        self.l2 = nn.Linear(64, code)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x_c = self.l1(input)
        x_c = self.softmax(x_c)
        x_h = self.tanh(self.l2(input))
        return x_c, x_h  
    
class Disease(nn.Module):
    def __init__(self,nDisease, code):
        super(Disease, self).__init__()
        self.l1 = nn.Linear(64, nDisease)
        self.softmax = nn.LogSoftmax(dim=1)
        self.l2 = nn.Linear(64, code)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x_c = self.l1(input)
        x_c = self.softmax(x_c)
        #print(x_c)
        x_h = self.tanh(self.l2(input))
        return x_c, x_h   
    
class Discriminator(nn.Module):
    def __init__(self, code):
        super(Discriminator, self).__init__()
        self.conv1d = nn.Conv1d(2, 1, 1)
        #self.fc1 = nn.Linear(code*3, 32)
        self.fc1 = nn.Linear(code, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        #self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.conv1d(x))
        x = self.relu(self.fc1(x))
        #x = F.relu(self.fc3(x))
        x = self.relu(self.fc3(x))
        return x