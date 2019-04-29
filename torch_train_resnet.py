import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
import torchvision.models as tmodels

from PIL import Image


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor,Resize

from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random
from sklearn.metrics import auc, roc_auc_score, accuracy_score, f1_score


use_gpu = torch.cuda.is_available()
from utils import *

from PIL import Image


########################

threshold = 0.5

#######################

print("STAGE 1")

baseFolder  = "/home/santhosr/Documents/Chexpert"

cols = ['Path',
 'Sex',
 'Age',
 'View',
 'AP/PA',
 'No_Finding',
 'Enlarged_Cardiomediastinum',
 'Cardiomegaly',
 'Lung_Opacity',
 'Lung_Lesion',
 'Edema',
 'Consolidation',
 'Pneumonia',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Effusion',
 'Pleural_Other',
 'Fracture',
 'Support_Devices']

def freezeBodyRes(model, unfreeze = False):
    
    if unfreeze==False:
        for param in model.densenet.parameters():
            param.requires_grad = False
            
        for param in model.densenet.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.densenet.parameters():
            param.requires_grad = True


class LungDataset(Dataset):
    def __init__(self, baseFolder, file, transform=None, type="All"):
        
        self.baseFolder = baseFolder
        
        self.df = pd.read_csv(file)

        self.df = cleanLabelFile(self.df, cols[5:],sumCount = False)

        # if len(self.df >2000):
        #     self.df = self.df[:2000]
        
        self.transform = transform

    def __getitem__(self, index):
        
        x = self.df.iloc[index]
        path = os.path.join(self.baseFolder, x['Path'])
        image = Image.open(path).convert('RGB')
        
        label = np.array(x[cols[5:]])
        
        
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(list(label))

    def __len__(self):
        return len(self.df)


print("STAGE 2")

batchSize = 50
img_size = (224, 224)

#Transformations
transformList = []
transformList.append(transforms.Resize(img_size))
transformList.append(transforms.ColorJitter(brightness=.1, hue=.1, saturation=.1, contrast = 0.15))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))      
transformSequence=transforms.Compose(transformList)


trainData = LungDataset(baseFolder, 'trainNew.csv',transformSequence)
validData = LungDataset(baseFolder, 'validNew2.csv',transformSequence)

trainLoader = DataLoader(trainData, batch_size= batchSize, shuffle = True)
validLoader = DataLoader(validData, batch_size= batchSize, shuffle = True)


device = torch.device('cuda:0')


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.densenet = torchvision.models.resnet152(pretrained=True)
        self.densenet.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.densenet(x)
        return x


modelCount = 0
num_classes = 14
learning_rate = 0.0001

model = Net(num_classes).cuda()
criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.Adam (model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
print("STAGE 3")

def testModel(model,validLoader):
    
    predictions =[]
    labels = []

    model.eval()

    with torch.no_grad():
        for image, label in validLoader:
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
            
            predictions.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
            
            
    labelA = np.concatenate(labels)
    predA = np.concatenate(predictions)


    print("###########################")
    for i in range(num_classes):
        
        label = labelA[:,i]
        pred = predA[:,i]
        
        auc = roc_auc_score(label, pred)
        
        ind = pred >= threshold
        
        pred[ind] = 1.0
        pred[~ind] = 0.0
        
        acc = accuracy_score(pred, label)
        
        
        
        print("{m: <26} : AUC - {a:<5} Acc - {b: <6} ".format(m=cols[(5+i)], a=np.round(auc,3), b=np.round(acc,3)))
    
    print("###########################")

    model.train()



def train(net, optimizer, criterion, train_loader, test_loader, epochs, earlyBreak = None):
    model = net.to(device)
    total_step = len(train_loader)
    global modelCount
    print("Train Start")
    overall_step = 0
    for epoch in range(epochs):
    
        for i, (images, labels) in enumerate(train_loader):

            if earlyBreak!=None:
                if earlyBreak ==i:
                    break
            

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            _, prediction = torch.max(outputs,1)

            overall_step+=1
            if (i+1) % 200 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))           

        testModel(model, validLoader)
        torch.save(model.state_dict(), "resnet_model4_14class_" +str(modelCount) + "epochs.pt")
        modelCount += 1


### TRAINING PROCEDURE              
print("TRAINING START")

freezeBodyRes(model)
optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
train(model, optimizer, criterion, trainLoader, validLoader, 2, earlyBreak = 1500 )


freezeBodyRes(model,unfreeze = True)
optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
train(model, optimizer, criterion, trainLoader, validLoader, 3)


optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
train(model, optimizer, criterion, trainLoader, validLoader, 2)



optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
train(model, optimizer, criterion, trainLoader, validLoader, 1)

# model.load_state_dict(torch.load('densenet_model2_5class_8epochs.pt'))


# freezeUpto(model,paramCount = 70)
# optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
# train(model, optimizer, criterion, trainLoader, validLoader, 3)


# freezeUpto(model,paramCount = 200)
# optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.000005, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
# train(model, optimizer, criterion, trainLoader, validLoader, 2)


# freezeUpto(model,paramCount = 300)
# optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad,model.parameters()), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)              
# train(model, optimizer, criterion, trainLoader, validLoader, 2)



# torch.save(model.state_dict(), "densenet_model2_Final.pt")
