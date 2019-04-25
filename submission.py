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
import csv
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
import torch.nn.functional as func

import random

from torchvision import models
import torch.nn as nn
import torch.optim as optim

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

img_size = (224, 224)
#Transformations
transformList = []
transformList.append(transforms.Resize(img_size))
transformList.append(transforms.ToTensor())
transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))      
transformSequence=transforms.Compose(transformList)

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 196608
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.densenet = torchvision.models.densenet121(pretrained=False)
        print(self.densenet.classifier.in_features)
        self.densenet.classifier = nn.Sequential(
            nn.Linear(self.densenet.classifier.in_features, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.densenet(x)
        return x


model = Net(14)
model.load_state_dict(torch.load('src/densenet_5epochs.pt', map_location=device))
model = model.to(device)



class TestDataset(Dataset):
    def __init__(self, file_location, transform=None):
        image_files = []
           
        file_df = pd.read_csv(file_location)
        image_files = file_df['Path'].tolist()
        self.image_files = image_files
        self.transform = transform

    def __getitem__(self, index):
        location = self.image_files[index]
        image = Image.open(location).convert('RGB')
        study = location.rsplit('/', 1)[0]
        
        if self.transform is not None:
            image = self.transform(image)
        return image,study

    def __len__(self):
        return len(self.image_files)


testDataset = TestDataset(input_file_name, transformSequence) 
test_data_loader = DataLoader(testDataset, batch_size= 8, shuffle = False)

results_df = pd.DataFrame(columns=['Cardiomegaly','Edema','Consolidation','Atelectasis','Pleural Effusion', 'Study'])

model.eval()
with torch.no_grad():
    #print(image_files)
    for image, location in test_data_loader:
        print(image.shape)
        image = image.to(device)
        pred_label = model(image)
        #print(location)
        locationnp  = np.asarray(location)
        data = np.column_stack((pred_label.cpu().numpy()[:,[2,5,6,8,10]],locationnp))
        df = pd.DataFrame(data, columns=['Cardiomegaly','Edema','Consolidation','Atelectasis','Pleural Effusion', 'Study'])
        results_df = results_df.append(df)
#print(results_df)
results_df = results_df.groupby(by='Study').max()[['Cardiomegaly','Edema','Consolidation','Atelectasis','Pleural Effusion']].reset_index()
results_df.to_csv(output_file_name, index=False)

