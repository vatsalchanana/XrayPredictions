# -*- coding: utf-8 -*-

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
import cv2
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

from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random

use_gpu = torch.cuda.is_available()

"""Mount google drive"""

from google.colab import drive
drive.mount('/content/drive')

baseFolder  = "drive/My Drive/CheXpert Dataset/CheXpert-v1.0-small/"

#Utility functions for cleaning the data

def cleanLabel(x):
    
    labelCount = 0    
    if x.Pleural_Effusion == 1:
        labelCount += 1
    if x.Edema == 1:
        labelCount += 1
    if x.Cardiomegaly ==1:
        labelCount += 1
    if x.Pneumonia == 1:
        labelCount += 1
    return labelCount
    
    

def getLabel(x):
    
    if x.Pleural_Effusion ==1:
        return "Pleural_Effusion"
    elif x.Edema == 1:
        return "Edema"
    elif x.Cardiomegaly==1:
        return "Cardiomegaly"
    elif x.Pneumonia == 1:
        return "Pneumonia"
    else:
        return "None"

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
trainFile = pd.read_csv(os.path.join(baseFolder,'train.csv'), names = cols, header=0)
validFile = pd.read_csv(os.path.join(baseFolder,'valid.csv'), names = cols, header=0)

"""### Dataloader"""

labelMap = {"Pleural_Effusion":0, "Edema":1,"Cardiomegaly":2,"Pneumonia":3}

def getLabelDf(x):
    x = x[36:]          #To account for the extra "././" added before the Path variable
    x = df.loc[df.Path == x] 
    return labelMap[x.label.values[0]]

class LungDataset(Dataset):
    

    def __init__(self, csvFile, rootDir, transform = None):
        """
        Args:
            rootDir : Directory that has train, valid, train.csv and valid.csv
            csvFile : train.csv or valid.csv
        """
        
        self.rootDir = rootDir
        self.transform = transform
        
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
        
        self.df = pd.read_csv(os.path.join(rootDir,csvFile), names = cols, header=0)
        
        
        #Modifying the path variable
        self.df["Path"] = self.df.Path.apply(lambda x : x.replace('CheXpert-v1.0-small',"")[1:])
        
        #retaining important vars
        selectCols = ['Path',"View",'Sex',"Pleural_Effusion", "Edema","Cardiomegaly","Pneumonia"]
        self.df = self.df[selectCols]
        
        self.df['isClean'] = self.df.apply(lambda x : cleanLabel(x), axis = 1)
        #self.df["No"] = (self.df.isClean == 0).astype(int)
        #no_disease = self.df[self.df.isClean==0]
        #Retaining only samples with 1 disease
        self.df = self.df[self.df.isClean==1]
        #self.df = pd.concat([self.df,no_disease])
        
        #Creating the label variable
        self.df['label'] = self.df.apply(lambda x : labelMap[getLabel(x)], axis = 1)
                



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        imgPath = os.path.join( self.rootDir, self.df.iloc[idx].Path)
        image = Image.open(imgPath).convert('RGB')
        
        label = self.df.iloc[idx].label

        if self.transform:
            image = self.transform(image)

        return image, label

#Change train to valid if you want fast execution just for sanity checking the model (train image folders can be huge and can cause lame googledrive timout issues)
trainDataset = LungDataset('train.csv', baseFolder, transforms.Compose([Resize((256,256)), ToTensor()]))
validationDataset = LungDataset('valid.csv', baseFolder, transforms.Compose([Resize((256,256)), ToTensor()]))

e = next(iter(trainDataset))
e[0].size()

train_data_loader = DataLoader(trainDataset, batch_size= 64, shuffle = True, num_workers = 4)
validation_loader = DataLoader(validationDataset, batch_size= 64, shuffle = True, num_workers = 4)

for image, label in train_data_loader:
    print(label)
    
    break;



"""## Logger"""



LOG_DIR = './logs'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

!if [ -f ngrok ] ; then echo "Ngrok already installed" ; else wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip > /dev/null 2>&1 && unzip ngrok-stable-linux-amd64.zip > /dev/null 2>&1 ; fi

get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print('Tensorboard Link: ' +str(json.load(sys.stdin)['tunnels'][0]['public_url']))"

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
logger = Logger('./logs')

"""## Feedforward Network"""

from torchvision import models
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0')

input_size = 196608
hidden_size_1 = 1024
hidden_size_2 = 512
hidden_size_3 = 128
num_classes = 4
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1,hidden_size_2,hidden_size_3, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_3, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

model = NeuralNet(input_size, hidden_size_1,hidden_size_2,hidden_size_3, num_classes).to(device)
model = model.to(device)

#for image, label in train_data_loader:
 #   print(label)

"""### Training the model"""

learning_rate = 0.00003
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss(size_average = True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 25
# Train the model
def train(net, optimizer, criterion, trainLoader, test_loader, epochs, size, model_name,plot):
  model = net.to(device)
  overall_step = 0;
  for epoch in range(epochs):
    loss_epoch = 0
    for image, label in trainLoader:
        image = image.reshape(-1, size).to(device)
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss_epoch = loss
        loss.backward()
        #print(loss)
        optimizer.step()
        
        _, prediction = torch.max(output,1)
        accuracy = (label == prediction.squeeze()).float().mean()
        #print("Accuracy: " + str(accuracy))
        overall_step+=1
        if plot:
          info = { ('loss_' + model_name): loss.item(), ('accuracy_' + model_name): accuracy.item() }
          for tag, value in info.items():
            logger.scalar_summary(tag, value, overall_step+1)

train(model, optimizer, criterion, train_data_loader, validation_loader, epochs, 196608 , 'ff_2', True)

model

"""### Evaluation of the model"""

from torch.autograd import Variable
'''
Convert to onehot encoded vector from a single integer

'''

def to_one_hot(y, n_dims=None):
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

##Compute AUC for each of the classes
def computeAUC (data, predicted, classCount):
    auroc = []
    data_np = data.cpu().numpy()
    data_np_pred = predicted.cpu().numpy()
    for i in range(classCount):
        auroc.append(roc_auc_score(data_np[:, i], data_np_pred[:, i]))
    return auroc

##Compute the test accuracy and the AUC values
def test(model, data_loader, class_count, class_names):   
    out = torch.FloatTensor().cuda()
    out_pred = torch.FloatTensor().cuda()
    model.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image = image.reshape(-1, input_size).to(device)
            target = to_one_hot(label.cuda(), 4).to(device)
            out = torch.cat((out, target), 0).cuda()
            out = model(image)
            out_pred = torch.cat((out_pred, out), 0)
        aurocClass = computeAUC(out, out_pred, 4)
        aurocMean = np.array(aurocClass).mean()
        print ('AUC: ', aurocMean)
        for i in range (0, len(aurocClass)):
            print (class_names[i], ' ', aurocClass[i])
    return out, out_pred

class_names = ["Pleural_Effusion", "Edema","Cardiomegaly","Pneumonia"]
outGT1, outPRED1 = test(model, validation_loader, 4, class_names)

for i in range(4):
    fpr, tpr, threshold = metrics.roc_curve(out.cpu()[:,i], out_pred.cpu()[:,i])
    roc_auc = metrics.auc(fpr, tpr)
    f = plt.subplot(2, 7, i+1)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 20
    plt.rcParams["figure.figsize"] = fig_size
    plt.title('ROC for: ' + class_names[i])
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.show()

