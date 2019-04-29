import pandas as pd
import numpy as np

import os
from PIL import Image

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
import torchvision.models as tmodels


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
    
    

def getLabel2(x,disease):
    
    if x[disease] ==1:
        return disease
    else:
        return "Rest"



# def getLabel(x):
    
#     if x.Pleural_Effusion ==1:
#         return "Pleural_Effusion"
#     else:
#         return "Rest"


#     # elif x.Edema == 1:
#     #     return "Edema"
#     # elif x.Cardiomegaly==1:
#     #     return "Cardiomegaly"
#     # elif x.Pneumonia == 1:
#     #     return "Pneumonia"
#     # else:
#     #     return "0"
    

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


trainFile = pd.read_csv(os.path.join(baseFolder,'train.csv'), names = cols, header=0)
validFile = pd.read_csv(os.path.join(baseFolder,'valid.csv'), names = cols, header=0)

trainFile["Path"] = trainFile.Path.apply(lambda x : x.replace('CheXpert-v1.0-small',"")[1:])
validFile["Path"] = validFile.Path.apply(lambda x : x.replace('CheXpert-v1.0-small',"")[1:])


selectCols = ['Path',"View",'Sex',"Pleural_Effusion", "Edema","Cardiomegaly","Pneumonia"]

trainFile = trainFile[selectCols]
validFile = validFile[selectCols]

# -1 for Uncertain, 0 for negative, 1 for positive

trainFile['isClean'] = trainFile.apply(lambda x : cleanLabel(x), axis = 1)
validFile['isClean'] = validFile.apply(lambda x : cleanLabel(x), axis = 1)

trainFile['train'] = False
validFile['train'] = True

trainFile = trainFile[trainFile.isClean==1]
validFile = validFile[validFile.isClean==1]


df = pd.concat([trainFile,validFile])

df['label'] = df.apply(lambda x : getLabel(x), axis = 1)

labelMap = {"Pleural_Effusion":0, "Edema":1,"Cardiomegaly":2,"Pneumonia":3,"Rest":4}

def getLabelDf(x):
    
#     print(x)
    
    x = x[36:]          #To account for the extra "././" added before the Path variable
#     print(x)
    x = df.loc[df.Path == x] 
#     print(x)
    
    return labelMap[x.label.values[0]]


class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, path:str='/home/santhosr/Documents/Courses/CIS700/Project/models',id:int=None,monitor:str='val_loss', mode:str='auto',modelName:str='resnet50'):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.bestAcc = 0.0001
        self.folderPath = path
        self.id = id
        self.modelName = modelName
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."

        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            if self.id==None:
                fileName = 'model_'+self.modelName+'_acc'+str(int(acc*1000))+"_loss"+str(int(val_loss*1000))
            else:
                fileName = 'model_'+self.modelName+'_id' + str(self.id) + '_acc' + str(int(acc*1000)) + "_loss" + str(int(val_loss*1000))
            fileName = os.path.join(self.folderPath, fileName)
            self.learn.save(fileName)


print("Data Creation Start")
data = ImageItemList.from_df(df=df,path=baseFolder, cols='Path').split_from_df(col='train').label_from_func(getLabelDf).transform(get_transforms(),size=256).databunch(bs=50).normalize()
print("Data Creation Complete")


learn = create_cnn(data, tmodels.resnet50, metrics=accuracy,pretrained=True)

# learn.load('/home/santhosr/Documents/Birad/ProcessedData/models/model_resnet50_acc668_loss600')

best_model_cb = partial(ModelTrackerCallback,id=6, modelName = "resnet50_Edema")
learn.callback_fns.append(best_model_cb)

learn.unfreeze()
learn.fit(30,1e-5)
