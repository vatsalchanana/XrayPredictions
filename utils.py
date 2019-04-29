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



def getLabelDf(x,df, labelMap=None, disease=None):
    
    x = x[36:]          #To account for the extra "././" added before the Path variable
    
    if disease == None:
        x = df.loc[df.Path == x] 
        return labelMap[x.label.values[0]]
    else:
        x = df.loc[df.Path == x] 
        if x.label.values[0]==disease:
            return 1
        else:
            return 0
    
    

def getLabel(x,disease):
    
    if x[disease] ==1:
        return disease
    else:
        return "Rest"


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





def createPlots(model, path, modelID):
    
    ### TRAIN LOSS
    plt.plot(model.recorder.losses,'r-')
    plt.ylabel("Train Loss")
    plt.xlabel("Iterations")
    plt.savefig(os.path.join(path,"train_loss_"+str(modelID)+".png"))
    plt.clf()
    plt.close()
    
    ### VAL LOSS
    plt.plot(model.recorder.val_losses,'r-')
    plt.ylabel("Validation Loss")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(path,"valid_loss_"+str(modelID)+".png"))
    plt.clf()
    plt.close()
    
    ### VAL ACCURACY
    plt.plot(model.recorder.metrics,'r-')
    plt.ylabel("Validation Acc")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(path,"valid_acc_"+str(modelID)+".png"))
    plt.clf()
    plt.close()



def create1vAllData(df, disease):
    
    t = df[df[disease] == 1.0]
    f = df[df[disease] == 0.0]
    
    num_t = len(t)
    
    f = f.sample(n= num_t, replace = False)
    
    out = pd.concat([t,f])
    
    out = out.sample(frac = 1.0, replace = False)
    
    return out




def cleanLabelFile(df, disCols,sumCount = True):
    
    for i in range(len(disCols)):

        #Changing NaN labels
        df.loc[pd.isna(df[disCols[i]]) , disCols[i]] = 0.0

        #Changing unsure labels
        df.loc[ df[disCols[i]]==-1.0 , disCols[i]] = 1.0
        
        #Removing samples with no labels at all
        if sumCount == True:
            df['sumCount'] = df.apply(lambda x : getSum(x,disCols), axis=1)
            
            df = df[df.sumCount!=0]
            
            df.drop("sumCount",inplace =True, axis =1 )
        

    return df


def getSum(x,diseaseList):
    
    y = x[diseaseList]
    return np.sum(y)




def freezeBody(model, unfreeze = False):
    
    if unfreeze==False:
        for param in model.densenet.features.parameters():
            param.requires_grad = False
    else:
        for param in model.densenet.features.parameters():
            param.requires_grad = True
            
            
def freezeUpto(model, paramCount=10):
    
    paramCount = min(paramCount, 362)
    
    for param in model.densenet.features.parameters():
        param.requires_grad = False