
# %%
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


print(torch.cuda.is_available())
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from torch import nn
from model import *
from config import *
from dataload import *
from util import *
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler

import shutil
import pandas as pd

outptfile1=outptfile


indir=r'.\instance-segmentation-building-dataset-of-china\building/'

X_train=sorted(glob.glob(indir+"/train/Images/wh*.tif"))
y_train=sorted(glob.glob(indir+"/train/PNG/wh*.png"))
# X_val=sorted(glob.glob(indir+"/test/Images/*.tif"))
# y_val=sorted(glob.glob(indir+"/test/PNG/*.png"))

X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3, random_state=42)


train_dataset = LoadData(X_train, y_train)
valid_dataset = LoadData(X_val, y_val,mode='val')


train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
)

valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
)


model=SegmentationModel()
if loadstate:
    model.load_state_dict(torch.load(basedir+rf'\{loadstateptfile}'))
model.to(DEVICE)

optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer=torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,eta_min=1e-5)

outloss={}
trainloss=outloss.setdefault('trainloss',[])
valloss=outloss.setdefault('valloss',[])

best_val_dice_loss=np.inf
# best_val_bce_loss=np.Inf
best_val_loss=np.inf
for i in range(EPOCHS):
    outfile=basedir+rf"jpgoutnew/{str(i)}.jpg"
    # train_loss=1
    # valid_loss=1
    os.makedirs(os.path.split(outfile)[0],exist_ok=True)
    train_loss = train_fn(train_loader,model,optimizer)
    valid_loss = eval_fn(valid_loader,model,outfile)
    scheduler.step()
    # train_dice,train_bce=train_loss
    # valid_dice,valid_bce=valid_loss
    # print(f'Epochs:{i+1}\nTrain_loss --> Dice: {train_dice} BCE: {train_bce} \nValid_loss --> Dice: {valid_dice} BCE: {valid_bce}')
    trainloss.append(train_loss)
    valloss.append(valid_loss)
    
    print(f'Epochs:{i+1}\nTrain_loss --> {train_loss} \nValid_loss --> { valid_loss:} ')
    
    if valid_loss< best_val_loss:
        torch.save(model.state_dict(),outptfile1)
        print('Model Saved')
        # best_val_dice_loss=valid_dice
        best_val_loss= valid_loss

outcsv=pd.DataFrame(outloss)
outcsv.to_csv(outptfile.replace('.pt','.csv'))


