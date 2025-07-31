import numpy as np
import matplotlib.pyplot as plt
import torch
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
from sklearn.model_selection import train_test_split
from model import *
from loss import *
from dataload import LoadData, datapre
import skimage
from skimage import morphology
from skimage import io
from config import *

# 检查是否有可用的GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

indir=r'.\instance-segmentation-building-dataset-of-china\building/'

X_train=sorted(glob.glob(indir+"/train/Images/sh*.tif"))
y_train=sorted(glob.glob(indir+"/train/PNG/sh*.png"))
# X_val=sorted(glob.glob(indir+"/test/Images/*.tif"))
# y_val=sorted(glob.glob(indir+"/test/PNG/*.png"))

X_train, X, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3,random_state=41)
batch_size = 32
checkpoint_path =outptfile
outdir = basedir+'/predict'
os.makedirs(outdir,exist_ok=True)

num = 80
ratio = 0.2

# 加载模型
model = SegmentationModel()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# 处理和保存图像
for file in tqdm(X):
#     if 'data_5608' not  in file:continue
    img = datapre(file).getite()
    logits_mask = model(img.to(device, dtype=torch.float32).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > ratio) * 1.0
    pre = pred_mask.detach().cpu().numpy().squeeze(0).squeeze(0).astype(np.uint8)
    pre[pre == 1] = 255
    # 将原始图像从PIL格式转换为numpy数组
    original_img = io.imread(file)
    original_img = np.array(original_img)

#     plt.imshow(original_img[:,:,-1])
#     plt.show()

#     plt.imshow(predictimages[99][-1])
    # 保存预测结果图像
    outfile = os.path.join(outdir, os.path.basename(file))
    img1 = Image.fromarray(pre)
#     img1.save(outfile)

    # 创建带有两个子图的图像
    plt.figure(figsize=(12, 6))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_img,'jet')

    # 显示预测结果
    plt.subplot(1, 2, 2)
    plt.title('Prediction Mask')
    plt.imshow(pre, cmap='gray')

    # 保存合并后的图像
    combined_outfile = os.path.join(outdir, 'combined_' + os.path.basename(file))
    plt.savefig(combined_outfile)
    
    plt.close()


