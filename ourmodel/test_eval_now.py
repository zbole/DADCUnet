import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,classification_report
import seaborn as sns
import pandas as pd
from model import *
from loss import *
# from evaluate import *
from util import *


indir=r'.\instance-segmentation-building-dataset-of-china\building/'

X_train=sorted(glob.glob(indir+"/train/Images/wh*.tif"))
y_train=sorted(glob.glob(indir+"/train/PNG/wh*.png"))
# X_val=sorted(glob.glob(indir+"/test/Images/*.tif"))
# y_val=sorted(glob.glob(indir+"/test/PNG/*.png"))

X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3,random_state=42)

# Create datasets and dataloaders
train_dataset = LoadData(X_train, y_train)
valid_dataset = LoadData(X_val, y_val,mode='val')

batch_size = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = 'cuda'
model = SegmentationModel()
model = model.to(device)
loss_fn = DiceBCELoss()
# Load model weights
model.load_state_dict(torch.load(outptfile))

def compute_miou(preds, labels, num_classes):
    """计算每个类别的 IoU，并计算 mIoU"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(np.float32)
        label_cls = (labels == cls).astype(np.float32)
        intersection = np.sum(pred_cls * label_cls)
        union = np.sum(pred_cls + label_cls) - intersection
        if union == 0:
            iou = float('nan')  # 当没有样本时 IoU 为 NaN
        else:
            iou = intersection / union
        ious.append(iou)
    return np.nanmean(ious)  # 返回所有类别 IoU 的平均值

def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred_binary = (y_pred > 0.5).float()
            all_preds.append(y_pred_binary.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        epoch_loss = epoch_loss / len(loader)
        
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    # 计算每个类别的混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    print(classification_report(all_labels, all_preds))
    
    # 计算每个类别的 IoU
    miou = compute_miou(all_preds, all_labels, num_classes)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, acc, f1, miou, conf_matrix

def save_results_to_csv(acc, f1, miou, conf_matrix, output_dir):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存评估结果到 CSV 文件
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'mIoU'],
        'Value': [acc, f1, miou]
    })
    
    results_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    # 保存混淆矩阵到 CSV 文件
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Class {i}' for i in range(conf_matrix.shape[0])],
                                  columns=[f'Class {i}' for i in range(conf_matrix.shape[1])])
    
    conf_matrix_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    conf_matrix_df.to_csv(conf_matrix_csv_path)
    confusion_matrix_decimal = conf_matrix / conf_matrix.sum(axis=0)
    # 保存混淆矩阵图像
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix_decimal, annot=True, fmt='.2f', cmap='Blues', xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    
    
    print(f'Evaluation results saved to {results_csv_path}')
    print(f'Confusion matrix saved to {conf_matrix_csv_path}')

# 设置类别数量
num_classes = 2  # 根据实际情况调整类别数量

# 设定统一的输出文件夹路径
output_dir = os.path.join(basedir, 'evaluation_results2')
valid_loss, acc, f1, miou, conf_matrix = evaluate(model,valid_loader, loss_fn, device, num_classes)

print(f"valid_loss: {valid_loss:.4f}")  
print(f"acc: {acc:.2f}")  
print(f"f1: {f1:.2f}")  
print(f"miou: {miou:.2f}")  
# 打印混淆矩阵  

print("conf_matrix:")  
print(conf_matrix)

save_results_to_csv(acc, f1, miou, conf_matrix, output_dir)

