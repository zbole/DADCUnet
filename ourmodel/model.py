
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp



from segmentation_models_pytorch.losses import DiceLoss
from config import *
class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, images):
        x = self.conv(images)
        p = self.pool(x)

        return x, p

# %%
class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = conv(out_channels * 2, out_channels)

    def forward(self, images, prev):
        x = self.upconv(images)
        x = torch.cat([x, prev], axis=1)
        x = self.conv(x)

        return x

# %% [markdown]
# Burada kafa karıştıran bölüm fonksiyonlar arasında bağlantı olmamasına rağmen fonksiyonların bağlı olması olabilir. Bunu sağlayanın class'ın başlangıcında yazdığımız nn.Module'dür. 
# 
# nn.Module forward fonksiyonunu __init__ ile bağlayıp bir mimarı oluşturuyor...

# %%
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder(3, 64)
        self.e2 = encoder(64, 128)
        self.e3 = encoder(128, 256)
        self.e4 = encoder(256, 512)

        self.b = conv(512, 1024)

        self.d1 = decoder(1024, 512)
        self.d2 = decoder(512, 256)
        self.d3 = decoder(256, 128)
        self.d4 = decoder(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, images):
        x1, p1 = self.e1(images)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)

        b = self.b(p4)
        
        d1 = self.d1(b, x4)
        d2 = self.d2(d1, x3)
        d3 = self.d3(d2, x2)
        d4 = self.d4(d3, x1)

        output_mask = self.output(d4)

        return output_mask  
    


from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


from networks_2.bra_unet import BRAUnet

class SegmentationModel(nn.Module):  
  
    # 初始化方法，当创建这个类的一个实例时，这个方法会被调用。  
    def __init__(self):  
        # 调用父类nn.Module的初始化方法，是PyTorch的标准做法。  
        super(SegmentationModel,self).__init__()  
  
        # 定义模型的主要架构，这里使用了segmentation_models_pytorch库的Unet结构。  
        # ENCODER和WEIGHTS应该是在类外部定义的变量，它们分别代表所使用的预训练编码器的名称和权重。  




        
        
        config_vit='R50-ViT-B_16'
        self.arc = ViT_seg(CONFIGS_ViT_seg[config_vit], img_size=224, num_classes=1)
        # self.arc.load()

        
        #EOF

        

        
 
        

    # 定义前向传播方法，当模型接收到输入数据时，这个方法会被调用。  
    def forward(self, images, masks=None):  
        # 将输入图像通过模型的主要架构（即Unet）进行前向传播，得到logits。  
        logits = self.arc(images)  

        # 如果提供了masks（即标签数据），则计算两种损失。  
        if masks != None:  
            # masks=masks.squeeze(1)
            # 使用Dice损失函数计算logits和masks之间的损失，模式为'binary'。  
            loss1 = DiceLoss(mode='binary')(logits, masks)  
            # 使用二元交叉熵损失函数（带logits）计算logits和masks之间的损失。  
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)  
            
            
            # 返回logits和两个损失值。  
            return logits, loss1, loss2  

        # 如果没有提供masks，则只返回logits。  
        return logits
    
    