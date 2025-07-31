# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
import torch
import torch.nn as nn
from networks_2.bra_unet_system import BRAUnetSystem
logger = logging.getLogger(__name__)

class up(nn.Module):
    def __init__(self, in_ch, mode='bilinear', size=False):
        super(up, self).__init__()
        if mode == 'bilinear':
            if not size:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                 self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)               
        elif mode == 'pixelshuffle':
            self.up = nn.Sequential(nn.Conv2d(in_ch, 4*in_ch, kernel_size=3, padding=1, dilation=1),
              			nn.PixelShuffle(2))
        elif mode == 'ConvTrans':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # elif mode == 'CARA':
        #     self.up = CARAFE(in_ch)

    def forward(self, x):
        x = self.up(x)
        return x
class BRAUnet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1, n_win=8):
        super(BRAUnet, self).__init__()
        self.up2 = up(in_chans, size=(224, 224))#, 'CARA')
        
        self.up3 = up(in_chans, size=(224, 224))#, 'CARA')
        self.bra_unet = BRAUnetSystem(img_size=img_size,
                                      in_chans=in_chans,
                                      num_classes=num_classes,
                                      head_dim=32,
                                      n_win=n_win,
                                      embed_dim=[96, 192, 384, 768],
                                      depth=[2, 2, 8, 2],
                                      depths_decoder=[2, 8, 2, 2],
                                      mlp_ratios=[3, 3, 3, 3],
                                      drop_path_rate=0.2,
                                      topks=[2, 4, 8, -2],
                                      qk_dims=[96, 192, 384, 768])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # x=self.up2(x)
        logits = self.bra_unet(x)
        
        
        # logits=self.up3(logits)
        
        return logits
    def load_from(self):
        pretrained_path = r'E:\Projects\export_海冰分割\modelweight_data_pretrainmodel\biformer_base_best.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device,weights_only=False)
            model_dict = self.bra_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.bra_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")
