
import os
DEVICE='cuda'
EPOCHS=200
BATCH_SIZE=12
LR=0.001
ratio=0.5 #Various ratios could perform better for visualization
sample_num=3
MAXMIN=False
height,width = (500, 500)

# ENCODER='resnet50'

ENCODER='efficientnet-b4'
WEIGHTS='imagenet'

# outptfile='lab3labelimage_efficientnet.pt'


name='ViT_seg_500_final_wh_2822_no_short_x2in_v1'
basedir=rf'./{name}/'
os.makedirs(basedir,exist_ok=True)
outptfile=basedir+f'{ENCODER}_{WEIGHTS}_{name}.pt'


loadstate=False
loadstateptfile=outptfile
def log(traintxt,ds):
    with open(traintxt,'a') as  f:
        f.write(ds)



import glob
import shutil



for file in glob.glob('./*.py'):
    os.makedirs(basedir+'code',exist_ok=True) 
    shutil.copyfile(file,basedir+'/'+'code/'+os.path.basename(file))
    
    