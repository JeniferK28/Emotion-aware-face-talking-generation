import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import argparse
from land2img import Image_translation_block
import platform
import torch
from torch.utils.data import DataLoader
from data_loader import lmrk_img_dataset

seed = 5930
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_frames', type=int, default=1, help='')
parser.add_argument('--num_workers', type=int, default=4, help='number of frames extracted from each video')
parser.add_argument('--lr', type=float, default=0.0001, help='')

parser.add_argument('--write', default=True, action='store_true')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--name', type=str, default='all')
parser.add_argument('--test_speed', default=False, action='store_true')

parser.add_argument('--jpg_dir', type=str, default=jpg_dir)
parser.add_argument('--ckpt_dir', type=str, default=ckpt_dir)
parser.add_argument('--log_dir', type=str, default=log_dir)

parser.add_argument('--jpg_freq', type=int, default=50, help='')
parser.add_argument('--ckpt_last_freq', type=int, default=1000, help='')
parser.add_argument('--ckpt_epoch_freq', type=int, default=1, help='')

parser.add_argument('--load_G_name', type=str, default='')
parser.add_argument('--use_vox_dataset', type=str, default='raw')

parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--fan_2or3D', type=str, default='3D')

parser.add_argument('--single_test', type=str, default='')

parser.add_argument("--lmrks_train", type=str, default="/mnt/40E42154E4214D8A/pross_data/lmrks")
parser.add_argument("--lmarks_val", type=str, default="/mnt/40E42154E4214D8A/pross_data/Test/lmrks")

parser.add_argument("--train_ref_img", type=str, default="/mnt/40E42154E4214D8A/img_ref/MEAD_train")
parser.add_argument("--train_img_path", type=str, default="/mnt/40E42154E4214D8A/pross_data/Images")

parser.add_argument("--val_ref_img", type=str, default="/mnt/40E42154E4214D8A/img_ref/MEAD_test")
parser.add_argument("--val_img_path", type=str, default="/mnt/40E42154E4214D8A/pross_data/Test/Images")

config = parser.parse_args()

jpg_dir =  'tmp_v'
ckpt_dir = 'ckpt'
log_dir = 'log'

#### Data preparation 

from files_random import select_files

#lmrk_train = select_files('train')
#lmrk_val = select_files('val')

for lmrk in os.listdir(config.lmrk_train):
    lmrk_train.append(os.path.join(config.lmrk_train, lmrk))

for lmrk in os.listdir(config.lmrk_val):
    lmrk_val.append(os.path.join(config.lmrk_val, lmrk))


train_set = lmrk_img_dataset(lmrk_train, config.train_ref_img, config.train_img_path)
val_set = lmrk_img_dataset(lmrk_val, config.val_ref_img, config.val_img_path)

# train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
train_loader = DataLoader(train_set, batch_size=config.batch_size,
                          shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size,
                        shuffle=True, drop_last=True)


### Train the network 
model = Image_translation_block(config)

for epoch in range(config.nepoch):
    model.train(train_loader, epoch)
    with torch.no_grad():
        model.test(val_loader, epoch)
