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

seed1 = 5930
np.random.seed(seed1)
os.environ["PYTHONHASHSEED"] = str(seed1)
torch.cuda.manual_seed(seed1)
torch.cuda.manual_seed_all(seed1) # if you are using multi-GPU
torch.manual_seed(seed1)

root = r'//mnt/40E42154E4214D8A/audio_test/lmrk2img'

jpg_dir = os.path.join(root, 'tmp_v')
ckpt_dir = os.path.join(root, 'ckpt')
log_dir = os.path.join(root, 'log')


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

parser.add_argument("--train_dir", type=str, default="/mnt/40E42154E4214D8A/pross_data/lmrks")
parser.add_argument("--val_dir", type=str, default="/mnt/40E42154E4214D8A/pross_data/Test/lmrks")



config = parser.parse_args()

''' Step 1. Data preparation '''

from files_random import select_files
lmrk_train = select_files('train')
lmrk_val = select_files('val')
# for lmrk in os.listdir(config.train_dir):
#     lmrk_train.append(os.path.join(config.train_dir, lmrk))

# for lmrk in os.listdir(config.val_dir):
#     lmrk_val.append(os.path.join(config.val_dir, lmrk))

train_ref_img = '/mnt/40E42154E4214D8A/img_ref/MEAD_train'
img_path = '/mnt/40E42154E4214D8A/pross_data/Images'
train_set = lmrk_img_dataset(lmrk_train, train_ref_img, img_path)

val_ref_img = '/mnt/40E42154E4214D8A/img_ref/MEAD_test'
img_path = '/mnt/40E42154E4214D8A/pross_data/Test/Images'
val_set = lmrk_img_dataset(lmrk_val, val_ref_img, img_path)

# train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
train_loader = DataLoader(train_set, batch_size=config.batch_size,
                          shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size,
                        shuffle=True, drop_last=True)


''' Step 2. Train the network '''
model = Image_translation_block(config)

for epoch in range(config.nepoch):
    model.train(train_loader, epoch)
    with torch.no_grad():
        model.test(val_loader, epoch)