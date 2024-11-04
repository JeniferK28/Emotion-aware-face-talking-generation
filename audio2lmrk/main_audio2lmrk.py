import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import argparse
from audio2lmrk import train
import platform
import torch
from torch_geometric.data import  DataLoader
from a2l_dataloader import audio_lmrk_vertice, audio_lmrk_graph
from gcn_model_audio import a2lNet, a2lNet_pretrain
from files_random import file_select
from scipy import sparse
import pandas as pd
import clip


seed1 = 5930
np.random.seed(seed1)
os.environ["PYTHONHASHSEED"] = str(seed1)
torch.cuda.manual_seed(seed1)
torch.cuda.manual_seed_all(seed1) # if you are using multi-GPU
torch.manual_seed(seed1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda1", type=int, default=100)

    parser.add_argument("--alfa", type=float, default=0.5)
    parser.add_argument("--beta",  type=float, default=0.5)
    parser.add_argument("--lamb",  type=int, default=0.02)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--cuda",  default=True)
    parser.add_argument("--train_dir", type=str, default="/media/jen/6dc9b3fb-1ae6-456d-8bc3-a056396531bf/pre-trained_data/emo/train/cnt_emb_5")
    parser.add_argument("--val_dir", type=str, default="/media/jen/6dc9b3fb-1ae6-456d-8bc3-a056396531bf/pre-trained_data/emo/test/cnt_emb_5")
    parser.add_argument('--device_ids', type=str, default=['0'])
    parser.add_argument("--model_dir", type=str, default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/models/")
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--scales_train', type=int, default=[1, 0.5, 0.25, 0.125])
    parser.add_argument('--epoch_milestones', default = [60 ,90])
    parser.add_argument('--triplet_margin', type=int, default=1)
    parser.add_argument('--triplet_weight', type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/log/")
    parser.add_argument("--lmrk_dir", type=str, default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/samples/")
    parser.add_argument("--name", type=str, default='time_direct_emo_MSE_5')
    parser.add_argument("--device", type=str,
                        default="cuda")
    parser.add_argument("--audio_pretrain", type=str,
                        default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/models/audio2lmrk_neutral_pretrain_time_MSE_disc_1st_frame_elu_ct_99.pt")
    parser.add_argument("--train_spk", type=str,default="/mnt/40E42154E4214D8A/spk_emb/train")
    parser.add_argument("--train_path_lmrk", type=str,default="/mnt/40E42154E4214D8A/pross_data/lmrks")
    parser.add_argument("--train_ref_lmrk", type=str,default="/mnt/40E42154E4214D8A/img_ref/MEAD_train/lmrks/close_lips")
    
    parser.add_argument("--val_spk", type=str,default="/mnt/40E42154E4214D8A/spk_emb/test")
    parser.add_argument("--val_path_lmrk", type=str,default="/mnt/40E42154E4214D8A/pross_data/Test/lmrks")
    parser.add_argument("--val_ref_lmrk", type=str,default="/mnt/40E42154E4214D8A/img_ref/MEAD_test/lmrks/close_lips")
    
    

    return parser.parse_args()

if __name__=="__main__":

    config = parse_args()
    print(config.name)
    print('Load data begin....')
    config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_files = ''
    val_files = ''


    lmrk_train = file_select(config.train_dir, 'train')
    lmrk_val = file_select(config.val_dir, 'val')

    df = pd.read_excel('/mnt/40E42154E4214D8A/audio_test/landmark_matrix.xlsx')
    adj = pd.DataFrame(df).to_numpy()
    vertices = sparse.csc_matrix(adj)
    pos = torch.arange(1, 69)
    s = torch.zeros([68,8])
    s[0:17,0]= 1; s[17:22, 1] = 1; s[22:27, 2] = 1; s[27:36, 3] = 1; s[36:42, 4] = 1; s[42:48, 5] = 1
    s[48:60, 6] = 1; s[60:68, 7] = 1


    train_set = audio_lmrk_graph(lmrk_train, config.train_ref_lmrk, config.train_spk, config.train_dir, config.train_path_lmrk, vertices, adj, pos,s, config.device)
    val_set = audio_lmrk_graph(lmrk_val, config.val_ref_lmrk, config.val_spk, config.val_dir, config.val_path_lmrk,  vertices, adj, pos, s, config.device)


    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=True, drop_last=True)


    model = a2lNet(config).to(config.device)
    train(config, train_loader, val_loader, model)
