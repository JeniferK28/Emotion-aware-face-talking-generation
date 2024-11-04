import torch
import os
import ffmpeg
import cv2
import argparse
import numpy as np
import shutil
import face_alignment
import utils
from audio2lmrk.gcn_model_audio import a2lNet
from audio2lmrk.gcn_model_audio import a2lNet_pretrain
from audio2lmrk.model_emotion import emo2lNet
import pandas as pd
from scipy import sparse
import glob
from scipy.signal import savgol_filter
from lmrk2img.land2img import  Image_translation_block
from lmrk2img.main_land2img import parse_args as img_config

seed1 = 5930
np.random.seed(seed1)
os.environ["PYTHONHASHSEED"] = str(seed1)
torch.cuda.manual_seed(seed1)
torch.cuda.manual_seed_all(seed1)     # if you are using multi-GPU
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
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--cuda",  default=True)
    parser.add_argument("--train_dir", type=str, default="/mnt/40E42154E4214D8A/pross_data/audio_lvl3")
    parser.add_argument("--val_dir", type=str, default="/mnt/40E42154E4214D8A/pross_data/Test/audio_lvl3")
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
    parser.add_argument("--name", type=str, default="gcn_mouth_dis_2d_50_2branch")
    parser.add_argument("--audio_pretrain", type=str, default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/models/audio2lmrk_neutral_pretrain_time_MSE_disc_1st_frame_elu_ct_99.pt")
    #parser.add_argument("--a2l_pretrain", type=str,default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/models/audio2lmrk_emo_time_bothMSE_elu_nodisc_ct_emoclass_80.pt")
    parser.add_argument("--a2l_pretrain", type=str, default="/mnt/40E42154E4214D8A/audio_test/audio2lmrk/models/audio2lmrk_time_direct_emo_MSE_5_99.pt")
    #time_bothMSE_attention, time_bothMSE_attention_linear, time_bothMSE_direct_linear
    parser.add_argument("--emo", type=str, default="angry")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audio_file", type=str, default="/media/jen/T7 Touch/Audio/test/M003/neutral/level_1/005.m4a")
    parser.add_argument("--img_ref_path", type=str, default="/mnt/40E42154E4214D8A/img_ref/MEAD_test/img")
    return parser.parse_args()


#M005. W019. W033

config = parse_args()
audio_id = config.audio_file.split('/')[-1].split('.')[0]
sub_id = config.audio_file.split('/')[6] + '_' + config.audio_file.split('/')[7] + '_' + config.audio_file.split('/')[8] + '_'+ config.audio_file.split('/')[9].split('.')[0]
shutil.copy(config.audio_file, '/results/{}.wav'.format(audio_id))

config.img_ref_file =  os.path.join(config.img_ref_path, sub_id.split('_')[0] + '.jpg')

config.audio = audio_id + '.wav'
config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from skimage import io, img_as_float32

''' Preprocess input reference image '''
img = cv2.imread(config.img_ref_file)

predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
shapes = predictor.get_landmarks(img)
#img = img_as_float32(io.imread(config.jpg))
if (not shapes or len(shapes) != 1):
    print('Cannot detect face landmarks. Exit.')
    exit(-1)

shape_3d = shapes[0]
shape_3d = utils.close_input_face_mouth(shape_3d)


''' Normalize face '''
#shape_3d, scale, shift = utils.norm_input_face(shape_3d)


shutil.copyfile('results/{}'.format(config.audio), 'results/tmp.wav')

# speaking embedding
spk_emb, a = utils.get_spk_emb('results/{}'.format(config.audio))


print('Processing audio file', audio)
#m = utils.audio_preprocessing(audio)
m = utils.wav2data(audio,config.device)

# if (os.path.isfile('results/tmp.wav')):
#     os.remove('results/tmp.wav')


df = pd.read_excel('/mnt/40E42154E4214D8A/audio_test/landmark_matrix.xlsx')
adj = pd.DataFrame(df).to_numpy()
vertices = sparse.csc_matrix(adj)
pos = torch.arange(1, 69)
s = torch.zeros([68, 8])
s[0:17, 0] = 1;s[17:22, 1] = 1;s[22:27, 2] = 1;s[27:36, 3] = 1;s[36:42, 4] = 1;s[42:48, 5] = 1
s[48:60, 6] = 1;s[60:68, 7] = 1


audio2lmrk = a2lNet(config).to(config.device)
#audio2lmrk = a2lNet_pretrain(config).to(config.device)
audio2lmrk.load_state_dict(torch.load(config.a2l_pretrain))

#emo = audio.split('/')[6]
emo = config.emo


fls = torch.zeros([len(m), 68,2])
c=0
for mfcc in m:
    #elements to device
    data, audio, ref_lmrks = utils.a2l_data(shape_3d, mfcc, emo, spk_emb, adj, vertices, s, pos,  config.device)
    
    #recunstruct landmarks per audio chunck
    lmrks_reconstructed, ct_emb, lmrk_emb = audio2lmrk(data, audio, ref_lmrks, config.device, train = 'False')
    lmrks_reconstructed = torch.squeeze(lmrks_reconstructed).cpu().detach().numpy()

    fls[c,:,:] = torch.Tensor(lmrks_reconstructed)
    c = c + 1


fls = fls.reshape((-1, 136)).cpu().detach().numpy()
fls[:, :48 * 2] = savgol_filter(fls[:, :48 * 2], 15, 3, axis=0)
fls[:, 48 * 2:] = savgol_filter(fls[:, 48 * 2:], 5, 3, axis=0)
fls = fls.reshape((-1, 68, 2))

''' lmrk2Img '''
model = Image_translation_block(img_config(), single_test=True)
with torch.no_grad():
    model.single_test(jpg=img, fls=fls, prefix=sub_id)
    print('Gen')





























