import torch
from torch.utils.data import Dataset
import os
from skimage import io, img_as_float32, transform
import random
import numpy as np
from torchvision import transforms
import cv2


emo_label = {'angry': [1,0,0,0,0,0,0,0], 'contempt':[0,1,0,0,0,0,0,0], 'disgusted':[0,0,1,0,0,0,0,0], 'fear':[0,0,0,1,0,0,0,0], 'happy':[0,0,0,0,1,0,0,0],
             'neutral':[0,0,0,0,0,1,0,0],'sad':[0,0,0,0,0,0,1,0], 'surprised':[0,0,0,0,0,0,0,1]}

id_label = {'M005':0,'M007':1,'M009':2,'M011':3,'M012':4,'M019':5,'M023':6,'M024':7,'M025':8,'M026':9,'M027':10,'M028':11,'M029':12,'M030':13,'M031':14,
            'M033':15,'M034':16,'M035':17,'M036':18,'M037':19,'M039':20,'M040':21,'M042':22,'W009':23,'W011':24,'W015':26,'W016':27,
            'W018':28,'W021':29,'W023':30,'W024':31,'W025':32,'W026':33,'W029':34,'W035':35,'W037':36,'W038':37,'W040':38}


class Speaker_dataset(Dataset):
    def __init__(self, dataset_dir, train, path):
        # data path
        self.all_data = dataset_dir
        self.train = train
        self.path = path
        self.utter_num = 128

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        emotion = self.all_data[idx].split('/')[-1].split('_')[1]
        label_emo = torch.Tensor([emo_label[emotion]])

        c = self.all_data[idx].split('/')[-1].split('_')
        file = c[0]+ '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + c[4] + '.npy'
        spk_emb = np.load(os.path.join(self.path,file))
        spk_emb = torch.FloatTensor(spk_emb)
        mfcc_path = os.path.join(self.all_data[idx])
        mfcc = np.load(mfcc_path)
        mfcc = mfcc[:, 1:]
        mfcc = torch.FloatTensor(mfcc)
        data = torch.unsqueeze(mfcc, 0)

        if self.train == True:
            audio_id = c[0]
            label_id = torch.Tensor([id_label[audio_id]])
            return data, spk_emb, label_emo, label_id
        else:
            label_id = c[0]
            return data, spk_emb, label_emo, label_id

class MEAD_dataset(Dataset):
    def __init__(self, dataset_dir, path_lmrk, path_spk, path_img, path_ref_img, path_ref_lmrk,train):
        # data path
        self.all_data = dataset_dir
        self.train = train
        self.path_lmrk = path_lmrk
        self.path_spk = path_spk
        self.path_img = path_img
        self.path_ref_img = path_ref_img
        self.path_ref_lmrk = path_ref_lmrk
        self.utter_num = 128

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        emotion = self.all_data[idx].split('/')[-1].split('_')[1]
        c = self.all_data[idx].split('/')[-1].split('_')

        audio_file = c[0]+ '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + c[4] + '.npy'
        if int(c[4])<10:
            n = '00' + c[4]
        else : n = '0' + c[4]
        img_file = c[0] + '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + n + '_' + c[5].split('.')[0]

        ref_landmarks = np.load(os.path.join(self.path_ref_lmrk, c[0] + '.npy'))
        landmarks = np.load(os.path.join(self.path_lmrk, img_file + '.npy'))

        spk_emb = np.load(os.path.join(self.path_spk, audio_file))
        spk_emb = torch.FloatTensor(spk_emb)

        mfcc_path = os.path.join(self.all_data[idx])
        mfcc = np.load(mfcc_path, allow_pickle=True)
        mfcc = mfcc[:, 1:]
        mfcc = torch.FloatTensor(mfcc)
        data = torch.unsqueeze(mfcc, 0)
        #data = torch.tensor(np.transpose(data, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]

        input = {}
        out = {}

        input['speaker_emb'] = spk_emb
        input['audio'] = np.array(data, dtype='float32')
        input['emotion'] = torch.Tensor([emo_label[emotion]])
        input['landmarks'] = ref_landmarks
        out['landmarks'] = landmarks
        return input, out

class lmrk_img_dataset(Dataset):
    def __init__(self, dataset_dir,   path_ref_img, path_img):
        # data path
        self.all_data = dataset_dir
        self.path_ref_img = path_ref_img
        self.path_img = path_img
        self.trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.179, 0.1963, 0.159), (0.083, 0.066, 0.053))])

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):

        lmrks = np.load(self.all_data[idx])

        c = self.all_data[idx].split('/')[-1].split('_')
        img_file = self.all_data[idx].split('/')[-1].split('.')[0]

        img = cv2.imread(os.path.join(self.path_ref_img, c[0] + '.jpg'))
        img= cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = torch.Tensor(img).cuda()

        img_out = cv2.imread(os.path.join(self.path_img, img_file + '.jpg'))
        img_out = cv2.normalize(img_out, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        input = {}
        out = {}

        input['lmrks'] = torch.Tensor(lmrks).cuda()
        input['img_ref'] = torch.tensor(img)
        out['img'] = torch.Tensor(img_out).cuda()
        return input, out

