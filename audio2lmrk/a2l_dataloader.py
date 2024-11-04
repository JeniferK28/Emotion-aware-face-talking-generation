import torch
import os
from skimage import io, img_as_float32, transform
import random
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import clip
from utils import close_input_face_mouth
from torch_geometric.utils.convert import from_scipy_sparse_matrix


emo_vec= {'angry': [1,0,0,0,0,0,0,0], 'contempt':[0,1,0,0,0,0,0,0], 'disgusted':[0,0,1,0,0,0,0,0], 'fear':[0,0,0,1,0,0,0,0], 'happy':[0,0,0,0,1,0,0,0],
             'neutral':[0,0,0,0,0,1,0,0],'sad':[0,0,0,0,0,0,1,0], 'surprised':[0,0,0,0,0,0,0,1]}

emo_class= {'angry': 0, 'contempt':1, 'disgusted':2, 'fear':3, 'happy':4,
             'neutral':5,'sad':6, 'surprised':7}

id_label = {'M005':0,'M007':1,'M009':2,'M011':3,'M012':4,'M019':5,'M023':6,'M024':7,'M025':8,'M026':9,'M027':10,'M028':11,'M029':12,'M030':13,'M031':14,
            'M033':15,'M034':16,'M035':17,'M036':18,'M037':19,'M039':20,'M040':21,'M042':22,'W009':23,'W011':24,'W015':26,'W016':27,
            'W018':28,'W021':29,'W023':30,'W024':31,'W025':32,'W026':33,'W029':34,'W035':35,'W037':36,'W038':37,'W040':38}


class audio_lmrk_graph(Dataset):
    def __init__(self, dataset_dir, path_ref_lmrk, path_spk_emb, path_audio, path_lmrk, vertices, adj,pos, s, device):
        # data path
        self.all_data = dataset_dir
        self.path_ref_lmrk = path_ref_lmrk
        self.path_spk_emb = path_spk_emb
        self.path_audio = path_audio
        self.path_lmrk = path_lmrk
        self.vertices = vertices
        self.pos = pos
        self.s = s
        self.adj = adj
        self.device = device


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        emotion = self.all_data[idx].split('/')[-1].split('_')[1]
        c = self.all_data[idx].split('/')[-1].split('\\')[-1].split('_')
        index = c[5].split('.')[0]
        file_name = self.all_data[idx].split('/')[-1].split(index)[0]
        all_lmrks = []

        out_lmrks = np.load(os.path.join(self.path_lmrk, self.all_data[idx].split('/')[-1]))

        edge_index, edge_attr = from_scipy_sparse_matrix(self.vertices)

        out_lmrks = torch.tensor(out_lmrks).to(torch.float64)
        out = out_lmrks[:, 0:2]

        data_out = Data(x=out, edge_index=edge_index.contiguous(), edge_attr=edge_attr)
        out = torch.unsqueeze(out, 0).to(self.device)

        ref_lmrk = np.load(os.path.join(self.path_ref_lmrk, c[0] + '.npy'))

        ref_lmrk = torch.tensor(ref_lmrk).to(torch.float64)
        ref_lmrk = ref_lmrk[:,0:2]
        data_in = Data(x=ref_lmrk, edge_index=edge_index.contiguous(), edge_attr=edge_attr, pos=self.pos)
        input = torch.unsqueeze(ref_lmrk, 0).to(self.device)

        #neutral or emo
        file = c[0] + '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + str(int(c[4])) + '.npy'
        spk_emb = np.load(os.path.join(self.path_spk_emb, file))
        spk_emb = torch.FloatTensor(spk_emb).to(self.device)

        audio_emb = np.load(os.path.join(self.path_audio, self.all_data[idx]))

        audio_emb = torch.FloatTensor(audio_emb)
        audio = {}
        lmrks = {}
        # conv torch.unsqueze
        audio['mfcc'] = audio_emb.to(self.device)
        audio['emb_spk'] = spk_emb
        audio['emo_vec'] = torch.Tensor([emo_vec[emotion]]).to(self.device)
        audio['adj'] = torch.FloatTensor(self.adj).to(self.device)
        audio['s'] = self.s.to(self.device)
        audio['emo_class'] = torch.Tensor([emo_class[emotion]]).to(self.device)
        audio['emotion'] = torch.squeeze(clip.tokenize(emotion)).to(self.device)
        lmrks['out'] = out
        lmrks['in'] = input
        lmrks['pre'] = all_lmrks

        return data_in.to(self.device), data_out.to(self.device),lmrks, audio


class audio_lmrk_vertice(Dataset):
    def __init__(self, dataset_dir, path_ref_lmrk, path_spk_emb, path_audio, path_lmrk, device):
        # data path
        self.all_data = dataset_dir
        self.path_ref_lmrk = path_ref_lmrk
        self.path_spk_emb = path_spk_emb
        self.path_audio = path_audio
        self.path_lmrk = path_lmrk
        self.device = device

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):

        emotion = self.all_data[idx].split('/')[-1].split('_')[1]
        c = self.all_data[idx].split('/')[-1].split('\\')[-1].split('_')

        if int(c[4])<10:
            n = '00' + c[4]
        else : n = '0' + c[4]

        lmrk_file = c[0] + '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + n + '_' + c[5]
        lmrks = np.load(os.path.join(self.path_lmrk,lmrk_file))
        out = torch.FloatTensor(lmrks)
        out = torch.unsqueeze(out, 0).to(self.device)

        ref_lmrk = np.load(os.path.join(self.path_ref_lmrk, c[0] + '.npy'))
        data = torch.FloatTensor(ref_lmrk)
        data = torch.unsqueeze(data, 0).to(self.device)

        file = c[0]+ '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + c[4] + '.npy'
        spk_emb = np.load(os.path.join(self.path_spk_emb,file))
        spk_emb = torch.FloatTensor(spk_emb).to(self.device)

        mfcc = np.load(os.path.join(self.path_audio, self.all_data[idx]))

        mfcc = mfcc[:, 1:]
        mfcc = torch.FloatTensor(mfcc)
        mfcc = torch.unsqueeze(mfcc, 0).to(self.device)
        audio = {}

        audio['mfcc'] = mfcc
        audio['emb_spk'] = spk_emb
        audio['emo_vec'] = torch.Tensor([emo_vec[emotion]]).to(self.device)
        return data, out, audio


class audio_lmrk_graph_series(Dataset):
    def __init__(self, dataset_dir, path_ref_lmrk, path_spk_emb, path_audio, path_lmrk, vertices, adj,pos, s, device):
        # data path
        self.all_data = dataset_dir
        self.path_ref_lmrk = path_ref_lmrk
        self.path_spk_emb = path_spk_emb
        self.path_audio = path_audio
        self.path_lmrk = path_lmrk
        self.vertices = vertices
        self.pos = pos
        self.s = s
        self.adj = adj
        self.device = device


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        emotion = self.all_data[idx].split('/')[-1].split('_')[1]
        c = self.all_data[idx].split('/')[-1].split('\\')[-1].split('_')
        mffcs = torch.zeros([5,7*768,2])
        for i in range(5):
            if int(c[5])<6:
                index = c[0] + '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + c[4] + '_'+ str(int(c[5])-i)+ '.npy'
                mffc = np.load(os.path.join(self.path_audio, index))
                mffc = torch.FloatTensor(mffc)
                mffcs[5-i,:,:] = mffc


        out_lmrks = np.load(os.path.join(self.path_lmrk, self.all_data[idx].split('/')[-1]))

        edge_index, edge_attr = from_scipy_sparse_matrix(self.vertices)

        out_lmrks = torch.tensor(out_lmrks).to(torch.float64)
        out = out_lmrks[:, 0:2]

        data_out = Data(x=out, edge_index=edge_index.contiguous(), edge_attr=edge_attr)
        out = torch.unsqueeze(out, 0).to(self.device)

        ref_lmrk = np.load(os.path.join(self.path_ref_lmrk, c[0] + '.npy'))

        ref_lmrk = torch.tensor(ref_lmrk).to(torch.float64)
        ref_lmrk = ref_lmrk[:,0:2]
        data_in = Data(x=ref_lmrk, edge_index=edge_index.contiguous(), edge_attr=edge_attr, pos=self.pos)
        input = torch.unsqueeze(ref_lmrk, 0).to(self.device)

        file = c[0] + '_' + c[1] + '_' + c[2] + '_' + c[3] + '_' + str(int(c[4])) + '.npy'
        spk_emb = np.load(os.path.join(self.path_spk_emb, file))
        spk_emb = torch.FloatTensor(spk_emb).to(self.device)


        audio_emb = np.load(os.path.join(self.path_audio, self.all_data[idx]))
        audio_emb = torch.FloatTensor(audio_emb)
        audio = {}
        lmrks = {}
        # conv torch.unsqueze
        audio['mfcc'] = mffcs.to(self.device)
        audio['emb_spk'] = spk_emb
        audio['emo_vec'] = torch.Tensor([emo_vec[emotion]]).to(self.device)
        audio['adj'] = torch.FloatTensor(self.adj).to(self.device)
        audio['s'] = self.s.to(self.device)
        audio['emo_class'] = torch.Tensor([emo_class[emotion]]).to(self.device)
        audio['emotion'] = torch.squeeze(clip.tokenize(emotion)).to(self.device)
        #device = torch.device('cuda')
        lmrks['out'] = out
        lmrks['in'] = input


        return data_in.to(self.device), data_out.to(self.device),lmrks, audio
