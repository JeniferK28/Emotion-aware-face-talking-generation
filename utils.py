from __future__ import print_function, division
import librosa
import numpy as np
import torch
import python_speech_features
import torch.autograd as grad
from resemblyzer import preprocess_wav, VoiceEncoder
import torch.nn.functional as F
import torch.nn as nn
import math
from functools import reduce
import cv2


#from hparam import hparam as hp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_centroids_prior(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid / len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids


def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid / (len(embeddings[speaker_num]) - 1)
    return centroid


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance
    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim_prior(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0), embeddings.size(1), centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance, centroid, dim=0) + 1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim


def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def calc_loss_prior(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum() + 1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

#data_sr = 16000, data_window=0.025, data_hop = 0.01, nmels = 40, data.tisv_frame = 180 (Max number of time steps in input after preproces)


def mfccs_and_spec(wav_file, wav_process=False, calc_mfccs=False, calc_mag_db=False):
    sound_file, _ = librosa.core.load(wav_file, sr=16000)
    window_length = int(0.025 * 16000)
    hop_length = int(0.01 * 16000)
    duration = 180 * 0.01 + 0.025

    # Cut silence and fix length
    # nfft = 512
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(16000 * duration)
        sound_file = librosa.util.fix_length(sound_file, length)

    spec = librosa.stft(sound_file, n_fft=512, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(16000, 512, n_mels=40)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    # db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T

    return mfccs, mel_db, mag_db

def copy_state_dict(state_dict, model, strip=None, replace=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and replace is None and name.startswith(strip):
            name = name[len(strip):]
        if strip is not None and replace is not None:
            name = name.replace(strip, replace)
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

# #############
# Get_spk_emb
# #############

def get_spk_emb(audio_file_dir, segment_len=960000):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dvs = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    resemblyzer_encoder = VoiceEncoder(device=dvs)

    wav = preprocess_wav(audio_file_dir)
    l = len(wav) // segment_len # segment_len = 16000 * 60
    l = np.max([1, l])
    all_embeds = []
    for i in range(l):
        mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
            wav[segment_len * i:segment_len* (i + 1)], return_partials=True, rate=2)
        all_embeds.append(mean_embeds)
    all_embeds = np.array(all_embeds)
    mean_embed = np.mean(all_embeds, axis=0)

    return mean_embed, all_embeds

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        #device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        return loss

def transform(point, center, scale, resolution, rotation=0, invert=False):
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if rotation != 0:
        rotation = -rotation
        r = np.eye(3)
        ang = rotation * math.pi / 180.0
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c

        t_ = np.eye(3)
        t_[0][2] = -resolution / 2.0
        t_[1][2] = -resolution / 2.0
        t_inv = torch.eye(3)
        t_inv[0][2] = resolution / 2.0
        t_inv[1][2] = resolution / 2.0
        t = reduce(np.matmul, [t_inv, r, t_, t])

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(int)

def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(preds[i, j], center, scale, hm.size(2), rot, True)

    return preds, preds_orig

def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''
    def draw_curve(img,idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(img,list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(img,list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(img,(range(22, 26)), color=(50, 205, 50))
    draw_curve(img,list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(img,list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(img,(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(img,list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(img,list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img

def area_of_triangle(pts):
    AB = pts[1, :] - pts[0, :]
    AC = pts[2, :] - pts[0, :]
    return 0.5 * np.linalg.norm(np.cross(AB, AC))

def area_of_polygon(pts):
    l = pts.shape[0]
    s = 0
    for i in range(1, l-1):
        s += area_of_triangle(pts[(0, i, i+1), :])
    return s

def norm_input_face(shape_3d):
    scale = 1.6 / (shape_3d[0, 0] - shape_3d[16, 0])
    shift = - 0.5 * (shape_3d[0, 0:2] + shape_3d[16, 0:2])
    shape_3d[:, 0:2] = (shape_3d[:, 0:2] + shift) * scale
    face_std = np.loadtxt('src/dataset/utils/STD_FACE_LANDMARKS.txt').reshape(68, 3)
    shape_3d[:, -1] = face_std[:, -1] * 0.1
    shape_3d[:, 0:2] = -shape_3d[:, 0:2]

    return shape_3d, scale, shift

def close_face_lip(fl):
    facelandmark = fl.reshape(-1, 68, 3)
    min_area_lip, idx = 999, 0
    for i, fls in enumerate(facelandmark):
        area_of_mouth = area_of_polygon(fls[list(range(60, 68)), 0:2])
        if (area_of_mouth < min_area_lip):
            min_area_lip = area_of_mouth
            idx = i
    return idx

def close_input_face_mouth(shape_3d, p1=0.7, p2=0.5):
    shape_3d = shape_3d.reshape((1, 68, 3))
    index1 = list(range(60 - 1, 55 - 1, -1))
    index2 = list(range(68 - 1, 65 - 1, -1))
    mean_out = 0.5 * (shape_3d[:, 49:54] + shape_3d[:, index1])
    mean_in = 0.5 * (shape_3d[:, 61:64] + shape_3d[:, index2])
    shape_3d[:, 50:53] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, list(range(59 - 1, 56 - 1, -1))] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d[:, 49] -= (shape_3d[:, 61] - mean_in[:, 0]) * p2
    shape_3d[:, 53] -= (shape_3d[:, 63] - mean_in[:, -1]) * p2
    shape_3d[:, 59] -= (shape_3d[:, 67] - mean_in[:, 0]) * p2
    shape_3d[:, 55] -= (shape_3d[:, 65] - mean_in[:, -1]) * p2
    # shape_3d[:, 61:64] = shape_3d[:, index2] = mean_in
    shape_3d[:, 61:64] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, index2] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d = shape_3d.reshape((68, 3))

    return shape_3d

#class audio_processing():

def audio_preprocessing(audio_file):
    mel = []
    speech, sr = librosa.load(audio_file, sr=16000)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)
    time_len = mfcc.shape[0]
    for idx in range(int((time_len - 28) / 4) + 1):
        input_mfcc = mfcc[4 * idx: 4 * idx + 28, :]
        mel.append(input_mfcc)
    return mel


import torchaudio

def wav2seq(audio_file, device):
    mel = []
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    speech, sr = librosa.load(audio_file, sr=16000)
    speech = torch.Tensor(speech).to(device)
    speech = torch.unsqueeze(speech,0)

    if sr != bundle.sample_rate:
        speech = torchaudio.functional.resample(speech, sr, bundle.sample_rate)

    with torch.inference_mode():
        features, _ = model.extract_features(speech)

    cnt_ft = features[0]

    #add = torch.zeros(1, 6, cnt_ft.size(2)).to(device)
    add = torch.zeros(1, 6 , cnt_ft.size(2)).to(device)
    #cnt_ft = torch.cat([add, cnt_ft, add], 1)
    cnt_ft = torch.cat([add, cnt_ft], 1)
    time_len = cnt_ft.shape[1]
    audio_size = int((time_len - 8)/2) + 1
    mfcc = np.zeros((audio_size,5, 10, 768))

    for idx in range(int(((time_len - 8) / 2)) + 1):
        input_mfcc = cnt_ft[0,2 * idx: 2 * idx + 10, :]
        mel.append(input_mfcc)

    for m in range(len(mel)):
        z = 0
        while z in range(6):
            if m<5:
                if (5-m-z)>0:
                    for i in range(5-m):
                        mfcc[m,z,:,:] = np.zeros((10,768))
                        z += 1
                mfcc[m, z-1, :, :] = mel[m - 4 + z].cpu().detach().numpy()
                z +=1
            else:
                if z<5:
                    mfcc[m,z,:,:] = mel[m-5+z].cpu().detach().numpy()
                z += 1

    return mfcc

def wav2data(audio_file, device):
    mel = []
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    speech, sr = librosa.load(audio_file, sr=16000)
    speech = torch.Tensor(speech).to(device)
    speech = torch.unsqueeze(speech,0)

    if sr != bundle.sample_rate:
        speech = torchaudio.functional.resample(speech, sr, bundle.sample_rate)

    with torch.inference_mode():
        features, _ = model.extract_features(speech)

    cnt_ft = features[0]
    #add = torch.zeros(1, 6, cnt_ft.size(2)).to(device)
    add = torch.zeros(1, 8, cnt_ft.size(2)).to(device)
    #cnt_ft = torch.cat([add, cnt_ft, add], 1)
    cnt_ft = torch.cat([add, cnt_ft], 1)
    time_len = cnt_ft.shape[1]

    for idx in range(int(((time_len - 10) / 2)) + 1):
        input_mfcc = cnt_ft[0,2 * idx: 2 * idx + 10, :]
        mel.append(input_mfcc)
    return mel





from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import clip

def a2l_data(shape_3d, audio_emb, emo, spk_emb, adj, vertices, s, pos, device):
    emo_vec = {'angry': [1, 0, 0, 0, 0, 0, 0, 0], 'contempt': [0, 1, 0, 0, 0, 0, 0, 0],
               'disgusted': [0, 0, 1, 0, 0, 0, 0, 0], 'fear': [0, 0, 0, 1, 0, 0, 0, 0],
               'happy': [0, 0, 0, 0, 1, 0, 0, 0],
               'neutral': [0, 0, 0, 0, 0, 1, 0, 0], 'sad': [0, 0, 0, 0, 0, 0, 1, 0],
               'surprised': [0, 0, 0, 0, 0, 0, 0, 1]}

    #mfcc = mfcc[:, 1:]
    #audio_emb = torch.FloatTensor(audio_emb)

    audio_emb = torch.unsqueeze(audio_emb, 0)
    audio_emb = torch.unsqueeze(audio_emb, 0).to(device)
    edge_index, edge_attr = from_scipy_sparse_matrix(vertices)
    shape_3d= torch.tensor(shape_3d).to(torch.float64)
    shape_3d = shape_3d[:, 0:2]

    audio= {}
    data= Data(x=shape_3d, edge_index=edge_index.contiguous(), edge_attr=edge_attr, pos=pos)
    lmrks = torch.unsqueeze(shape_3d, 0)
    lmrks = torch.unsqueeze(lmrks, 0).to(device)
    spk_emb = torch.FloatTensor(spk_emb)
    spk_emb= torch.unsqueeze(spk_emb, 0)
    emo_vec = torch.Tensor([emo_vec[emo]]).to(device)
    audio['mfcc'] =torch.tensor(audio_emb).to(device)
    audio['emo_vec'] = torch.unsqueeze(emo_vec, 0)
    audio['emb_spk'] = torch.FloatTensor(spk_emb).to(device)
    audio['emotion'] = torch.squeeze(clip.tokenize(emo)).to(device)
    audio['adj']= torch.FloatTensor(adj).to(device)
    audio['s'] = torch.tensor(s).to(device)

    return data.to(device), audio, lmrks

def a2l_seq(shape_3d, audio_emb, emo, spk_emb, adj, vertices, s, pos, device):
    emo_vec = {'angry': [1, 0, 0, 0, 0, 0, 0, 0], 'contempt': [0, 1, 0, 0, 0, 0, 0, 0],
               'disgusted': [0, 0, 1, 0, 0, 0, 0, 0], 'fear': [0, 0, 0, 1, 0, 0, 0, 0],
               'happy': [0, 0, 0, 0, 1, 0, 0, 0],
               'neutral': [0, 0, 0, 0, 0, 1, 0, 0], 'sad': [0, 0, 0, 0, 0, 0, 1, 0],
               'surprised': [0, 0, 0, 0, 0, 0, 0, 1]}

    #mfcc = mfcc[:, 1:]
    #audio_emb = torch.FloatTensor(audio_emb)

    #audio_emb = torch.unsqueeze(audio_emb, 0)
    audio_emb = torch.unsqueeze(audio_emb, 0).to(device)
    edge_index, edge_attr = from_scipy_sparse_matrix(vertices)
    shape_3d= torch.tensor(shape_3d).to(torch.float64)
    shape_3d = shape_3d[:, 0:2]

    audio= {}
    data= Data(x=shape_3d, edge_index=edge_index.contiguous(), edge_attr=edge_attr, pos=pos)
    lmrks = torch.unsqueeze(shape_3d, 0)
    lmrks = torch.unsqueeze(lmrks, 0).to(device)
    spk_emb = torch.FloatTensor(spk_emb)
    spk_emb= torch.unsqueeze(spk_emb, 0)
    emo_vec = torch.Tensor([emo_vec[emo]]).to(device)
    audio['mfcc'] =torch.tensor(audio_emb).to(device)
    audio['emo_vec'] = torch.unsqueeze(emo_vec, 0)
    audio['emb_spk'] = torch.FloatTensor(spk_emb).to(device)
    audio['emotion'] = torch.squeeze(clip.tokenize(emo)).to(device)
    audio['adj']= torch.FloatTensor(adj).to(device)
    audio['s'] = torch.tensor(s).to(device)

    return data.to(device), audio, lmrks

def loss_custom(real_lmrks, fake_lmrks,s, device):
    l1_loss = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    #Define number of landmarks per area
    n = torch.Tensor([17, 5 , 5 , 6, 6, 9, 12, 8]).to(device)
    #Define distance matrix
    real_dis = torch.zeros(real_lmrks.size(0), real_lmrks.size(2),real_lmrks.size(2),real_lmrks.size(3)).to(device)
    fake_dis = torch.zeros(real_lmrks.size(0), real_lmrks.size(2),real_lmrks.size(2),real_lmrks.size(3)).to(device)
    l_real = torch.zeros(real_lmrks.size(0),real_lmrks.size(2),real_lmrks.size(3)).to(device)
    l_fake = torch.zeros(real_lmrks.size(0),real_lmrks.size(2),real_lmrks.size(3)).to(device)
    for i in range(68):
        for j in range(68):
            real_dis[i][j] = torch.squeeze((real_lmrks[i] - real_lmrks[j]),0)
            fake_dis[i][j] = torch.squeeze((fake_lmrks[i] - fake_lmrks[j]),0)

    l_real[:, :, 0] = torch.mean(torch.div(torch.matmul(real_dis[:, :, :, 0], s),n - 1), 2)
    l_real[:, :, 1] = torch.mean(torch.div(torch.matmul(real_dis[:, :, :, 1], s), n - 1), 2)
    l_fake[:, :, 0] = torch.mean(torch.div(torch.matmul(fake_dis[:, :, :, 0], s), n - 1), 2)
    l_fake[:, :, 1] = torch.mean(torch.div(torch.matmul(fake_dis[:, :, :, 1], s), n - 1), 2)
    loss = MSE(l_fake,l_real)
    return loss


def graph_displacement(real_lrmks, fake_lmrks, img):
    graph = torch.zeros(real_lrmks.size(0),256,256)
    for i in range(real_lrmks.size(0)):
        x1 = real_lrmks[i, :, 0]
        x2 = fake_lmrks[i, :, 0]
        y1 = real_lrmks[i, :, 1]
        y2 = fake_lmrks[i, :, 1]
        dist = torch.sqrt(((x1 - x2)^2 + (y1 - y2)^2))
        graph[i, x1:x2, y1:y2] += dist
    heatmap_img = cv2.applyColorMap(graph, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    cv2.imshow('plot', super_imposed_img)

def loss_pre(lmrks_pre, lmrks):
    MSE = torch.nn.MSELoss()
    lmrks_mean = torch.mean(lmrks_pre,0)
    loss = MSE(lmrks_mean,lmrks)
    return loss