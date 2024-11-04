import math

import torch
import clip
import torch_geometric
import numpy as np
import torch.nn as nn
from audio.ANet import ANet
from main import parse_args
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv, GraphConv
from torch_geometric.nn import Set2Set, dense_diff_pool, knn_interpolate
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import to_dense_adj
from utils import loss_custom, loss_pre
from lmrk_model import lmrkNet
import math
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.fc_v = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(256,256)
        #self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        #self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, input):

        Q = self.fc_q(input)
        K = self.fc_k(input)
        V = self.fc_v(input)
        d_k = K.shape[-1]
        scores = torch.matmul(Q,K.transpose(-2,-1))/ math.sqrt(d_k)
        scores = F.softmax(scores, dim= -1)
        output = torch.matmul(scores, V)

        return self.out(output)


class lmrk_encoder_fc(nn.Module):
    def __init__(self):
        super(lmrk_encoder_fc, self).__init__()
        self.fc1 = nn.Linear(68*2, 256, dtype=torch.float64)
        self.bactcn_norm = nn.BatchNorm1d(256,dtype=torch.float64)
        self.fc2 = nn.Linear(256,512, dtype=torch.float64)


    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.fc1(input)
        x = self.fc2(x)
        return x


class lmrk_encoder(nn.Module):
    def __init__(self):
        super(lmrk_encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,32, (5,1), 1, 0, dtype=torch.float64)
        self.conv2 = nn.Conv2d(32,64,(5,1),1,0, dtype=torch.float64)
        self.conv3 = nn.Conv2d(64,128,(5,1),1,0, dtype=torch.float64)
        self.bn1 = nn.BatchNorm2d(32, dtype=torch.float64)
        self.bn2 = nn.BatchNorm2d(64, dtype=torch.float64)
        self.bn3 = nn.BatchNorm2d(128, dtype=torch.float64)
        self.pooling = nn.MaxPool2d((2,2),(2,2))


    def forward(self, input):
        x = self.conv1(input).relu()
        x = self.bn1(x)
        x = self.conv2(x).relu()
        x = self.bn2(x)
        x = self.conv3(x).relu()

        # Pool to global representation
        x = self.pooling(x)

        return x


class lmrk_encoder_h(nn.Module):
    def __init__(self):
        super(lmrk_encoder_h, self).__init__()
        self.feature_size = 2
        self.encoder_embedding_size = 128

        self.conv1 = GraphConv(self.feature_size, self.encoder_embedding_size).double()
        self.conv2 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        self.conv3 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.bn3 = BatchNorm(self.encoder_embedding_size)

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=2)


    def forward(self, input, adj, s,device, train):
        x = self.conv1(input.x, input.edge_index).relu()
        x = self.conv2(x, input.edge_index).relu()
        x = self.conv3(x, input.edge_index).relu()

        x_resize = torch.reshape(x,(input.num_graphs,68,128)).to(torch.float32)

        # Pool to global representation
        x, adj, l1, e1 = dense_diff_pool(x_resize.to(device), adj, s)

        return x, adj, l1, e1, input.pos


class lmrk_discriminator(nn.Module):
    def __init__(self):
        super(lmrk_discriminator, self).__init__()
        self.feature_size = 2
        self.encoder_embedding_size = 128

        self.conv1 = GraphConv(self.feature_size, self.encoder_embedding_size).double()
        self.conv2 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        self.conv3 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.bn3 = BatchNorm(self.encoder_embedding_size)


        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=2)
        self.maxpooling = nn.MaxPool2d((1,8))
        self.fc = nn.Linear(128,1)


    def forward(self, input, adj, s,device, train):
        x = self.conv1(input.x, input.edge_index).relu()
        x = self.conv2(x, input.edge_index).relu()
        x = self.conv3(x, input.edge_index).relu()

        if train:
            x_resize = torch.reshape(x, (input.num_graphs, 68, 128)).to(torch.float32)
        else: x_resize = torch.reshape(x, (1, 68, 128)).to(torch.float32)
        x, adj, l1, e1 = dense_diff_pool(x_resize.to(device), adj, s)
        x = self.maxpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class lmrk_decoder_up(nn.Module):
    def __init__(self):
        super(lmrk_decoder_up, self).__init__()
        #self.fc = nn.Linear(1024, 128)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1792, 256, kernel_size=4, stride=2, padding=1, bias=True),  # 2,2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 2), stride=2, padding=1, bias=True),  # 4,2
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 12,3
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 28,3
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(16, 2), stride=2, padding=1, bias=True),  # 56,3

            nn.Tanh(),
        )

    def forward(self, x, ct_emb, spk_emb, emo_vec):
         # connect tensors inputs and dimension
        x_fts = x.view(x.size(0), -1)
        features = torch.cat([x_fts,ct_emb, spk_emb, emo_vec], 1)
        features = torch.unsqueeze(features, 2)
        features = torch.unsqueeze(features, 3)
        x = self.deconv1(features)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = 90 * self.deconv4(x)  # [1, 1,x, y]
        return x

class lmrk_decoder_fc(nn.Module):
    def __init__(self):
        super(lmrk_decoder_fc, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=True),  # 2,2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 2), stride=2, padding=1, bias=True),  # 4,2
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 12,3
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 28,3
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(16, 2), stride=2, padding=1, bias=True),  # 56,3
            nn.Tanh(),
        )

    def forward(self, x, ct_emb, spk_emb, emo_vec):
        features = torch.cat([x, ct_emb, spk_emb, emo_vec], 1)  # connect tensors inputs and dimension
        features = torch.unsqueeze(features, 2)
        features = torch.unsqueeze(features, 3)
        x = self.deconv1(features)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = 90 * self.deconv4(x)  # [1, 1,x, y]
        return x

class lmrk_decoder_ct(nn.Module):
    def __init__(self):
        super(lmrk_decoder_ct, self).__init__()
        #1536, 1280, 768, 3584, 4352
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1792 + 256, 256, kernel_size=4, stride=2, padding=1, bias=True),  # 2,2
            nn.BatchNorm2d(256),
            nn.ELU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 2), stride=2, padding=1, bias=True),  # 4,2
            nn.BatchNorm2d(128),
            nn.ELU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 12,3
            nn.BatchNorm2d(64),
            nn.ELU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 28,3
            nn.BatchNorm2d(32),
            nn.ELU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(16, 2), stride=2, padding=1, bias=True),  # 56,3
            nn.Tanh(),
        )

    def forward(self, x, ct_emb, spk_emb, emo_vec):
            x_fts = x.view(x.size(0), -1)
            features = torch.cat([x_fts, ct_emb, spk_emb, emo_vec], 1)  # connect tensors inputs and dimension
            #features = torch.cat([x_fts, ct_emb, emo_vec], 1)
            features = torch.unsqueeze(features, 2)
            features = torch.unsqueeze(features, 3)
            x = self.deconv1(features.float())
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = 90 * self.deconv4(x)  # [1, 1,x, y]
            return x

class lmrk_decoder_no_ct(nn.Module):
    def __init__(self):
        super(lmrk_decoder_no_ct, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1792, 256, kernel_size=4, stride=2, padding=1, bias=True),  # 2,2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 2), stride=2, padding=1, bias=True),  # 4,2
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 12,3
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(8, 2), stride=2, padding=1, bias=True),  # 28,3
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(16, 2), stride=2, padding=1, bias=True),  # 56,3
            nn.Tanh(),
        )


    def forward(self, x, ct_emb, spk_emb, emo_vec):
            x_fts = x.view(x.size(0), -1)
            features = torch.cat([x_fts, ct_emb, spk_emb], 1)  # connect tensors inputs and dimension
            features = torch.unsqueeze(features, 2)
            features = torch.unsqueeze(features, 3)
            x = self.deconv1(features)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = 90 * self.deconv4(x)  # [1, 1,x, y]
            return x

    def reparameterize(self, mu, logvar):
        if self.training:
            # Get standard deviation
            std = torch.exp(logvar)
            # Returns random numbers from a normal distribution
            eps = torch.randn_like(std)
            # Return sampled values
            return eps.mul(std).add_(mu)
        else:
            return

class emotion(nn.Module):
    def __init__(self):
        super(emotion, self).__init__()
        self.emotion_encoder = nn.Sequential(
            nn.Conv1d(1, 64, 3),
            nn.ELU(True),
            nn.Dropout(0.25),
            nn.Conv1d(64, 128, 3),
            nn.ELU(True),

        )
    def forward(self, emo):
        x = self.emotion_encoder(emo)
        x = x.view(x.size(0), -1)
        return x


class audio_fc(nn.Module):
    def __init__(self):
        super(audio_fc, self).__init__()
        self.audio_encoder= nn.Sequential(
            #7680, 10752
            nn.Linear(7680, 1024),
            nn.ELU(True),
            nn.Linear(1024, 256),
            nn.ELU(True),
        )
        #self.att = Attention()

    def forward(self, audio):
        audio = audio.view(audio.size(0), -1)
        out = self.audio_encoder(audio)
        #out = self.att(out)
        return out

class audio_conv(nn.Module):
    def __init__(self):
        super(audio_conv, self).__init__()
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(4, stride=(2, 4)),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(4, stride=(2, 4))
        )
        self.audio_encoder_fc = nn.Sequential(
            nn.Linear(1024 * 24, 2048),
            nn.ELU(True),
            nn.Linear(2048, 256),
            nn.ELU(True),
        )

    def forward(self, audio):
        feature = self.audio_encoder(audio)
        feature = feature.view(feature.size(0), -1)
        x = self.audio_encoder_fc(feature)
        return x



class a2lNet(nn.Module):
    def __init__(self, config):
        super(a2lNet, self).__init__()
        self.model_clip, self.preprocess = clip.load('ViT-B/32', config.device)
        self.device = config.device
        self.audio_fc = audio_fc()
        self.lmrk_encoder = lmrk_encoder_h()
        self.lmrk_decoder_ct = lmrk_decoder_ct()
        self.lmrk_D = lmrk_discriminator()
        self.optimizer = torch.optim.Adam(list(self.lmrk_encoder.parameters())
                                        #  + list(self.emo_enc.parameters())
                                          + list(self.audio_fc.parameters())
                                        #  + list(self.lmrk_decoder_no_ct.parameters())
                                          + list(self.lmrk_decoder_ct.parameters()), config.lr,
                                          betas=(config.beta1, config.beta2))
        self.CroEn_loss = nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.weight = config.triplet_weight
        self.criterion = nn.BCELoss()

    

    def forward(self, x, audio, lmrks, device, train):
        lmrk_emb, adj, l1, e1, pos= self.lmrk_encoder(x, audio['adj'], audio['s'], device, train = train)
        ct_emb = self.audio_fc(audio['mfcc'])
        text_features = self.model_clip.encode_text(torch.unsqueeze(audio['emotion'],0))
        lmrk_ct= self.lmrk_decoder_ct(lmrk_emb, ct_emb, audio['emb_spk'], text_features)
        lmrk_reconstructed = lmrk_ct #+ lmrk_no_ct
        return lmrk_reconstructed + lmrks, ct_emb, lmrk_emb

    def cross(self, x, audio, emo_vec, spk_emb, adj, s,text_features,device, train, lmrks_in):

        lmrk_emb, adj, l1, e1, pos= self.lmrk_encoder(x, adj, s, device, train = train)
        ct_emb = self.audio_fc(audio)
        lmrk_ct= self.lmrk_decoder_ct(lmrk_emb, ct_emb, spk_emb, text_features)
        lmrk_reconstructed = lmrk_ct #+ lmrk_no_ct
        return lmrk_reconstructed


    def process(self, data_in, data_out, out, audio,train):

        mfcc = audio['mfcc']
        spk_emb = audio['emb_spk']
        emo_vec = audio['emo_vec']
        x = data_in
        y = data_out
        lmrks_out = out['out']
        lmrks_in = out['in']
        adj = audio['adj']
        s = audio['s']
        emo_text =audio['emotion']
        losses = {}
        text_features = self.model_clip.encode_text(emo_text)
        displacement = self.cross(x, mfcc, emo_vec, spk_emb, adj, s,text_features, self.device, train, lmrks_in)
        real_label = 1.
        fake_label = 0.
        b_size = audio['emb_spk'].size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)

        losses['l1'] = self.MSE(displacement + lmrks_in, lmrks_out)
  
        return displacement+lmrks_in, losses


    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        #self.optimizerD.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.optimizerD.step()


    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)


    def train_func(self, data, data_out, out, audio):
        self.lmrk_encoder.train()
        self.lmrk_decoder_ct.train()
        self.audio_fc.train()
        outputs, losses = self.process(data, data_out, out, audio, train=True)
        self.update_network(losses)

        return outputs, losses


    def val_func(self, data, data_out, out, audio):
        self.lmrk_encoder.eval()
        self.lmrk_decoder_ct.eval()
        self.audio_fc.eval()

        with torch.no_grad():
            outputs, losses = self.process(data, data_out, out, audio, train=False)

        return outputs, losses

class a2lNet_pretrain(nn.Module):
    def __init__(self, config):
        super(a2lNet_pretrain, self).__init__()
        self.model_clip, self.preprocess = clip.load('ViT-B/32', config.device)
        self.device = config.device
        self.u = parse_args()
        self.lmrk_net = lmrkNet(self.u)
        self.lmrk_net.load_state_dict(torch.load(config.lmrk_pretrain))
        self.audio_fc = audio_fc()
        self.lmrk_decoder_ct = lmrk_decoder_ct()
        self.lmrk_D = lmrk_discriminator()
        self.optimizer = torch.optim.Adam(list(self.audio_fc.parameters())
                                          + list(self.lmrk_decoder_ct.parameters()), config.lr,
                                          betas=(config.beta1, config.beta2))
        self.CroEn_loss = nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.weight = config.triplet_weight
        self.criterion = nn.BCELoss()

    def forward(self, x, audio, lmrks, device, train):
        self.lmrk_net.eval()
        lmrk_emb = self.lmrk_net(x, audio['adj'], audio['s'], device, train=train)
        ct_emb = self.audio_fc(audio['mfcc'])
        text_features = self.model_clip.encode_text(torch.unsqueeze(audio['emotion'], 0))
        lmrk_ct = self.lmrk_decoder_ct(lmrk_emb, ct_emb, audio['emb_spk'], text_features)
        lmrk_reconstructed = lmrk_ct  # + lmrk_no_ct
        return lmrk_reconstructed + lmrks, ct_emb, lmrk_emb


    def cross(self, x, audio, emo_vec, spk_emb, adj, s, text_features, device, train, lmrks_in):
        self.lmrk_net.eval()
        lmrk_emb = self.lmrk_net(x, adj, s, device, train=train)
        ct_emb = self.audio_fc(audio)
        lmrk_ct = self.lmrk_decoder_ct(lmrk_emb, ct_emb, spk_emb, text_features)
        lmrk_reconstructed = lmrk_ct  # + lmrk_no_ct
        return lmrk_reconstructed

    def process(self, data_in, data_out, out, audio, train):
        mfcc = audio['mfcc']
        spk_emb = audio['emb_spk']
        emo_vec = audio['emo_vec']
        x = data_in
        y = data_out
        lmrks_out = out['out']
        lmrks_in = out['in']
        adj = audio['adj']
        s = audio['s']
        emo_text = audio['emotion']
        losses = {}
        text_features = self.model_clip.encode_text(emo_text)
        displacement = self.cross(x, mfcc, emo_vec, spk_emb, adj, s, text_features, self.device, train, lmrks_in)
        real_label = 1.
        fake_label = 0.
        b_size = audio['emb_spk'].size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)

        losses['l1'] = self.MSE(displacement[:,:,17:48,:] + lmrks_in[:,:,17:48,:], lmrks_out[:,:,17:48,:])
        losses['l1_mouth'] = self.MSE(displacement[:,:,48:68,:] + lmrks_in[:,:,48:68,:], lmrks_out[:,:,48:68,:]) + self.MSE(displacement[:,:,0:17,:] + lmrks_in[:,:,0:17,:], lmrks_out[:,:,0:17,:])

        return displacement + lmrks_in, losses

    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data, data_out, out, audio):
        #self.lmrk_encoder.train()
        self.lmrk_decoder_ct.train()
        self.audio_fc.train()
        outputs, losses = self.process(data, data_out, out, audio, train=True)
        self.update_network(losses)

        return outputs, losses

    def val_func(self, data, data_out, out, audio):
        self.lmrk_decoder_ct.eval()
        self.audio_fc.eval()

        with torch.no_grad():
            outputs, losses = self.process(data, data_out, out, audio, train=False)

        return outputs, losses

class att(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(att,self).__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs):

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]


        energy = torch.tanh(self.attn(encoder_outputs))  # [batch_size,seq_len,512]

        attention = self.v(energy).squeeze(2)  # [batch_size,seq_len]
        return F.softmax(attention, dim=1)

class decoder_lstm(nn.Module):
    def __init__(self):
        super(decoder_lstm,self).__init__()

        self.batch_first = True
        self.ldmk_dim = 136*5
        self.enc_hid_dim = 256
        self.dec_hid_dim = 256
        self.attention = att(self.enc_hid_dim, self.dec_hid_dim)
        self.rnn = nn.GRU(self.enc_hid_dim , self.dec_hid_dim, batch_first=self.batch_first,
                          bidirectional=False, dropout=0.2)

        self.fc_out = nn.Linear(self.enc_hid_dim + self.dec_hid_dim, self.ldmk_dim)  # 最开始的fc
   

    def forward(self, encoder_outputs):

        a = self.attention(encoder_outputs)
     
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)  # a.shape=[batch_size,1,5] encoder_outputs.shape=[batch_size,5,1024]
        rnn_input = weighted

        output, hidden = self.rnn(rnn_input)  # output.shape=[batch_size,1,512]

        assert (output.transpose(0, 1) == hidden).all()

        prediction = self.fc_out(torch.cat((output, weighted), dim=2))  # 这里线性层的多少有影响吗

        return prediction, hidden.squeeze(0), a

class lmrk_series(nn.Module):
    def __init__(self, config):
        super(lmrk_series, self).__init__()
        self.model_clip, self.preprocess = clip.load('ViT-B/32', config.device)
        self.device = config.device
        self.a2l = a2lNet(config)
        self.a2l.load_state_dict(torch.load(config.audio_pretrain))
        self.decoder_lstm = decoder_lstm()

        self.optimizer = torch.optim.Adam(list(self.decoder_lstm.parameters()), lr=0.0001,
                                          weight_decay=0.01)

        self.MSE = nn.MSELoss()

    def forward(self, x, audio, mfcc, lmrks, device, train):
        lmrk = torch.zeros((256,136,5))
        for i in range(5):
            audio['mfcc'] = torch.squeeze(mfcc[i, :, :, :])
            lmrk_reconstructed, ct_emb, lmrk_emb = self.a2l(x, audio, lmrks, device, train)
            lmrk_reconstructed = lmrk_reconstructed.view(lmrk_reconstructed.size(0), -1)
            lmrk[:,:,i] = (lmrk_reconstructed)

        lmrk = lmrk.view(lmrk.size(0),-1)
        output = self.decoder_lstm(lmrk)
        return output


    def cross(self, x, audio, mfcc, lmrks, device, train):
        lmrk = torch.zeros((256, 136, 5))
        for i in range(5):
            audio['mfcc'] = torch.squeeze(mfcc[i, :, :, :])
            lmrk_reconstructed, ct_emb, lmrk_emb = self.a2l(x, audio, lmrks, device, train)
            lmrk_reconstructed = lmrk_reconstructed.view(lmrk_reconstructed.size(0), -1)
            lmrk[:, :, i] = (lmrk_reconstructed)

        lmrk = lmrk.view(lmrk.size(0), -1)
        output = self.decoder_lstm(lmrk)
        return output
    def process(self, data_in, data_out, out, audio, train):
        mfcc = audio['mfcc']
        spk_emb = audio['emb_spk']
        emo_vec = audio['emo_vec']
        x = data_in
        y = data_out
        lmrks_out = out['out']
        lmrks_in = out['in']
        adj = audio['adj']
        s = audio['s']
        emo_text = audio['emotion']
        losses = {}
        text_features = self.model_clip.encode_text(emo_text)
        displacement = self.cross(x, mfcc, emo_vec, spk_emb, adj, s, text_features, self.device, train, lmrks_in)
        real_label = 1.
        fake_label = 0.
        b_size = audio['emb_spk'].size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)

        losses['l1'] = self.MSE(displacement + lmrks_in, lmrks_out)
        losses['l1'] = 0.5 * self.MSE(displacement[:,:,17:48,:] + lmrks_in[:,:,17:48,:], lmrks_out[:,:,17:48,:])

        return displacement + lmrks_in, losses

    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data, data_out, out, audio):
        self.decoder_lstm.train()
        outputs, losses = self.process(data, data_out, out, audio, train=True)
        self.update_network(losses)

        return outputs, losses

    def val_func(self, data, data_out, out, audio):
        self.decoder_lstm.eval()

        with torch.no_grad():
            outputs, losses = self.process(data, data_out, out, audio, train=False)

        return outputs, losses


def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd = logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div

def slice_graph_targets(graph_id, edge_targets, node_targets, batch_index):

    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph edge targets
    graph_edge_targets = edge_targets[graph_mask][:, graph_mask]
    # Get triangular upper part of adjacency matrix for targets
    size = graph_edge_targets.shape[0]
    triu_indices = torch.triu_indices(size, size, offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    graph_edge_targets = graph_edge_targets[triu_mask]
    # Slice node targets
    graph_node_targets = node_targets[graph_mask]
    return graph_edge_targets, graph_node_targets


def slice_graph_predictions(triu_logits, node_logits, graph_triu_size, triu_start_point, graph_size,
                            node_start_point):

    #Slice edge logits
    graph_logits_triu = torch.squeeze(triu_logits[triu_start_point:triu_start_point + graph_triu_size])
    #Slice node logits
    graph_node_logits = torch.squeeze(node_logits[node_start_point:node_start_point + graph_size])
    return graph_logits_triu, graph_node_logits



