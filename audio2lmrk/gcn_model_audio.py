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
#from audio2lmrk.lmrk_model import lmrkNet
from lmrk_model import lmrkNet
import math
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        #batch_size = encoder_outputs.shape[0]
        #src_len = encoder_outputs.shape[1]
        #hidden = hidden.transpose(0, 1).repeat(1, src_len, 1)
        #input = input.view(input.size(0), -1)
        Q = self.fc_q(input)
        K = self.fc_k(input)
        V = self.fc_v(input)
        d_k = K.shape[-1]
        # ...

        scores = torch.matmul(Q,K.transpose(-2,-1))/ math.sqrt(d_k)
        scores = F.softmax(scores, dim= -1)
        output = torch.matmul(scores, V)

        #energy = torch.matmul(Q, K.permute(0, 1)) / math.sqrt(d_k)
        # ...
        #attention = torch.softmax(energy, dim=-1)
        #x1 = torch.matmul(attention, V)

        #energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size,seq_len,512]

        #attention = self.v(energy).squeeze(2)  # [batch_size,seq_len]
        #return F.softmax(attention, dim=1)
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
        #x = self.bactcn_norm(x)
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
        #x = self.bn3(x)
        #x = self.conv4(x, edge_index).relu()

        # Pool to global representation
        x = self.pooling(x)

        # Latent transform layers
        #mu = self.mu_transform(x)
        #logvar = self.logvar_transform(x)

        return x


class lmrk_encoder_h(nn.Module):
    def __init__(self):
        super(lmrk_encoder_h, self).__init__()
        self.feature_size = 2
        self.encoder_embedding_size = 128

        self.conv1 = GraphConv(self.feature_size, self.encoder_embedding_size).double()
        self.conv2 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        self.conv3 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        # self.conv1 = TransformerConv(self.feature_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        # self.conv2 = TransformerConv(self.encoder_embedding_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        # self.conv3 = TransformerConv(self.encoder_embedding_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        # self.conv4 = TransformerConv(self.encoder_embedding_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=2)


    def forward(self, input, adj, s,device, train):
        x = self.conv1(input.x, input.edge_index).relu()
        #x = self.bn1(x)
        x = self.conv2(x, input.edge_index).relu()
        #x = self.bn2(x)
        x = self.conv3(x, input.edge_index).relu()
        # x_resize = torch.zeros((1, 68, 128))
        # c = 0
        # for i in range(1):
        #     x_resize[i, :, :] = x[68 * c:68 * (c + 1), :]
        #     c += 1
        #if train == True :

        x_resize = torch.reshape(x,(input.num_graphs,68,128)).to(torch.float32)
        #else:
        #x_resize = torch.reshape(x, (1, 68, 128)).to(torch.float32)
        #x = self.bn3(x)
        #x = self.conv4(x, edge_index).relu()

        # Pool to global representation
        #x = self.pooling(x, batch_index)
        x, adj, l1, e1 = dense_diff_pool(x_resize.to(device), adj, s)
        # Latent transform layers
        #mu = self.mu_transform(x)
        #logvar = self.logvar_transform(x)

        return x, adj, l1, e1, input.pos


class lmrk_discriminator(nn.Module):
    def __init__(self):
        super(lmrk_discriminator, self).__init__()
        self.feature_size = 2
        self.encoder_embedding_size = 128

        self.conv1 = GraphConv(self.feature_size, self.encoder_embedding_size).double()
        self.conv2 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        self.conv3 = GraphConv(self.encoder_embedding_size, self.encoder_embedding_size).double()
        # self.conv1 = TransformerConv(self.feature_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        # self.conv2 = TransformerConv(self.encoder_embedding_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        # self.conv3 = TransformerConv(self.encoder_embedding_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        # self.conv4 = TransformerConv(self.encoder_embedding_size,
        #                              self.encoder_embedding_size,
        #                              heads=4,
        #                              concat=False,
        #                              beta=True)

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=2)
        self.maxpooling = nn.MaxPool2d((1,8))
        self.fc = nn.Linear(128,1)


    def forward(self, input, adj, s,device, train):
        x = self.conv1(input.x, input.edge_index).relu()
        #x = self.bn1(x)
        x = self.conv2(x, input.edge_index).relu()
        #x = self.bn2(x)
        x = self.conv3(x, input.edge_index).relu()
        #x_resize = torch.zeros((1,68,128))
        # c= 0
        #
        # for i in range(1):
        #     x_resize[i,:,:] = x[68*c:68*(c+1),:]
        #     c += 1
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
        #features = self.fc(features)
        x_fts = x.view(x.size(0), -1)
        features = torch.cat([x_fts,ct_emb, spk_emb, emo_vec], 1)
        features = torch.unsqueeze(features, 2)
        features = torch.unsqueeze(features, 3)
        # for i in range(8):
        #         x_fts[:,i,128:256] = features
        #knn_interpolate(x_fts, pos, pos_skip, batch, batch_skip, k=self.k)
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
        #self.u = parse_args()
        #self.Anet = ANet(self.u)
        #self.Anet.load_state_dict(torch.load(config.audio_pretrain))
        self.audio_fc = audio_fc()
        #self.audio_fc = Attention()
        #self.emo_enc = emotion()
        #self.lmrk_encoder =lmrk_encoder()
        self.lmrk_encoder = lmrk_encoder_h()
        #self.lmrk_encoder = lmrk_encoder_fc()
        #self.classify = class_emo()
        #self.lmrk_decoder = lmrk_decoder_up()
        #self.lmrk_decoder = lmrk_decoder_fc()
        self.lmrk_decoder_ct = lmrk_decoder_ct()
        #self.lmrk_decoder_no_ct = lmrk_decoder_no_ct()
        self.lmrk_D = lmrk_discriminator()
        #self.optimizerD = torch.optim.Adam(self.lmrk_D.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
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
        #self.Anet.eval()
        lmrk_emb, adj, l1, e1, pos= self.lmrk_encoder(x, audio['adj'], audio['s'], device, train = train)
        #lmrk_emb = self.lmrk_encoder(lmrks)
        ct_emb = self.audio_fc(audio['mfcc'])
        text_features = self.model_clip.encode_text(torch.unsqueeze(audio['emotion'],0))
        #emo_emb = self.emo_enc(audio['emo_vec'])
        #audio_reconstructed, emo_c, ct_emb, spk_emb, emo_emb = self.Anet(audio['mfcc'], audio['emo_vec'], audio['emb_spk'])
        #lmrk_reconstructed = self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        #lmrk_reconstructed =  self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        #lmrk_ct = self.lmrk_decoder_ct(lmrk_emb, ct_emb, audio['emb_spk'], emo_emb)
        lmrk_ct= self.lmrk_decoder_ct(lmrk_emb, ct_emb, audio['emb_spk'], text_features)
        #lmrk_no_ct = self.lmrk_decoder_no_ct(lmrk_emb, ct_emb, audio['emb_spk'], emo_emb)
        lmrk_reconstructed = lmrk_ct #+ lmrk_no_ct
        return lmrk_reconstructed + lmrks, ct_emb, lmrk_emb

    # def compute_acc(self, input_label, out):
    #     _, pred = out.topk(1, 1)
    #     pred0 = pred.squeeze().data
    #     acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
    #     return acc

    def cross(self, x, audio, emo_vec, spk_emb, adj, s,text_features,device, train, lmrks_in):

        #lmrk_emb = self.lmrk_encoder(lmrks_in)
        lmrk_emb, adj, l1, e1, pos= self.lmrk_encoder(x, adj, s, device, train = train)
        ct_emb = self.audio_fc(audio)
        #emo_emb = self.emo_enc(emo_vec)
        #audio_reconstructed, emo_c, ct_emb, spk_emb, emo_emb = self.Anet(audio, emo_vec, spk_emb)
        #lmrk_reconstructed = self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        #lmrk_reconstructed =  self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        #lmrk_ct = self.lmrk_decoder_ct(lmrk_emb, ct_emb, spk_emb, emo_emb)
        lmrk_ct= self.lmrk_decoder_ct(lmrk_emb, ct_emb, spk_emb, text_features)
        #lmrk_no_ct = self.lmrk_decoder_no_ct(lmrk_emb, ct_emb, spk_emb, emo_emb)
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

        # if train:
        #
        #     self.lmrk_D.zero_grad()
        #     output = self.lmrk_D(y, adj, s, self.device, train)
        #     err_real = 0.5 * torch.mean((output-label)**2)
        #     err_real.backward(retain_graph=True)
        #
        #     label.fill_(fake_label)
        #     new_lmrks = displacement + lmrks_in
        #     y.x = torch.reshape(new_lmrks, (y.num_graphs * 68, 2)).to(torch.float64)
        #     output = self.lmrk_D(y,adj, s, self.device, train)
        #     err_fake = 0.5 * torch.mean((output-label)**2)
        #     err_fake.backward(retain_graph=True)
        #
        #     self.optimizerD.step()
        #
        #     #lips_loss = self.l1loss(displacement[:, :, 48:68, :] + lmrks_in[:, :, 48:68, :], lmrks_out[:, :, 48:68, :])
        # label.fill_(real_label)  # fake labels are real for generator cost
        #     # Since we just updated D, perform another forward pass of all-fake batch through D
        # output = self.lmrk_D(y,adj, s, self.device, train).view(-1)
        #
        #     # Calculate G's loss based on this output
        # losses['errG'] = 0.25 * torch.mean((output - label)**2)

        #jaw_loss = self.l1loss(displacement[:, :, 3:14, :] + lmrks_in[:, :, 3:14, :], lmrks_out[:, :, 3:14, :])
        #losses['MSE'] = 0.5 * loss_custom(lmrks_out, displacement + lmrks_in, s,self.device)
        losses['l1'] = self.MSE(displacement + lmrks_in, lmrks_out)
        #losses['l1'] = 0.5 * self.MSE(displacement[:,:,17:48,:] + lmrks_in[:,:,17:48,:], lmrks_out[:,:,17:48,:])
        #losses['l1_mouth'] = self.MSE(displacement[:,:,48:68,:] + lmrks_in[:,:,48:68,:], lmrks_out[:,:,48:68,:]) + self.MSE(displacement[:,:,0:17,:] + lmrks_in[:,:,0:17,:], lmrks_out[:,:,0:17,:])
        #losses['pre'] = 0.5 * loss_pre(out['pre'], displacement + lmrks_in)


        # outputs_dict = {
        #     "out_1": out1,
        #     "out_2": out2,
        # }
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
        #self.lmrk_decoder_no_ct.train()
        #self.lmrk_D.train()
        #self.emo_enc.train()
        self.audio_fc.train()
        outputs, losses = self.process(data, data_out, out, audio, train=True)
        self.update_network(losses)

        return outputs, losses


    def val_func(self, data, data_out, out, audio):
        self.lmrk_encoder.eval()
        self.lmrk_decoder_ct.eval()
        #self.lmrk_decoder_no_ct.eval()
        #self.lmrk_D.eval()
        #self.emo_enc.eval()
        self.audio_fc.eval()
        #self.lmrk_D.eval()

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
        #lmrk_emb, adj, l1, e1, pos = self.lmrk_net(x, audio['adj'], audio['s'], device, train=train)
        lmrk_emb = self.lmrk_net(x, audio['adj'], audio['s'], device, train=train)
        # lmrk_emb = self.lmrk_encoder(lmrks['in'])
        ct_emb = self.audio_fc(audio['mfcc'])
        text_features = self.model_clip.encode_text(torch.unsqueeze(audio['emotion'], 0))
        # emo_emb = self.emo_enc(audio['emo_vec'])
        # audio_reconstructed, emo_c, ct_emb, spk_emb, emo_emb = self.Anet(audio['mfcc'], audio['emo_vec'], audio['emb_spk'])
        # lmrk_reconstructed = self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        # lmrk_reconstructed =  self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        lmrk_ct = self.lmrk_decoder_ct(lmrk_emb, ct_emb, audio['emb_spk'], text_features)
        # lmrk_no_ct = self.lmrk_decoder_no_ct(lmrk_emb, ct_emb, audio['emb_spk'], emo_emb)
        lmrk_reconstructed = lmrk_ct  # + lmrk_no_ct
        return lmrk_reconstructed + lmrks, ct_emb, lmrk_emb

    # def compute_acc(self, input_label, out):
    #     _, pred = out.topk(1, 1)
    #     pred0 = pred.squeeze().data
    #     acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
    #     return acc

    def cross(self, x, audio, emo_vec, spk_emb, adj, s, text_features, device, train, lmrks_in):
        self.lmrk_net.eval()
        # lmrk_emb, adj, l1, e1, pos = self.lmrk_net(x, audio['adj'], audio['s'], device, train=train)
        lmrk_emb = self.lmrk_net(x, adj, s, device, train=train)
        ct_emb = self.audio_fc(audio)
        # emo_emb = self.emo_enc(emo_vec)
        # audio_reconstructed, emo_c, ct_emb, spk_emb, emo_emb = self.Anet(audio, emo_vec, spk_emb)
        # lmrk_reconstructed = self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        # lmrk_reconstructed =  self.lmrk_decoder(lmrk_emb, ct_emb, spk_emb, emo_emb)
        lmrk_ct = self.lmrk_decoder_ct(lmrk_emb, ct_emb, spk_emb, text_features)
        # lmrk_no_ct = self.lmrk_decoder_no_ct(lmrk_emb, ct_emb, spk_emb, emo_emb)
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

        # if train:
        #
        #     self.lmrk_D.zero_grad()
        #     output = self.lmrk_D(y, adj, s, self.device, train)
        #     err_real = 0.5 * torch.mean((output-label)**2)
        #     err_real.backward(retain_graph=True)
        #
        #     label.fill_(fake_label)
        #     new_lmrks = displacement + lmrks_in
        #     y.x = torch.reshape(new_lmrks, (y.num_graphs * 68, 2)).to(torch.float64)
        #     output = self.lmrk_D(y,adj, s, self.device, train)
        #     err_fake = 0.5 * torch.mean((output-label)**2)
        #     err_fake.backward(retain_graph=True)
        #
        #     self.optimizerD.step()
        #
        #     #lips_loss = self.l1loss(displacement[:, :, 48:68, :] + lmrks_in[:, :, 48:68, :], lmrks_out[:, :, 48:68, :])
        # label.fill_(real_label)  # fake labels are real for generator cost
        #     # Since we just updated D, perform another forward pass of all-fake batch through D
        # output = self.lmrk_D(y,adj, s, self.device, train).view(-1)
        #
        #     # Calculate G's loss based on this output
        # losses['errG'] = 0.25 * torch.mean((output - label)**2)

        # jaw_loss = self.l1loss(displacement[:, :, 3:14, :] + lmrks_in[:, :, 3:14, :], lmrks_out[:, :, 3:14, :])
        # losses['MSE'] = 0.5 * loss_custom(lmrks_out, displacement + lmrks_in, s,self.device)
        #losses['l1'] = self.MSE(displacement + lmrks_in, lmrks_out)
        losses['l1'] = self.MSE(displacement[:,:,17:48,:] + lmrks_in[:,:,17:48,:], lmrks_out[:,:,17:48,:])
        losses['l1_mouth'] = self.MSE(displacement[:,:,48:68,:] + lmrks_in[:,:,48:68,:], lmrks_out[:,:,48:68,:]) + self.MSE(displacement[:,:,0:17,:] + lmrks_in[:,:,0:17,:], lmrks_out[:,:,0:17,:])
        # losses['pre'] = 0.5 * loss_pre(out['pre'], displacement + lmrks_in)

        # outputs_dict = {
        #     "out_1": out1,
        #     "out_2": out2,
        # }
        return displacement + lmrks_in, losses

    def update_network(self, loss_dcit):
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        # self.optimizerD.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.optimizerD.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data, data_out, out, audio):
        #self.lmrk_encoder.train()
        self.lmrk_decoder_ct.train()
        # self.lmrk_decoder_no_ct.train()
        # self.lmrk_D.train()
        # self.emo_enc.train()
        self.audio_fc.train()
        outputs, losses = self.process(data, data_out, out, audio, train=True)
        self.update_network(losses)

        return outputs, losses

    def val_func(self, data, data_out, out, audio):
        #self.lmrk_encoder.eval()
        self.lmrk_decoder_ct.eval()
        # self.lmrk_decoder_no_ct.eval()
        # self.lmrk_D.eval()
        # self.emo_enc.eval()
        self.audio_fc.eval()
        # self.lmrk_D.eval()

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
        # self.out = nn.Sequential(
        #     nn.Linear((self.enc_hid_dim * 2) + self.dec_hid_dim + self.ldmk_dim, out_features=512),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(256, 136),
        # )

    def forward(self, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [ batch size, src len, enc hid dim * 2]
        # input = [batch, 1, ldmk_dim]

        #         print(hidden.shape)
        a = self.attention(encoder_outputs)
        # a = [batch size, src len]

        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]
        # encoder_outputs = [ batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)  # a.shape=[batch_size,1,5] encoder_outputs.shape=[batch_size,5,1024]
        # weighted = [batch size, 1, enc hid dim * 2]

        #rnn_input = torch.cat((input, weighted), dim=2)
        rnn_input = weighted
        # rnn_input = [ batch size,1, (enc hid dim * 2) + ldmk_dim]

        output, hidden = self.rnn(rnn_input)  # output.shape=[batch_size,1,512]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [batch size, 1, dec hid dim]
        # hidden = [1, batch size, dec hid dim]

        # this also means that output.permute(0,1, 2) == hidden

        assert (output.transpose(0, 1) == hidden).all()

        prediction = self.fc_out(torch.cat((output, weighted), dim=2))  # 这里线性层的多少有影响吗
        # prediction = self.out(torch.cat((output, weighted, input), dim=2))
        # prediction = [batch size, 1, ldmk_dim]

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


    # def save_fig(self, data, outputs, save_path):
    #     #    output1 = outputs['output1']
    #     #    output2 = outputs['output2']
    #     #    output12 = outputs['output12']
    #     #    output21 = outputs['output21']
    #
    #     #    target1 = data['target11']
    #     #    target2 = data['target22']
    #     #    target12 = data['target12']
    #     #    target21 = data['target21']
    #
    #     a = ['out_1', 'out_2']
    #     b = ['audio_1', 'audio_2']
    #
    #     for j in range(len(a)):
    #         output = outputs[a[j]]
    #         target = data[b[j]]
    #
    #         for i in range(3):
    #             g = target[i, :, :, :].squeeze()
    #             g = g.cpu().numpy()
    #
    #             o = output[i, :, :, :].squeeze()
    #             o = o.cpu().detach().numpy()
    #
    #             go = np.concatenate((g, o), axis=0, out=None, dtype=None, casting="same_kind")
    #
    #             # plt.figure()
    #             ax = sns.heatmap(go, vmin=-100, vmax=100, cmap='rocket_r')  # frames
    #             plt.savefig(os.path.join(save_path + '_' + 'out_' + str(j) + '_' + str(i) + '.png'))
    #             plt.close()
    # #      plt.show()


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
    """
    Slices out the upper triangular part of an adjacency matrix for
    a single graph from a large adjacency matrix for a full batch.
    For the node features the corresponding section in the batch is sliced out.
    ----------
    graph_id: The ID of the graph (in the batch index) to slice
    edge_targets: A dense adjacency matrix for the whole batch
    node_targets: A tensor of node labels for the whole batch
    batch_index: The node to graph map for the batch
    """
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
    """
    Slices out the corresponding section from a list of batch triu values.
    Given a start point and the size of a graph's triu, simply slices
    the section from the batch list.
    -------
    triu_logits: A batch of triu predictions of different graphs
    node_logits: A batch of node predictions with fixed size MAX_GRAPH_SIZE
    graph_triu_size: Size of the triu of the graph to slice
    triu_start_point: Index of the first node of this graph in the triu batch
    graph_size: Max graph size
    node_start_point: Index of the first node of this graph in the nodes batch
    """
    #Slice edge logits
    graph_logits_triu = torch.squeeze(triu_logits[triu_start_point:triu_start_point + graph_triu_size])
    #Slice node logits
    graph_node_logits = torch.squeeze(node_logits[node_start_point:node_start_point + graph_size])
    return graph_logits_triu, graph_node_logits


# def gvae_loss(triu_logits, node_logits, edge_index, edge_types, node_types, mu, logvar, batch_index, kl_beta):
#     """
#     Calculates the loss for the graph variational autoencoder,
#     consiting of a node loss, an edge loss and the KL divergence.
#     """
#     # Convert target edge index to dense adjacency matrix
#     batch_edge_targets = torch.squeeze(to_dense_adj(edge_index))
#
#     # Add edge types to adjacency targets
#     batch_edge_targets[edge_index[0], edge_index[1]] = edge_types[:, 1].float()
#
#     # For this model we always have the same (fixed) output dimension
#     graph_size = MAX_MOLECULE_SIZE * (len(SUPPORTED_ATOMS) + 1)
#     graph_triu_size = int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1)) / 2) * (len(SUPPORTED_EDGES) + 1)
#
#     # Reconstruction loss per graph
#     batch_recon_loss = []
#     triu_indices_counter = 0
#     graph_size_counter = 0
#
#     # Loop over graphs in this batch
#     for graph_id in torch.unique(batch_index):
#         # Get upper triangular targets for this graph from the whole batch
#         triu_targets, node_targets = slice_graph_targets(graph_id,
#                                                          batch_edge_targets,
#                                                          node_types,
#                                                          batch_index)
#
#         # Get upper triangular predictions for this graph from the whole batch
#         triu_preds, node_preds = slice_graph_predictions(triu_logits,
#                                                          node_logits,
#                                                          graph_triu_size,
#                                                          triu_indices_counter,
#                                                          graph_size,
#                                                          graph_size_counter)
#
#         # Update counter to the index of the next (upper-triu) graph
#         triu_indices_counter = triu_indices_counter + graph_triu_size
#         graph_size_counter = graph_size_counter + graph_size
#
#         # Calculate losses
#         recon_loss = approximate_recon_loss(node_targets,
#                                             node_preds,
#                                             triu_targets,
#                                             triu_preds)
#         batch_recon_loss.append(recon_loss)
#
#         # Take average of all losses
#     num_graphs = torch.unique(batch_index).shape[0]
#     batch_recon_loss = torch.true_divide(sum(batch_recon_loss), num_graphs)
#
#     # KL Divergence
#     kl_divergence = kl_loss(mu, logvar)
#
#     return batch_recon_loss + kl_beta * kl_divergence, kl_divergence
#
#
# def squared_difference(input, target):
#     return (input - target) ** 2
#
#
# def triu_to_dense(triu_values, num_nodes):
#     """
#     Converts a triangular upper part of a matrix as flat vector
#     to a squared adjacency matrix with a specific size (num_nodes).
#     """
#     dense_adj = torch.zeros((num_nodes, num_nodes)).to(device).float()
#     triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
#     tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
#     dense_adj[triu_indices[0], triu_indices[1]] = triu_values
#     dense_adj[tril_indices[0], tril_indices[1]] = triu_values
#     return dense_adj
#
#
# def triu_to_3d_dense(triu_values, num_nodes, depth=len(SUPPORTED_EDGES)):
#     """
#     Converts the triangular upper part of a matrix
#     for several dimensions into a 3d tensor.
#     """
#     # Create placeholder for 3d matrix
#     adj_matrix_3d = torch.empty((num_nodes, num_nodes, depth), dtype=torch.float, device=device)
#     for edge_type in range(len(SUPPORTED_EDGES)):
#         adj_mat_edge_type = triu_to_dense(triu_values[:, edge_type].float(), num_nodes)
#         adj_matrix_3d[:, :, edge_type] = adj_mat_edge_type
#     return adj_matrix_3d
#
#
# def calculate_node_edge_pair_loss(node_tar, edge_tar, node_pred, edge_pred):
#     """
#     Calculates a loss based on the sum of node-edge pairs.
#     node_tar:  [nodes, supported atoms]
#     node_pred: [max nodes, supported atoms + 1]
#     edge_tar:  [triu values for target nodes, supported edges]
#     edge_pred: [triu values for predicted nodes, supported edges]
#     """
#     # Recover full 3d adjacency matrix for edge predictions
#     edge_pred_3d = triu_to_3d_dense(edge_pred, node_pred.shape[0])  # [num nodes, num nodes, edge types]
#
#     # Recover full 3d adjacency matrix for edge targets
#     edge_tar_3d = triu_to_3d_dense(edge_tar, node_tar.shape[0])  # [num nodes, num nodes, edge types]
#
#     # --- The two output matrices tell us how many edges are connected with each of the atom types
#     # Multiply each of the edge types with the atom types for the predictions
#     node_edge_preds = torch.empty((MAX_MOLECULE_SIZE, len(SUPPORTED_ATOMS), len(SUPPORTED_EDGES)), dtype=torch.float,
#                                   device=device)
#     for edge in range(len(SUPPORTED_EDGES)):
#         node_edge_preds[:, :, edge] = torch.matmul(edge_pred_3d[:, :, edge], node_pred[:, :9])
#
#     # Multiply each of the edge types with the atom types for the targets
#     node_edge_tar = torch.empty((node_tar.shape[0], len(SUPPORTED_ATOMS), len(SUPPORTED_EDGES)), dtype=torch.float,
#                                 device=device)
#     for edge in range(len(SUPPORTED_EDGES)):
#         node_edge_tar[:, :, edge] = torch.matmul(edge_tar_3d[:, :, edge], node_tar.float())
#
#     # Reduce to matrix with [num atom types, num edge types]
#     node_edge_pred_matrix = torch.sum(node_edge_preds, dim=0)
#     node_edge_tar_matrix = torch.sum(node_edge_tar, dim=0)
#
#     if torch.equal(node_edge_pred_matrix.int(), node_edge_tar_matrix.int()):
#         print("Reconstructed node-edge pairs: ", node_edge_pred_matrix.int())
#
#     node_edge_loss = torch.mean(sum(squared_difference(node_edge_pred_matrix, node_edge_tar_matrix.float())))
#
#     # Calculate node-edge-node for preds
#     node_edge_node_preds = torch.empty((MAX_MOLECULE_SIZE, MAX_MOLECULE_SIZE, len(SUPPORTED_EDGES)), dtype=torch.float,
#                                        device=device)
#     for edge in range(len(SUPPORTED_EDGES)):
#         node_edge_node_preds[:, :, edge] = torch.matmul(node_edge_preds[:, :, edge], node_pred[:, :9].t())
#
#     # Calculate node-edge-node for targets
#     node_edge_node_tar = torch.empty((node_tar.shape[0], node_tar.shape[0], len(SUPPORTED_EDGES)), dtype=torch.float,
#                                      device=device)
#     for edge in range(len(SUPPORTED_EDGES)):
#         node_edge_node_tar[:, :, edge] = torch.matmul(node_edge_tar[:, :, edge], node_tar.float().t())
#
#     # Node edge node loss
#     node_edge_node_loss = sum(squared_difference(torch.sum(node_edge_node_preds, [0, 1]),
#                                                  torch.sum(node_edge_node_tar, [0, 1])))
#
#     # TODO: Improve loss
#     return node_edge_loss  # * node_edge_node_loss
#
# def to_one_hot(x, options):
#     """
#     Converts a tensor of values to a one-hot vector
#     based on the entries in options.
#     """
#     return torch.nn.functional.one_hot(x.long(), len(options))
#
# def approximate_recon_loss(node_targets, node_preds, triu_targets, triu_preds):
#     """
#     TODO: Improve loss function
#     """
#     # Convert targets to one hot
#     onehot_node_targets = to_one_hot(node_targets, SUPPORTED_ATOMS)  # + ["None"]
#     onehot_triu_targets = to_one_hot(triu_targets, ["None"] + SUPPORTED_EDGES)
#
#     # Reshape node predictions
#     node_matrix_shape = (MAX_MOLECULE_SIZE, (len(SUPPORTED_ATOMS) + 1))
#     node_preds_matrix = node_preds.reshape(node_matrix_shape)
#
#     # Reshape triu predictions
#     edge_matrix_shape = (int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1)) / 2), len(SUPPORTED_EDGES) + 1)
#     triu_preds_matrix = triu_preds.reshape(edge_matrix_shape)
#
#     # Apply sum on labels per (node/edge) type and discard "none" types
#     node_preds_reduced = torch.sum(node_preds_matrix[:, :9], 0)
#     node_targets_reduced = torch.sum(onehot_node_targets, 0)
#     triu_preds_reduced = torch.sum(triu_preds_matrix[:, 1:], 0)
#     triu_targets_reduced = torch.sum(onehot_triu_targets[:, 1:], 0)
#
#     # Calculate node-sum loss and edge-sum loss
#     node_loss = sum(squared_difference(node_preds_reduced, node_targets_reduced.float()))
#     edge_loss = sum(squared_difference(triu_preds_reduced, triu_targets_reduced.float()))
#
#     # Calculate node-edge-sum loss
#     # Forces the model to properly arrange the matrices
#     node_edge_loss = calculate_node_edge_pair_loss(onehot_node_targets,
#                                                    onehot_triu_targets,
#                                                    node_preds_matrix,
#                                                    triu_preds_matrix)
#
#     approx_loss = node_loss + edge_loss + node_edge_loss
#
#     if all(node_targets_reduced == node_preds_reduced.int()) and \
#             all(triu_targets_reduced == triu_preds_reduced.int()):
#         print("Reconstructed all edges: ", node_targets_reduced)
#         print("and all nodes: ", node_targets_reduced)
#     return approx_loss
