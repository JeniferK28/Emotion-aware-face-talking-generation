import torch
import torch_geometric
import numpy as np
import torch.nn as nn
from audio.ANet import ANet
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv, GraphConv
from torch_geometric.nn import Set2Set, dense_diff_pool, knn_interpolate
from torch_geometric.nn import BatchNorm
from audio2lmrk.gcn_model_audio import a2lNet
#from gcn_model_audio import a2lNet
from torch_geometric.utils import to_dense_adj
from utils import loss_custom


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class lmrk_encoder(nn.Module):
    def __init__(self):
        super(lmrk_encoder, self).__init__()
        self.feature_size = 2
        self.encoder_embedding_size = 128

        self.conv1 = TransformerConv(self.feature_size,
                                     self.encoder_embedding_size,
                                     heads=4,
                                     concat=False,
                                     beta=True)
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.conv2 = TransformerConv(self.encoder_embedding_size,
                                     self.encoder_embedding_size,
                                     heads=4,
                                     concat=False,
                                     beta=True)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.conv3 = TransformerConv(self.encoder_embedding_size,
                                     self.encoder_embedding_size,
                                     heads=4,
                                     concat=False,
                                     beta=True)
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        self.conv4 = TransformerConv(self.encoder_embedding_size,
                                     self.encoder_embedding_size,
                                     heads=4,
                                     concat=False,
                                     beta=True)

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=2)

    def forward(self, input, edge_attr, batch_index):
        x = self.conv1(input.x, input.edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, x.edge_index).relu()
        x = self.bn2(x)
        x = self.conv3(x, x.edge_index).relu()
        # x = self.bn3(x)
        # x = self.conv4(x, edge_index).relu()

        # Pool to global representation
        x = self.pooling(x, batch_index)

        # Latent transform layers
        # mu = self.mu_transform(x)
        # logvar = self.logvar_transform(x)

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

    def forward(self, input, adj, s, device):
        x = self.conv1(input.x, input.edge_index).relu()
        # x = self.bn1(x)
        x = self.conv2(x, input.edge_index).relu()
        # x = self.bn2(x)
        x = self.conv3(x, input.edge_index).relu()
        # x_resize = torch.zeros((1, 68, 128))
        # c = 0
        # for i in range(1):
        #     x_resize[i, :, :] = x[68 * c:68 * (c + 1), :]
        #     c += 1
        x_resize = torch.reshape(x, (input.num_graphs, 68, 128)).to(torch.float32)
        #x_resize = torch.reshape(x, (1, 68, 128)).to(torch.float32)
        # x = self.bn3(x)
        # x = self.conv4(x, edge_index).relu()

        # Pool to global representation
        # x = self.pooling(x, batch_index)
        x, adj, l1, e1 = dense_diff_pool(x_resize.to(device), adj, s)
        # Latent transform layers
        # mu = self.mu_transform(x)
        # logvar = self.logvar_transform(x)

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
        self.maxpooling = nn.MaxPool2d((1, 8))
        self.fc = nn.Linear(128, 1)

    def forward(self, input, adj, s, device, train):
        x = self.conv1(input.x, input.edge_index).relu()
        # x = self.bn1(x)
        x = self.conv2(x, input.edge_index).relu()
        # x = self.bn2(x)
        x = self.conv3(x, input.edge_index).relu()
        # x_resize = torch.zeros((1,68,128))
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
        # self.fc = nn.Linear(1024, 128)
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
        # features = self.fc(features)
        x_fts = x.view(x.size(0), -1)
        features = torch.cat([x_fts, ct_emb, spk_emb, emo_vec], 1)
        features = torch.unsqueeze(features, 2)
        features = torch.unsqueeze(features, 3)
        # for i in range(8):
        #         x_fts[:,i,128:256] = features
        # knn_interpolate(x_fts, pos, pos_skip, batch, batch_skip, k=self.k)
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
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1536, 256, kernel_size=4, stride=2, padding=1, bias=True),  # 2,2
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

    def forward(self, x, ct_emb, spk_emb):
        x_fts = x.view(x.size(0), -1)
        features = torch.cat([x_fts, ct_emb, spk_emb], 1)  # connect tensors inputs and dimension
        features = torch.unsqueeze(features, 2)
        features = torch.unsqueeze(features, 3)
        x = self.deconv1(features)
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
        features = torch.cat([x_fts, ct_emb, spk_emb, emo_vec], 1)  # connect tensors inputs and dimension
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
            nn.Conv1d(1, 64, 5),
            nn.ELU(True),
            #nn.Conv1d(32, 64, 2),
        )

    def forward(self, emo):
        x = self.emotion_encoder(emo)
        x = x.view(x.size(0), -1)
        return x


class audio_fc(nn.Module):
    def __init__(self):
        super(audio_fc, self).__init__()
        self.audio_encoder = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(True),
            nn.Linear(512, 256),
            nn.ELU(True),
        )

    def forward(self, audio):
        audio = audio.view(audio.size(0), -1)
        out = self.audio_encoder(audio)
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


## lmrk emo classifier
class emo_class(nn.Module):
    def __init__(self):
        super(emo_class, self).__init__()
        self.emo_fc = nn.Sequential(
            nn.Linear(136, 64, dtype=torch.float64),
            nn.ELU(True),
            nn.Linear(64, 8, dtype=torch.float64),
            nn.ELU(True),
        )

    def forward(self, feature):
        feature = feature.view(feature.size(0), -1)
        x = self.emo_fc(feature)
        return x
class emo2lNet(nn.Module):
    def __init__(self, config):
        super(emo2lNet, self).__init__()
        self.device = config.device
        self.u = config
        self.cont_net = a2lNet(self.u)
        self.cont_net.load_state_dict(torch.load(config.audio_pretrain))
        #self.audio_fc = audio_conv()
        self.emo_enc = emotion()
        # self.lmrk_encoder =lmrk_encoder()
        #self.lmrk_encoder = lmrk_encoder_h()
        # self.classify = class_emo()
        # self.lmrk_decoder = lmrk_decoder_up()
        # self.lmrk_decoder = lmrk_decoder_fc()
        #self.lmrk_decoder_ct = lmrk_decoder_ct()
        self.lmrk_decoder_no_ct = lmrk_decoder_no_ct()
        self.emo_class = emo_class()
        #self.lmrk_D = lmrk_discriminator()
        #self.optimizerD = torch.optim.Adam(self.lmrk_D.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizer = torch.optim.Adam(#list(self.lmrk_encoder.parameters())
                                          list(self.emo_enc.parameters())
                                          + list(self.emo_class.parameters())
                                          #+ list(self.audio_fc.parameters())
                                          + list(self.lmrk_decoder_no_ct.parameters()),
                                          #+ list(self.lmrk_decoder_ct.parameters())
                                          config.lr, betas=(config.beta1, config.beta2))
        self.CroEn_loss = nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.weight = config.triplet_weight
        self.criterion = nn.BCELoss()

    def forward(self, x, audio, lmrks,  device):
        self.cont_net.eval()

        lmrk_ct, ct_emb, lmrk_emb = self.cont_net(x, audio, lmrks, device, train='False')
        emo_emb = self.emo_enc(audio['emo_vec'])
        lmrk_no_ct = self.lmrk_decoder_no_ct(lmrk_emb, ct_emb, audio['emb_spk'], emo_emb)
        lmrk_reconstructed = lmrk_no_ct + lmrk_ct
        return lmrk_reconstructed

    def cross(self, x, audio, emo_vec, lmrks, spk_emb,  device):
        self.cont_net.eval()
        lmrk_ct, ct_emb, lmrk_emb = self.cont_net(x, audio, lmrks, device, train= 'True')
        emo_emb = self.emo_enc(emo_vec)
        lmrk_no_ct = self.lmrk_decoder_no_ct(lmrk_emb, ct_emb, spk_emb, emo_emb)
        lmrk_reconstructed = lmrk_no_ct + lmrk_ct
        return lmrk_reconstructed

    def process(self, data_in, data_out, out, audio, train):
        #mfcc = audio['mfcc']
        spk_emb = audio['emb_spk']
        emo_vec = audio['emo_vec']
        emo_label = audio['emo_class']
        x = data_in
        y = data_out
        lmrks_out = out['out']
        lmrks_in = out['in']
        adj = audio['adj']
        s = audio['s']
        losses = {}

        displacement = self.cross(x, audio, emo_vec, lmrks_in, spk_emb,  self.device)
        emo = self.emo_class(displacement + lmrks_in)
        emo_label = emo_label.squeeze_().type(torch.LongTensor)
        losses['emo'] = 10* self.CroEn_loss(emo, emo_label.to(self.device))
        real_label = 1.
        fake_label = 0.
        b_size = audio['emb_spk'].size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)

        # if train:
        #     self.lmrk_D.zero_grad()
        #     output = self.lmrk_D(y, adj, s, self.device, train)
        #     err_real = 0.5 * torch.mean((output - label) ** 2)
        #     err_real.backward(retain_graph=True)
        #
        #     label.fill_(fake_label)
        #     new_lmrks = displacement
        #     y.x = torch.reshape(new_lmrks, (y.num_graphs * 68, 2)).to(torch.float64)
        #     output = self.lmrk_D(y, adj, s, self.device, train)
        #     err_fake = 0.5 * torch.mean((output - label) ** 2)
        #     err_fake.backward(retain_graph=True)

        #     self.optimizerD.step()
        #    # lips_loss = self.l1loss(displacement[:, :, 48:68, :] + lmrks_in[:, :, 48:68, :], lmrks_out[:, :, 48:68, :])
        # label.fill_(real_label)  # fake labels are real for generator cost
        # #Since we just updated D, perform another forward pass of all-fake batch through D
        # output = self.lmrk_D(y, adj, s, self.device, train).view(-1)
        #
        # #Calculate G's loss based on this output
        # losses['errG'] = 0.25 * torch.mean((output - label) ** 2)

        # jaw_loss = self.l1loss(displacement[:, :, 3:14, :] + lmrks_in[:, :, 3:14, :], lmrks_out[:, :, 3:14, :])
        losses['MSE'] = 0.5*loss_custom(lmrks_out, displacement + lmrks_in, s,self.device)
        losses['l1'] = self.MSE(displacement, lmrks_out)

        # outputs_dict = {
        #     "out_1": out1,
        #     "out_2": out2,
        # }
        return displacement, losses

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

        self.lmrk_decoder_no_ct.train()
        self.emo_class.train()
        #self.lmrk_D.train()
        self.emo_enc.train()

        outputs, losses = self.process(data, data_out, out, audio, train=True)
        self.update_network(losses)

        return outputs, losses

    def val_func(self, data, data_out, out, audio):

        self.lmrk_decoder_no_ct.eval()
        #self.lmrk_D.eval()
        self.emo_class.eval()
        self.emo_enc.eval()

        with torch.no_grad():
            outputs, losses = self.process(data, data_out, out, audio, train=False)
        return outputs, losses


