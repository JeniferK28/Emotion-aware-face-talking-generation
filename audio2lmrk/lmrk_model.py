import torch.nn as nn
import torch
from torch_geometric.nn.conv import TransformerConv, GraphConv
from torch_geometric.nn import Set2Set, dense_diff_pool, knn_interpolate
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import to_dense_adj

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

    def forward(self, input, adj, s, device, train):
        x = self.conv1(input.x, input.edge_index).relu()
        x = self.conv2(x, input.edge_index).relu()
        x = self.conv3(x, input.edge_index).relu()

        x_resize = torch.reshape(x, (1, 68, 128)).to(torch.float32)

        # Pool to global representation
        x, adj, l1, e1 = dense_diff_pool(x_resize.to(device), adj, s)

        return x, adj, l1, e1, input.pos

class lmrk_fc(nn.Module):
    def __init__(self):
        super(lmrk_fc, self).__init__()
        self.lmrks_enc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ELU(True),
        )
        # self.att = Attention()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.lmrks_enc(x)
        # out = self.att(out)
        return out

class lmrk_decoder(nn.Module):
    def __init__(self):
        super(lmrk_decoder, self).__init__()
        # 1536, 1280, 768
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=True),  # 2,2
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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #features = torch.cat([x_fts, ct_emb, spk_emb, emo_vec], 1)  # connect tensors inputs and dimension
        features = torch.unsqueeze(x, 2)
        features = torch.unsqueeze(features, 3)
        x = self.deconv1(features.float())
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = 256* self.deconv4(x)  # [1, 1,x, y]
        return x

class lmrkNet(nn.Module):
    def __init__(self,config):
        super(lmrkNet,self).__init__()
        self.device = config.device
        self.encoder = lmrk_encoder_h()
        self.fc = lmrk_fc()
        self.decoder = lmrk_decoder()
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())
                                          + list(self.fc.parameters())
                                          + list(self.decoder.parameters()),
                                          config.lr, betas=(config.beta1, config.beta2))
        self.l1loss = nn.L1Loss()
        self.MSE = nn.MSELoss()

    def forward(self, x,  adj, s, device, train):
        feature, adj, l1, e1, pos = self.encoder(x, adj, s, device, train)
        return feature

    def cross(self, x, adj, s, device, train):
        lmrk_emb, adj, l1, e1, pos = self.encoder(x, adj, s, device, train=train)
        lmrk = self.decoder(lmrk_emb)
        return  lmrk, lmrk_emb

    def process(self, lmrk, data_in,  train):
        lmrks_in = data_in['in']
        adj = data_in['adj']
        s = data_in['s']
        losses = {}
        recons_lmrk, ct_emb = self.cross(lmrk, adj, s, self.device, train)
        losses['recons'] = self.MSE(recons_lmrk, lmrks_in.type(torch.cuda.FloatTensor))
        x = lmrk
        recons_lmrk = torch.squeeze(recons_lmrk,1)
        recons_lmrk = recons_lmrk.to(torch.float64)
        x.x = recons_lmrk.view(recons_lmrk.size(1) * recons_lmrk.size(0), 2)
        _, ct_emb_fake = self.cross(x, adj, s, self.device, train)
        losses['emb'] = self.l1loss(ct_emb_fake, ct_emb)

        return recons_lmrk,losses

    def update_network(self, loss_dict):

        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, lmrk, data):
        self.encoder.train()
        self.decoder.train()

        self.fc.train()
        outputs, losses = self.process(lmrk, data, train=True)
        self.update_network(losses)

        return outputs, losses

    def val_func(self, lmrk, data):
        self.encoder.eval()
        self.decoder.eval()
        self.fc.eval()
 
        with torch.no_grad():
            outputs, losses = self.process(lmrk, data, train=False)

        return outputs, losses
