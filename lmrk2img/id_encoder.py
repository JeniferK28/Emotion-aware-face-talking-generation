import torch.nn as nn
import torch.nn.functional as F
from base_network import BaseNetwork
from torchvision.models.resnet import ResNet, Bottleneck
import utils
import torch
import argparse


def parser_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_classes', type=int, default=5830, help='num classes')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--num_inputs', type=int, default=1, help='num of inputs to the network')

        return parser.parse_args()

config = parser_args()
model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
}

class ResNeXt50(BaseNetwork):
    def __init__(self, opt):
        super(ResNeXt50, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
        self.opt = opt
        # self.reduced_id_dim = opt.reduced_id_dim
        self.conv1x1 = nn.Conv2d(512 * Bottleneck.expansion, 512, kernel_size=1, padding=0)
        self.fc = nn.Linear(512 * Bottleneck.expansion, opt.num_classes)
        #self.fc_pre = nn.Sequential(nn.Linear(512 * Bottleneck.expansion, self.reduced_id_dim), nn.ReLU())


    def load_pretrain(self):
        check_point = torch.load(model_urls['resnext50_32x4d'])
        utils.copy_state_dict(check_point, self.model)

    def forward_feature(self, input):
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        net = self.model.avgpool(x)
        net = torch.flatten(net, 1)
        x = self.conv1x1(x)
        # x = self.fc_pre(x)
        return net, x

    def forward(self, input):
        input_batch = input.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        net, x = self.forward_feature(input_batch)
        net = net.view(-1, self.opt.num_inputs, 512 * Bottleneck.expansion)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(-1, self.opt.num_inputs, 512, 7, 7)
        net = torch.mean(net, 1)
        x = torch.mean(x, 1)
        cls_scores = self.fc(net)

        return [net, x], cls_scores


class ResNeXtEncoder(ResNeXt50):
    def __init__(self):
        super(ResNeXtEncoder, self).__init__(config)
