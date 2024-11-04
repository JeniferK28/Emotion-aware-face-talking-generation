import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from lmrk2img.model_img_trans import ResUnetGenerator, VGGLoss
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
import numpy as np
import cv2
import os, glob
from utils import vis_landmark_on_img


class Image_translation_block():

    def __init__(self, opt_parser, single_test=False):
        print('Run on device {}'.format(device))

        # for key in vars(opt_parser).keys():
        #     print(key, ':', vars(opt_parser)[key])
        self.opt_parser = opt_parser

        # model
        if(opt_parser.add_audio_in):
            self.G = ResUnetGenerator(input_nc=7, output_nc=3, num_downs=6, use_dropout=False)
        else:
            self.G = ResUnetGenerator(input_nc=6, output_nc=3, num_downs=6, use_dropout=False)

        if (opt_parser.load_G_name != ''):
            ckpt = torch.load(opt_parser.load_G_name)
            try:
                self.G.load_state_dict(ckpt['G'])
            except:
                tmp = nn.DataParallel(self.G)
                tmp.load_state_dict(ckpt['G'])
                self.G.load_state_dict(tmp.module.state_dict())
                del tmp

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs in G mode!")
            self.G = nn.DataParallel(self.G)

        self.G.to(device)

        if(not single_test):
            # criterion
            self.criterionL1 = nn.L1Loss()
            self.criterionVGG = VGGLoss()
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs in VGG model!")
                self.criterionVGG = nn.DataParallel(self.criterionVGG)
            self.criterionVGG.to(device)

            # optimizer
            self.optimizer = torch.optim.Adam(self.G.parameters(), lr=opt_parser.lr, betas=(0.5, 0.999))

            # writer
            if(opt_parser.write):
                self.writer = SummaryWriter(log_dir=os.path.join(opt_parser.log_dir, opt_parser.name))
                self.count = 0

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def __train_pass__(self, epoch, dataloader, is_training=True):
        st_epoch = time.time()
        if(is_training):
            self.G.train()
            status = 'TRAIN'
        else:
            self.G.eval()
            status = 'EVAL'

        g_time = 0.0

        for i, (data,out) in enumerate(dataloader):
            if(i >= len(dataloader)-2):
                break
            st_batch = time.time()
            pred_lmrks = data['lmrks']
            pred_lmrks = pred_lmrks.reshape(-1, 68, 3).detach().cpu().numpy()
            image_in = data['img_ref']
            image_out = out['img']
            input = torch.cat([image_in, image_in], dim=3)
            for i in range(self.opt_parser.batch_size):
                img_fl = np.ones(shape=(256, 256, 3), dtype=np.uint8) * 255
                fl = pred_lmrks[i].astype(int)

                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
                #img_fl.transpose((2, 0, 1)
                img_fl = np.stack(img_fl, axis=0).astype(np.float32) / 255.0
                image_fls_in = torch.tensor(img_fl, requires_grad=False).to(device)

                input[i] = torch.cat([image_fls_in, image_in[i]], dim=2)

            input = input.reshape(-1, 6, 256, 256).to(device)
            image_out = image_out.reshape(-1, 3, 256, 256).to(device)
            # image_in, image_out = \
            #     image_in.reshape(-1, 6, 256, 256).to(device), image_out.reshape(-1, 3, 256, 256).to(device)

            # image2image net fp
            g_out = self.G(input)
            g_out = torch.tanh(g_out)

            loss_l1 = self.criterionL1(g_out, image_out)
            loss_vgg, loss_style = self.criterionVGG(g_out, image_out, style=True)

            loss_vgg, loss_style = torch.mean(loss_vgg), torch.mean(loss_style)

            loss = loss_l1  + loss_vgg + loss_style
            if(is_training):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # log
            if(self.opt_parser.write):
                self.writer.add_scalar('Train/loss', loss.cpu().detach().numpy(), self.count)
                self.writer.add_scalar('Train/loss_l1', loss_l1.cpu().detach().numpy(), self.count)
                self.writer.add_scalar('Train/loss_vgg', loss_vgg.cpu().detach().numpy(), self.count)
                self.count += 1

            # save ckpt
            if (i % 10 == 0):
                print("Epoch {}, Batch {}/{}, loss {:.4f}, l1 {:.4f}, vggloss {:.4f}, styleloss {:.4f} time {:.4f}".format(
                    epoch, i, len(self.dataset) // self.opt_parser.batch_size,
                    loss.cpu().detach().numpy(),
                    loss_l1.cpu().detach().numpy(),
                    loss_vgg.cpu().detach().numpy(),
                    loss_style.cpu().detach().numpy(),
                              time.time() - st_batch))

            # save image to track training process
            if (i % 50 == 0):
                vis_in = np.concatenate([input[0, 3:6].cpu().detach().numpy().transpose((1, 2, 0)),
                                         input[0, 0:3].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                vis_out = np.concatenate([image_out[0].cpu().detach().numpy().transpose((1, 2, 0)),
                                          g_out[0].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                vis = np.concatenate([vis_in, vis_out], axis=0)
                try:
                    os.makedirs(os.path.join(self.opt_parser.jpg_dir, self.opt_parser.name))
                except:
                    pass
                cv2.imwrite(os.path.join(self.opt_parser.jpg_dir, self.opt_parser.name, 'e{:03d}_b{:04d}.jpg'.format(epoch, i)), vis * 255.0)

            g_time += time.time() - st_batch

            if(self.opt_parser.test_speed):
                if(i >= 100):
                    break

        #print('Epoch time usage:', time.time() - st_epoch, 'I/O time usage:', time.time() - st_epoch - g_time, '\n=========================')
        self.__save_model__('{:02d}'.format(epoch), epoch)


    def __save_model__(self, save_type, epoch):
        try:
            os.makedirs(os.path.join(self.opt_parser.ckpt_dir, self.opt_parser.name))
        except:
            pass
        if (self.opt_parser.write):
            torch.save({'G': self.G.state_dict(),'opt': self.optimizer, 'epoch': epoch
        }, os.path.join(self.opt_parser.ckpt_dir, self.opt_parser.name, 'ckpt_{}.pth'.format(save_type)))

    def train(self, dataloader, epoch):
        self.__train_pass__(epoch, dataloader, is_training=True)

    def test(self, dataloader, epoch):

        self.G.eval()
        for i, (data,out) in enumerate(dataloader):
            print(i, 50)
            if (i > 50):
                break

            st_batch = time.time()
            pred_lmrks = data['lmrks']
            pred_lmrks = pred_lmrks.reshape(-1, 68, 3).detach().cpu().numpy()
            image_in = data['img_ref']
            image_out = out['img']
            input = torch.cat([image_in, image_in], dim=3)

            for i in range(self.opt_parser.batch_size):
                img_fl = np.ones(shape=(256, 256, 3), dtype=np.uint8) * 255
                fl = pred_lmrks[i].astype(int)

                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
                # img_fl.transpose((2, 0, 1)
                img_fl = np.stack(img_fl, axis=0).astype(np.float32) / 255.0
                image_fls_in = torch.tensor(img_fl, requires_grad=False).to(device)

                input[i] = torch.cat([image_fls_in, image_in[i]], dim=2)

            input = input.reshape(-1, 6, 256, 256).to(device)
            image_out = image_out.reshape(-1, 3, 256, 256).to(device)

            # image_in, image_out = \
            #     image_in.reshape(-1, 6, 256, 256).to(device), image_out.reshape(-1, 3, 256, 256).to(device)

            # image2image net fp
            g_out = self.G(input)
            g_out = torch.tanh(g_out)

            loss_l1 = self.criterionL1(g_out, image_out)
            loss_vgg, loss_style = self.criterionVGG(g_out, image_out, style=True)

            loss_vgg, loss_style = torch.mean(loss_vgg), torch.mean(loss_style)

            loss = loss_l1 + loss_vgg + loss_style

            if (self.opt_parser.write):
                self.writer.add_scalar('Val/loss', loss.cpu().detach().numpy(), self.count)
                self.writer.add_scalar('Val/loss_l1', loss_l1.cpu().detach().numpy(), self.count)
                self.writer.add_scalar('Val/loss_vgg', loss_vgg.cpu().detach().numpy(), self.count)
                self.count += 1

                # save ckpt
            if (i % 100 == 0):
                print(
                    "Epoch {}, Batch {}/{}, loss {:.4f}, l1 {:.4f}, vggloss {:.4f}, styleloss {:.4f} time {:.4f}".format(
                         epoch, i, len(self.dataset) // self.opt_parser.batch_size,
                        loss.cpu().detach().numpy(),
                        loss_l1.cpu().detach().numpy(),
                        loss_vgg.cpu().detach().numpy(),
                        loss_style.cpu().detach().numpy(),
                                  time.time() - st_batch))

                # save image to track training process
            if (i % 500 == 0):
                vis_in = np.concatenate([input[0, 3:6].cpu().detach().numpy().transpose((1, 2, 0)),
                                         input[0, 0:3].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                vis_out = np.concatenate([image_out[0].cpu().detach().numpy().transpose((1, 2, 0)),
                                          g_out[0].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                vis = np.concatenate([vis_in, vis_out], axis=0)
                try:
                    os.makedirs(os.path.join(self.opt_parser.jpg_dir, self.opt_parser.name))
                except:
                    pass
                cv2.imwrite(
                    os.path.join(self.opt_parser.jpg_dir, self.opt_parser.name, 'e{:03d}_b{:04d}.jpg'.format(epoch, i)),
                    vis * 255.0)

    def single_test(self, jpg=None, fls=None, filename=None, prefix='', grey_only=False):
        import time
        st = time.time()
        self.G.eval()

        if(jpg is None):
            jpg = glob.glob1(self.opt_parser.single_test, '*.jpg')[0]
            jpg = cv2.imread(os.path.join(self.opt_parser.single_test, jpg))

        if(fls is None):
            fls = glob.glob1(self.opt_parser.single_test, '*.txt')[0]
            fls = np.loadtxt(os.path.join(self.opt_parser.single_test, fls))
            fls = fls * 95
            fls[:, 0::3] += 130
            fls[:, 1::3] += 80

        writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), 62.5, (256 * 3, 256))

        for i, frame in enumerate(fls):

            img_fl = np.ones(shape=(256, 256, 3)) * 255
            fl = frame.astype(int)
            img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
            frame = np.concatenate((img_fl, jpg), axis=2).astype(np.float32)/255.0

            image_in, image_out = frame.transpose((2, 0, 1)), np.zeros(shape=(3, 256, 256))
            # image_in, image_out = frame.transpose((2, 1, 0)), np.zeros(shape=(3, 256, 256))
            image_in, image_out = torch.tensor(image_in, requires_grad=False), \
                                  torch.tensor(image_out, requires_grad=False)

            image_in, image_out = image_in.reshape(-1, 6, 256, 256), image_out.reshape(-1, 3, 256, 256)
            image_in, image_out = image_in.to(device), image_out.to(device)

            g_out = self.G(image_in)
            g_out = torch.tanh(g_out)

            g_out = g_out.cpu().detach().numpy().transpose((0, 2, 3, 1))
            g_out[g_out < 0] = 0
            ref_in = image_in[:, 3:6, :, :].cpu().detach().numpy().transpose((0, 2, 3, 1))
            fls_in = image_in[:, 0:3, :, :].cpu().detach().numpy().transpose((0, 2, 3, 1))
            # g_out = g_out.cpu().detach().numpy().transpose((0, 3, 2, 1))
            # g_out[g_out < 0] = 0
            # ref_in = image_in[:, 3:6, :, :].cpu().detach().numpy().transpose((0, 3, 2, 1))
            # fls_in = image_in[:, 0:3, :, :].cpu().detach().numpy().transpose((0, 3, 2, 1))

            if(grey_only):
                g_out_grey =np.mean(g_out, axis=3, keepdims=True)
                g_out[:, :, :, 0:1] = g_out[:, :, :, 1:2] = g_out[:, :, :, 2:3] = g_out_grey


            for i in range(g_out.shape[0]):
                #frame = np.concatenate((ref_in[i], g_out[i], fls_in[i]), axis=1) * 255.0
                frame = g_out * 255.0
                writer.write(frame.astype(np.uint8))

        writer.release()
        print('Time - only video:', time.time() - st)

        if(filename is None):
            filename = 'v'
        os.system('ffmpeg -loglevel error -y -i out.mp4 -i {} -pix_fmt yuv420p -strict -2 examples/{}_{}.mp4'.format(
            'examples/'+filename[9:-16]+'.wav',
            prefix, filename[:-4]))
        # os.system('rm out.mp4')

        print('Time - ffmpeg add audio:', time.time() - st)