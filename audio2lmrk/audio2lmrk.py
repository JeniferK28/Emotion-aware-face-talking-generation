import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from lmrk2img.model_img_trans import ResUnetGenerator, VGGLoss
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
import numpy as np
import cv2
import os, glob
from utils import vis_landmark_on_img
import matplotlib.pyplot as plt
import tensorboardX

def draw_lmrk(data, out, output, batch_size):
    input = torch.zeros((1, 256, 256, 9))
    j = np.random.choice(batch_size,1)
    img_fl = np.ones(shape=(256, 256, 3), dtype=np.uint8) * 255
    fl = data[j].squeeze().cpu().detach().numpy().astype(int)
    img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 2)))
    # img_fl.transpose((2, 0, 1)
    img_fl = np.stack(img_fl, axis=0).astype(np.float32) / 255.0
    image_fls_in = torch.tensor(img_fl, requires_grad=False).to(device)

    img_fl_out = np.ones(shape=(256, 256, 3), dtype=np.uint8) * 255
    fl_out = out[j].squeeze().cpu().detach().numpy().astype(int)
    img_fl_out = vis_landmark_on_img(img_fl_out, np.reshape(fl_out, (68, 2)))
    # img_fl.transpose((2, 0, 1)
    img_fl_out = np.stack(img_fl_out, axis=0).astype(np.float32) / 255.0
    image_fls_out = torch.tensor(img_fl_out, requires_grad=False).to(device)

    img_fl_output = np.ones(shape=(256, 256, 3), dtype=np.uint8) * 255
    fl_output = output[j].squeeze().cpu().detach().numpy().astype(int)
    img_fl_output = vis_landmark_on_img(img_fl_output, np.reshape(fl_output, (68, 2)))
    # img_fl.transpose((2, 0, 1)
    img_fl_output = np.stack(img_fl_output, axis=0).astype(np.float32) / 255.0
    image_fls_output = torch.tensor(img_fl_output, requires_grad=False).to(device)

    input[0] = torch.cat([image_fls_in, image_fls_out, image_fls_output], dim=2)
    input = input.permute(0, 3, 1, 2).to(device)

    return input

def train(config, train_loader, test_loader, model):
    run_detail = config.name
    writer = tensorboardX.SummaryWriter(comment=run_detail)  ###########################################
    total_steps = 0

    train_iter = 0
    val_iter = 0
    a = time.time()

    for epoch in range(config.start_epoch, config.max_epochs):
        epoch_start_time = time.time()
        batch_size = config.batch_size
        for i, (data_in, data_out, lmrks, audio) in enumerate(train_loader):
            iter_start_time = time.time()


            #   data = Variable(data.float().cuda())
            outputs, losses = model.train_func(data_in, data_out, lmrks, audio)
            #losses_values = {k: v.item() for k, v in losses.items()}
            for k, v in losses.items():
                writer.add_scalar('Train/' + k + '_v', v, train_iter)

            loss = sum(losses.values())
            writer.add_scalar('Loss/Train', loss, train_iter)

            if (train_iter % 100 == 0):
                # for k, v in losses_values.items():
                #    print(k,v)

                print('[%d,%5d / %d] Rec loss :%.10f, lips loss : %.10f, time spent: %f s' % (
                    epoch + 1, i + 1, len(train_loader), loss, losses['l1'].item(), time.time() - a))

            if (train_iter % 100 == 0):  # 500
                with open(config.log_dir + 'train.txt', 'a') as file_handle:
                    file_handle.write(
                        '[%d,%5d / %d] Rec loss :%.10f,  time spent: %f s' % (
                            epoch + 1, i + 1, len(train_loader), losses['l1'].item(), time.time() - a))
                    file_handle.write('\n')

            # save image to track training process
            if (i % 500 == 0):
                input = draw_lmrk(lmrks['in'],lmrks['out'], outputs,batch_size)
                vis_in = np.concatenate([input[0, 0:3].cpu().detach().numpy().transpose((1, 2, 0)),
                                         input[0, 3:6].cpu().detach().numpy().transpose((1, 2, 0)),
                                        input[0, 6:9].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                try:
                    os.makedirs(os.path.join(config.lmrk_dir, config.name))
                except:
                    pass
                cv2.imwrite(
                    os.path.join(config.lmrk_dir, config.name,
                                 'e{:03d}_b{:04d}.jpg'.format(epoch, i)),
                    cv2.cvtColor(vis_in * 255.0, cv2.COLOR_RGB2BGR))

            train_iter += 1


        string_fts = os.path.join(config.model_dir, 'audio2lmrk_' + run_detail + '_' + str(epoch) + '.pt')
        # model.eval()
        torch.save(model.state_dict(), string_fts)

        print("start to validate, epoch %d" % (epoch + 1))


        with torch.no_grad():
            for i, (data, data_out, lmrks, audio) in enumerate(test_loader):
                if (i > 10):
                    break
                outputs, losses = model.val_func(data, data_out, lmrks, audio)

                for k, v in losses.items():
                    writer.add_scalar('Val/' + k + '_v', v, val_iter)
                loss = sum(losses.values())
                writer.add_scalar('Loss/Val', loss, val_iter)

                if (val_iter % 5 == 0):
                    print('[%d,%5d / %d] Rec loss :%.10f, lips loss : %.10f, time spent: %f s' % (
                        epoch + 1, i + 1, len(test_loader), loss, losses['l1'].item(), time.time() - a))

                if (i % 5 == 0):
                    input = draw_lmrk(lmrks['in'], lmrks['out'], outputs, batch_size)
                    vis_in = np.concatenate([input[0, 0:3].cpu().detach().numpy().transpose((1, 2, 0)),
                                             input[0, 3:6].cpu().detach().numpy().transpose((1, 2, 0)),
                                             input[0, 6:9].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                    try:
                        os.makedirs(os.path.join(config.lmrk_dir, config.name))
                    except:
                        pass
                    cv2.imwrite(
                        os.path.join(config.lmrk_dir, config.name,
                                     'Val_e{:03d}_b{:04d}.jpg'.format(epoch, i)),
                        cv2.cvtColor(vis_in * 255.0, cv2.COLOR_RGB2BGR))

                    with open(config.log_dir + 'val.txt', 'a') as file_handle:
                        file_handle.write(
                            '[%d,%5d / %d] Rec loss :%.10f,  time spent: %f s' % (
                        epoch + 1, i + 1, len(train_loader), losses['l1'].item(), time.time() - a))
                        file_handle.write('\n')
                val_iter += 1
