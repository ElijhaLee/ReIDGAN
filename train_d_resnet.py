import torch
from model.discriminator import Discriminator
from model.resnet50 import resnet50
from data_input.read_D import DataProvider
from tensorboardX import SummaryWriter
import os
from torch.autograd.variable import Variable
import torch.nn.functional as F
from torch.optim.adam import Adam
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
time.sleep(1.5)

# config
batch_size = 40
epoch_total = 1000
n_dis = 1
is_cuda = True
display_step = 10
save_step = 100
# net_pretrain = '/home/elijha/Documents/PycharmProjects/ReIDGAN/workdir/save/save-3400'
net_pretrain = True

# build graph
if net_pretrain:
    feature_extractor = resnet50(pretrained_path="/home/nhli/PycharmProj/ReIDGAN_/params/save-fea-2000")  # pretrained
    dis = Discriminator()
    dis.load_state_dict(torch.load("/home/nhli/PycharmProj/ReIDGAN_/params/save-dis-2000"))
else:
    feature_extractor = resnet50(pretrained_path=None)
    dis = Discriminator()

if is_cuda:
    feature_extractor.cuda()
    dis.cuda()

# input pipeline
data_iter = DataProvider(batch_size, is_cuda=is_cuda)

# summary writer
writer = SummaryWriter("/home/nhli/PycharmProj/ReIDGAN_/workdir/log/", 'comment test')

# opt
opt_d = Adam(dis.parameters())
opt_fea = Adam(feature_extractor.parameters())


def train_dis():
    # label
    real_label = Variable(torch.normal(torch.ones(batch_size), torch.zeros(batch_size) + 0.05)).cuda()
    fake_label = Variable(torch.normal(torch.zeros(batch_size), torch.zeros(batch_size) + 0.05)).cuda()

    anchor, real_img, wrong_img = data_iter.next()
    anchor, real_img, wrong_img = Variable(anchor), Variable(real_img), Variable(wrong_img)
    fea_anc = feature_extractor(anchor)
    fea_real = feature_extractor(real_img)
    fea_wrong = feature_extractor(wrong_img)

    score_real_logit = dis(fea_anc, fea_real)
    score_wrong_logit = dis(fea_anc, fea_wrong)

    loss = torch.mean(F.binary_cross_entropy_with_logits(score_real_logit, real_label, size_average=False) + \
                      F.binary_cross_entropy_with_logits(score_wrong_logit, fake_label, size_average=False))
    # loss = torch.mean(score_wrong_logit - score_real_logit)

    opt_d.zero_grad()
    opt_fea.zero_grad()
    loss.backward()
    opt_d.step()
    opt_fea.step()
    return loss, score_real_logit, score_wrong_logit


def display(global_step, epoch, loss, score_real_logit, score_wrong_logit):
    print('Step: %d (epoch: %d), loss: %.4f, real: %.4f, fake: %.4f,real_logit: %.4f, fake_logit: %.4f'
          % (global_step, epoch, loss.cpu().data.numpy(),
             torch.mean(torch.sigmoid(score_real_logit)).cpu().data.numpy(),
             torch.mean(torch.sigmoid(score_wrong_logit)).cpu().data.numpy(),
             torch.mean(score_real_logit).cpu().data.numpy(),
             torch.mean(score_wrong_logit).cpu().data.numpy()))
    writer.add_scalar('loss_dis', loss.cpu().data.numpy(), global_step=global_step)
    writer.add_scalar('score_real', torch.mean(score_real_logit).cpu().data.numpy(), global_step=global_step)
    writer.add_scalar('score_fake', torch.mean(score_wrong_logit).cpu().data.numpy(), global_step=global_step)

    # pass


global_step = 1
while data_iter.epoch <= epoch_total:
    # train
    loss, score_real_logit, score_wrong_logit = train_dis()
    # train_select()

    # print(global_step)
    if global_step % display_step == 0:
        display(global_step, data_iter.epoch, loss, score_real_logit, score_wrong_logit)

    if data_iter.epoch % save_step == 0:
        if data_iter.epoch == 0:
            continue
        if not os.path.isfile(
                        '/home/nhli/PycharmProj/ReIDGAN_/workdir/save/save-fea-%d' % data_iter.epoch):
            torch.save(feature_extractor.state_dict(),
                       '/home/nhli/PycharmProj/ReIDGAN_/workdir/save/save-fea-%d' % data_iter.epoch)
            torch.save(dis.state_dict(),
                       '/home/nhli/PycharmProj/ReIDGAN_/workdir/save/save-dis-%d' % data_iter.epoch)
            print(data_iter.epoch)

            # old_save = '/home/elijha/Documents/PycharmProjects/ReIDGAN/workdir/save/save-fea-%d' \
            #            % (global_step - 3 * save_step)
            # if os.path.isfile(old_save):
            #     os.remove(old_save)
            #
            # old_save = '/home/elijha/Documents/PycharmProjects/ReIDGAN/workdir/save/save-dis-%d' \
            #            % (global_step - 3 * save_step)
            # if os.path.isfile(old_save):
            #     os.remove(old_save)

    global_step += 1
