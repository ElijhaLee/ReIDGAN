import torch
from model.discriminator import Discriminator
from model.gv2model import Inceptionv2
from data_input.read_D import DataProvider
from tensorboardX import SummaryWriter
import os
from torch.autograd.variable import Variable
import torch.nn.functional as F
from torch.optim.adam import Adam
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
time.sleep(1.5)
# config
batch_size = 64
epoch_total = 10000
is_cuda = True
save_step = 100
display_step = 10
gv2_model_path = "/home/nhli/PycharmProj/ReIDGAN_/params/record-step-12685-model.pkl"
dis_model_path = None
# dis_model_path = "/home/nhli/PycharmProj/ReIDGAN_/workdir/trp-fc/epoch-4000/save-dis-3999"
save_path = "/home/nhli/PycharmProj/ReIDGAN_/workdir/trp-fc-new"
log_path = None

# build graph
feature_extractor = Inceptionv2()
feature_extractor.eval()
dis = Discriminator()
if gv2_model_path is not None:
    fea_dict = torch.load(gv2_model_path)
    fea_dict.pop('classifier.weight')
    fea_dict.pop('classifier.bias')
    fea_dict.pop('criterion2.center_feature')
    fea_dict.pop('criterion2.all_labels')
    feature_extractor.load_state_dict(fea_dict)
if dis_model_path is not None:
    dis.load_state_dict(torch.load(dis_model_path))

if is_cuda:
    feature_extractor.cuda()
    dis.cuda()

# input pipeline
data_iter = DataProvider(batch_size, is_cuda=is_cuda)

# summary writer
if log_path is not None:
    writer = SummaryWriter(log_path, 'comment test')
else:
    writer = None

# opt
opt_d = Adam(dis.parameters())


# opt_fea = Adam(feature_extractor.parameters())


def train_dis():
    anchor, real_img, wrong_img = data_iter.next()
    anchor, real_img, wrong_img = Variable(anchor), Variable(real_img), Variable(wrong_img)
    anchor.volatile = True
    real_img.volatile = True
    wrong_img.volatile = True
    fea_anc = feature_extractor(anchor).detach()
    fea_real = feature_extractor(real_img).detach()
    fea_wrong = feature_extractor(wrong_img).detach()
    fea_anc.volatile = False
    fea_real.volatile = False
    fea_wrong.volatile = False

    score_real_logit = dis(fea_anc, fea_real)
    score_wrong_logit = dis(fea_anc, fea_wrong)

    loss = dis.bp(score_real_logit, score_wrong_logit)

    return loss, score_real_logit, score_wrong_logit


def display(global_step, epoch, loss, score_real_logit, score_wrong_logit):
    margins = F.threshold(torch.sigmoid(score_real_logit) - torch.sigmoid(score_wrong_logit), 0, 0)
    nonzero_cnt = len(torch.nonzero(margins.data))
    print('Step: %d (epoch: %d), loss: %.4f, real: %.4f, fake: %.4f,real_logit: %.4f, fake_logit: %.4f, margin: %.4f'
          % (global_step, epoch, loss.cpu().data.numpy(),
             torch.mean(torch.sigmoid(score_real_logit)).cpu().data.numpy(),
             torch.mean(torch.sigmoid(score_wrong_logit)).cpu().data.numpy(),
             torch.mean(score_real_logit).cpu().data.numpy(),
             torch.mean(score_wrong_logit).cpu().data.numpy(),
             torch.sum(margins.data / nonzero_cnt) if nonzero_cnt > 0 else 0))
    if writer:
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
    if (global_step % display_step) == 0:
        display(global_step, data_iter.epoch, loss, score_real_logit, score_wrong_logit)

    if (data_iter.epoch + 1) % save_step == 0:
        if os.path.isfile(os.path.join(save_path, 'save-dis-%d' % data_iter.epoch)):
            continue
        # torch.save(feature_extractor.state_dict(), os.path.join(save_path, 'save-fea-%d' % data_iter.epoch))
        torch.save(dis.state_dict(), os.path.join(save_path, 'save-dis-%d' % data_iter.epoch))

    global_step += 1
