import torch
from data_input.read_D import get_person_table, mars_train_root, DatasetMini, DatasetFeature
from tensorboardX import SummaryWriter
import os
import numpy as np
from model.gv2model import Inceptionv2
from model.discriminator import Discriminator
from model.selector import Selector, sample, random_select
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
import os
import time
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
time.sleep(1.5)

# config
batch_size = 128
epoch_total = 10
n_gen = 1
n_dis = 5
is_cuda = True
display_step = 10
save_step = 10
net_pretrain = True
save_path = "/home/nhli/PycharmProj/ReIDGAN_/workdir/adv-res_11-15/from-0/"
log_path = "/home/nhli/PycharmProj/ReIDGAN_/workdir/adv-res_11-15/from-0/"

# build graph
if net_pretrain:
    feature_extractor = Inceptionv2()
    fea_ext_dict = torch.load("/home/nhli/PycharmProj/ReIDGAN_/params/record-step-12685-model.pkl")
    fea_ext_dict.pop('classifier.weight')
    fea_ext_dict.pop('classifier.bias')
    fea_ext_dict.pop('criterion2.center_feature')
    fea_ext_dict.pop('criterion2.all_labels')
    feature_extractor.load_state_dict(fea_ext_dict)

    dis = Discriminator()
    dis.load_state_dict(
        torch.load("/home/nhli/PycharmProj/ReIDGAN_/workdir/adv-res_11-15/from-0/save-dis_0-99"))

    selector = Selector()
    selector.load_state_dict(
        torch.load("/home/nhli/PycharmProj/ReIDGAN_/workdir/adv-res_11-15/from-0/save-sel_0-99"))
else:
    feature_extractor = Inceptionv2
    dis = Discriminator()
    selector = Selector()

feature_extractor.eval()

feature_extractor.cuda()
dis.cuda()
selector.cuda()

# summary writer
if log_path is not None:
    writer = SummaryWriter(log_path, 'comment test')
else:
    writer = None

epoch_done = 0
person_table = get_person_table(mars_train_root)
total_person = len(person_table.table)
transforms_ = transforms.Compose(
    [transforms.Scale([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

K = 100
M = 500
dataset = DatasetMini(person_table, mars_train_root, transforms_=transforms_)

for epoch_done in range(1,epoch_total):
    print('Epoch: %d' % (epoch_done + 1))
    for subset_idx in range(K):
        dataset.generate_subset()

        # features of this subset
        dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=1)
        fea_id = []
        fea_extracted = []
        print('Subset %d, calculating features...' % subset_idx, end=' ')
        for id, fea_d in dl:
            # Cuz feature extractor doesnt needs train, so volatile=True to save time
            fea_d = Variable(fea_d, volatile=True).cuda()
            fea_batch = feature_extractor(fea_d).detach()
            # set volatile to False for training following nets
            fea_batch.volatile = False
            fea_id.extend(id)
            fea_extracted.append(fea_batch.data)
        fea_extracted = torch.cat(fea_extracted, dim=0)
        dataset_fea = DatasetFeature(fea_id, fea_extracted)
        print('done!')

        # calculate distribution of all img in dataset
        for dist_idx in range(M):
            # print('dist_idx: %d:' % dist_idx, end='  ')
            # set a query person
            dataset_fea.random_query_gallery()

            # dl = DataLoader(dataset_fea, batch_size * 10, shuffle=True)
            dl = DataLoader(dataset_fea, len(dataset_fea), shuffle=True)
            # fake_dist_logit = []
            # fea_doc = []
            fake_dist_logit = None
            fea_doc = None
            # print('Calculating distribution...', end=' ')
            for _, fea_d, fea_q in dl:
                # Cuz feature extractor doesnt needs train, so volatile=True to save time
                fea_d = Variable(fea_d).cuda()
                fea_q = Variable(fea_q).cuda()

                fake_dist_logit_batch = selector.forward(fea_q, fea_d)
                # fea_doc.append(fea_d)
                # fake_dist_logit.append(fake_dist_logit_batch)
                fea_doc = fea_d
                fake_dist_logit = fake_dist_logit_batch
            fake_dist = nn.Softmax()(torch.cat(fake_dist_logit, dim=0).view(1, -1)).view(-1)
            # fea_doc = torch.cat(fea_doc)
            # print('done!', end='  ')

            # print('Training...', end=' ')
            # train D
            for k in range(n_dis):
                fea_fake, prob_fake = sample(fea_doc, fake_dist, batch_size)
                fea_real = Variable(random_select(dataset_fea.query_gallery, batch_size, is_cuda=is_cuda))
                fea_query = Variable(random_select(dataset_fea.query_gallery, batch_size, is_cuda=is_cuda))
                logit_real = dis(fea_query, fea_real)
                logit_fake = dis(fea_query, fea_fake)
                dis.bp(logit_real, logit_fake)

            # train G
            for j in range(n_gen):
                fea_fake, prob_fake = sample(fea_doc, fake_dist, batch_size)
                fea_query = Variable(random_select(dataset_fea.query_gallery, batch_size, is_cuda=is_cuda))
                logit_fake = dis(fea_query, fea_fake)
                selector.bp(logit_fake, prob_fake)
            # print('done!')

            # show
            if (dist_idx + 1) % display_step == 0:
                fea_fake, prob_fake = sample(fea_doc, fake_dist, batch_size)
                fea_real = Variable(random_select(dataset_fea.query_gallery, batch_size, is_cuda=is_cuda))
                fea_query = Variable(random_select(dataset_fea.query_gallery, batch_size, is_cuda=is_cuda))
                logit_real = dis(fea_query, fea_real)
                logit_fake = dis(fea_query, fea_fake)

                reward = torch.tanh(logit_fake)
                loss_g = -torch.mean(torch.log(1e-6 + prob_fake) * reward)
                loss_d = -(torch.mean(torch.log(torch.sigmoid(logit_real) + 1e-6))
                           + torch.mean(torch.log(1 - torch.sigmoid(logit_fake) + 1e-6)))

                print('dist: %d: loss_g: %.4f, loss_d: %.4f; real: %.4f /%.4f, fake: %.4f /%.4f; reward: %.4f'
                      % (dist_idx,
                         loss_g.data.cpu().numpy()[0], loss_d.data.cpu().numpy()[0],
                         torch.mean(logit_real).data.cpu().numpy()[0],
                         torch.mean(torch.sigmoid(logit_real)).data.cpu().numpy()[0],
                         torch.mean(logit_fake).data.cpu().numpy()[0],
                         torch.mean(torch.sigmoid(logit_fake)).data.cpu().numpy()[0],
                         torch.mean(reward).data.cpu().numpy()[0]))

                # w
                if writer is not None:
                    d = {'loss_g': loss_g.data,
                         'loss_d': loss_d.data,
                         'logit_real': torch.mean(logit_real).data,
                         'real': torch.mean(torch.sigmoid(logit_real)).data,
                         'logit_fake': torch.mean(logit_fake).data,
                         'fake': torch.mean(torch.sigmoid(logit_fake)).data,
                         'reward': torch.mean(reward).data
                         }
                    # writer.add_scalars('print', d, epoch_done * K * M + subset_idx * M + dist_idx)
                    writer.add_scalar('a', loss_g.data.cpu().numpy(), epoch_done * K * M + subset_idx * M + dist_idx)

        # save each subset
        torch.save(dis.state_dict(), os.path.join(save_path, 'save-dis_%d-%d' % (epoch_done, subset_idx)))
        torch.save(selector.state_dict(), os.path.join(save_path, 'save-sel_%d-%d' % (epoch_done, subset_idx)))
        print('Saved! Epoch: %d, subset_idx: %d' % (epoch_done, subset_idx))
