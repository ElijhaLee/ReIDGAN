from model.discriminator import Discriminator
from model.feature_extractor import resnet50
from model.selector import Selector
import torch.nn as nn
from torch.autograd.variable import Variable
import torch
from torch.optim.adam import Adam, Optimizer
import torch.nn.functional as F

BATCH_SIZE = 32
C_IN = 3
H_IN = 256
W_IN = 128


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_extractor = resnet50()  # Already pretrained
        # self.feature_extractor = resnet50(pretrained_path=None)
        self.selector = Selector()
        self.dis = Discriminator()
        self.optmzr_select = Adam(self.selector.parameters(), lr=1e-3)
        self.optmzr_dis = Adam(self.dis.parameters(), lr=1e-3)

    def forward(self, anchor: Variable, real_data: Variable, fake_data: Variable):
        assert len(anchor.size()) == 4 and len(anchor.size()) == 4

        fea_anchor = self.feature_extractor(anchor)
        fea_real = self.feature_extractor(real_data)
        fea_fake = self.feature_extractor(fake_data)

        # not train_feature:
        fea_anchor = fea_anchor.detach()
        fea_real = fea_real.detach()
        fea_fake = fea_fake.detach()

        score_real = self.dis(fea_anchor, fea_real)
        score_fake = self.dis(fea_anchor, fea_fake)

        return score_real, score_fake

    def bp_dis(self, score_real, score_fake):
        real_label = Variable(torch.normal(torch.ones(score_real.size()), torch.zeros(score_real.size()) + 0.05)).cuda()
        fake_label = Variable(
            torch.normal(torch.zeros(score_real.size()), torch.zeros(score_real.size()) + 0.05)).cuda()
        loss = torch.mean(F.binary_cross_entropy(score_real, real_label, size_average=False) + \
                          F.binary_cross_entropy(score_fake, fake_label, size_average=False))

        # loss = -(torch.mean(torch.log(score_real + 1e-6)) - torch.mean(torch.log(.5 + score_fake / 2 + 1e-6)))

        self.optmzr_dis.zero_grad()
        loss.backward()
        return self.optmzr_dis.step()

    def bp_select(self, score_fake: Variable, fake_prob):
        # torch.mean(torch.log(prob) * torch.log(1 - score_fake), 0)
        n_sample = score_fake.size()[0]
        self.optmzr_dis.zero_grad()
        re = (score_fake.data - .5) * 2
        torch.log(fake_prob).backward(re / n_sample)

        # def bp_select(self, score_fake: Variable, prob):
        #     # torch.mean(torch.log(prob) * torch.log(1 - score_fake), 0)
        #     n_sample = score_fake.size()[0]
        #     self.optmzr_dis.zero_grad()
        #     re = torch.log(1 - score_fake + 1e-6)
        #     torch.log(prob + 1e-6)
        #     torch.log(prob + 1e-6).backward(re.data)
        #     return self.optmzr_dis.step()
