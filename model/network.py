from model.discriminator import Discriminator
from model.feature_extractor import Inception3
from model.selector import Selector
import torch.nn as nn
from torch.autograd.variable import Variable
import torch
from torch.optim.adam import Adam, Optimizer

BATCH_SIZE = 32
C_IN = 3
H_IN = 256
W_IN = 128


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_extractor = Inception3()
        self.selector = Selector()
        self.dis = Discriminator()
        self.optmzr_select = Adam(self.selector.parameters(), lr=1e-4)
        self.optmzr_dis = Adam(self.dis.parameters(), lr=1e-4)

    def forward(self, anchor: Variable, candi: Variable, n_sample):
        # 1,1024
        fea_anchor = self.feature_extractor(anchor)
        # bs, 1024
        fea_candi = self.feature_extractor(candi)

        _, prob, sample = self.selector(fea_anchor, fea_candi, n_sample)

        score_real = self.dis(anchor, candi)
        score_fake = self.dis(anchor, candi)

        return score_real, score_fake, prob

    def loss_select(self, score_fake, prob):
        return torch.mean(torch.log(prob) * torch.log(1 - score_fake), 0)

    def loss_dis(self, score_real, score_fake):
        return -(torch.mean(torch.log(score_real), 0) + torch.mean(torch.log(1 - score_fake), 0))

    def bp(self, optmzr: Optimizer, loss: Variable):
        def closure():
            optmzr.zero_grad()
            loss.backward()

        optmzr.step(closure)
