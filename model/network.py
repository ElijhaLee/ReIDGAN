from model.discriminator import Discriminator
from model.feature_extractor import Inception3
from model.selector import Selector
import torch.nn as nn
from torch.autograd.variable import Variable

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

    def forward(self, anchor: Variable, candi: Variable):
        fea_anchor = self.feature_extractor(anchor)
        fea_candi = self.feature_extractor
