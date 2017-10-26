import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
import numpy as np

IN_DIM = 2048
BATCH_SIZE = 32


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear0 = nn.Linear(IN_DIM, 2 * IN_DIM)
        self.relu0 = nn.ReLU(True)

        self.linear1 = nn.Linear(4 * IN_DIM, 1 * IN_DIM)
        self.bn1 = nn.BatchNorm1d(1 * IN_DIM)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(1 * IN_DIM, 4 * IN_DIM)
        self.bn1 = nn.BatchNorm1d(4 * IN_DIM)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(4 * IN_DIM, 1 * IN_DIM)
        self.bn2 = nn.BatchNorm1d(IN_DIM)
        self.relu2 = nn.ReLU(inplace=True)

        self.linear3 = nn.Linear(IN_DIM, 1)

    def forward(self, candi, n_sample):
        x = self.linear0(candi)
        x = self.relu0(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        score = self.linear3(x)

        # output = output.view(-1, 4 * DIM, 4, 4)
        # # print output.size()
        # output = self.block1(output)
        # # print output.size()
        # output = output[:, :, :7, :7]
        # # print output.size()
        # output = self.block2(output)
        # # print output.size()
        # output = self.deconv_out(output)
        # output = self.sigmoid(output)
        # # print output.size()
        # return output.view(-1, OUTPUT_DIM)
        return score


if __name__ == '__main__':
    d = Discriminator(8, 256, 128, 3)
    in_ = Variable(torch.rand(8, 3, 256, 128))
    res = d(in_)
    print()
