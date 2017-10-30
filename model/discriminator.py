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

        self.linear_candi = nn.Linear(IN_DIM, 2 * IN_DIM)
        self.relu_candi = nn.ReLU(True)

        self.linear_anchor = nn.Linear(IN_DIM, 2 * IN_DIM)
        self.relu_anchor = nn.ReLU(True)

        self.linear1 = nn.Linear(4 * IN_DIM, 2 * IN_DIM)
        self.bn1 = nn.BatchNorm1d(2 * IN_DIM)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(2 * IN_DIM, 4 * IN_DIM)
        self.bn1 = nn.BatchNorm1d(4 * IN_DIM)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(4 * IN_DIM, 1 * IN_DIM)
        self.bn2 = nn.BatchNorm1d(IN_DIM)
        self.relu2 = nn.ReLU(inplace=True)

        self.linear3 = nn.Linear(IN_DIM, 1)

    def forward(self, anchor, candi):
        batch_size = candi.size()[0]

        x_anchor = self.linear_anchor(anchor)
        x_anchor = self.relu_anchor(x_anchor)
        x_anchor = x_anchor.expand(batch_size, 1)

        x_candi = self.linear_candi(candi)
        x_candi = self.relu_candi(x_candi)

        x = torch.cat([x_anchor, x_candi], 1)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.linear3(x)

        score = torch.tanh(x) / 2 + .5

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
