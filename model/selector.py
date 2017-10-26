import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

IN_DIM = 2048
BATCH_SIZE = 32


class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()

        self.linear_anchor = nn.Linear(IN_DIM, 2 * IN_DIM)
        self.relu_anchor = nn.ReLU(True)

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

    def forward(self, anchor, candi, n_sample):
        anchor = self.linear_anchor(anchor)
        anchor = self.relu_anchor(anchor)

        candi = self.linear0(candi)
        candi = self.relu0(candi)

        x = torch.cat([anchor, candi])

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        score = self.linear3(x)

        score = score.view(1, -1)

        prob = nn.Softmax()(score)
        sample = self.__sample(candi, prob, n_sample)

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
        return score, prob, sample

    def __sample(self, data, prob, n_sample):
        indexes = torch.multinomial(prob, n_sample)
        res = data[indexes, :, :, :]
        return res


if __name__ == '__main__':
    s = Selector()
    input_ = torch.autograd.Variable(torch.Tensor(BATCH_SIZE, 2048))
    o1, o2 = s(input_)
    print()
