import torch.nn as nn
import torch
from torch.autograd.variable import Variable
from torch.optim.adam import Adam
import torch.nn.functional as F
from data_input.read_D import DatasetMini
from torch.utils.data import DataLoader

IN_DIM = 1024
BATCH_SIZE = 32


def sample(feature, distribution, n_sample):
    indexes = torch.multinomial(distribution, n_sample, replacement=True).detach()
    selected_fea = torch.index_select(feature, 0, indexes)
    selected_prob = torch.index_select(distribution, 0, indexes)
    return selected_fea.detach(), selected_prob  # prob needs bp, so do not detach


def random_select(feature, n_sample, is_cuda):
    weight = torch.ones(feature.size()[0])
    idx = torch.multinomial(weight, n_sample, replacement=True)
    if is_cuda:
        idx = idx.cuda()
    return torch.index_select(feature, 0, idx)


class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()

        # net
        self.linear_candi = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu_candi = nn.LeakyReLU(inplace=True)

        self.linear_anchor = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu_anchor = nn.LeakyReLU(inplace=True)

        self.linear1 = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.linear2 = nn.Linear(IN_DIM // 2, 1)
        self.act_fn = nn.Tanh()

        # opt
        self.opt = Adam(self.parameters(), lr=1e-4)

    def forward(self, fea_q, fea_d):
        bs_anchor = fea_q.size()[0]
        bs_candi = fea_d.size()[0]
        assert bs_anchor == bs_candi or bs_anchor == 1

        x_anchor = self.linear_anchor(fea_q)
        x_anchor = self.relu_anchor(x_anchor)
        x_anchor = x_anchor.expand(bs_candi, x_anchor.size()[1])

        x_candi = self.linear_candi(fea_d)
        x_candi = self.relu_candi(x_candi)

        x = torch.cat([x_anchor, x_candi], 1)

        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        logit = x.view(-1)

        return logit

    def bp(self, fake_logit: Variable, prob):
        bs = fake_logit.size()[0]
        self.opt.zero_grad()
        reward = torch.tanh(fake_logit.detach())
        # loss = -(torch.mean(torch.log(prob) * reward)).backward()
        torch.log(prob).backward(-reward / bs)

        self.opt.step()


if __name__ == '__main__':
    s = Selector()
    input_ = torch.autograd.Variable(torch.Tensor(BATCH_SIZE, 2048))
    o1, o2 = s(input_)
    print()
