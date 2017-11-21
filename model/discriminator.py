import torch.nn as nn
import torch
from torch.autograd.variable import Variable
from torch.optim.adam import Adam
import torch.nn.functional as F

IN_DIM = 1024
BATCH_SIZE = 32


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # net
        self.linear0 = nn.Linear(2 * IN_DIM, IN_DIM)
        self.relu0 = nn.LeakyReLU(inplace=True)

        self.linear1 = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.linear2 = nn.Linear(IN_DIM // 2, 1)
        self.act_fn = nn.Tanh()

        # opt
        self.opt = Adam(self.parameters(), lr=1e-4)

    def forward(self, fea_q: Variable, fea_d: Variable):
        bs_anchor = fea_q.size()[0]
        bs_candi = fea_d.size()[0]
        assert bs_anchor == bs_candi or bs_anchor == 1

        fea_q = fea_q.expand(bs_candi, fea_q.size()[1])

        x = torch.cat([fea_q, fea_d], 1)

        x = self.linear0(x)
        x = self.relu0(x)

        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        x = x.view(-1)

        return x

    def bp(self, real_logit: Variable, fake_logit: Variable):
        size = real_logit.size()
        # real_label = Variable(torch.normal(torch.ones(size), torch.zeros(size) + 0.02)).cuda()
        # fake_label = Variable(torch.normal(torch.zeros(size), torch.zeros(size) + 0.02)).cuda()

        real_label = Variable(torch.ones(size)).cuda()
        fake_label = Variable(torch.zeros(size)).cuda()

        margins = F.threshold(0.8 - (torch.sigmoid(real_logit) - torch.sigmoid(fake_logit)), 0, 0)
        loss = torch.mean(F.binary_cross_entropy_with_logits(real_logit, real_label, size_average=False) + \
                          F.binary_cross_entropy_with_logits(fake_logit, fake_label, size_average=False)) + \
               torch.mean(margins)
        # loss = -(torch.mean(torch.log(score_real + 1e-6)) - torch.mean(torch.log(.5 + score_fake / 2 + 1e-6)))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


if __name__ == '__main__':
    d = Discriminator(8, 256, 128, 3)
    in_ = Variable(torch.rand(8, 3, 256, 128))
    res = d(in_)
    print()
