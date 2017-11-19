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
        self.linear_candi = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu_candi = nn.LeakyReLU(inplace=True)

        self.linear_anchor = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu_anchor = nn.LeakyReLU(inplace=True)

        self.linear1 = nn.Linear(IN_DIM, IN_DIM // 2)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.linear2 = nn.Linear(IN_DIM // 2, 1)
        self.act_fn = nn.Tanh()

        # opt
        self.opt = Adam(self.parameters(),lr=1e-4)

    def forward(self, fea_q: Variable, fea_d: Variable):
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

        x = x.view(-1)

        # score = self.act_fn(x) / 2 + .5
        # return score
        return x

    def bp(self, real_logit: Variable, fake_logit: Variable):
        size = real_logit.size()
        # real_label = Variable(torch.normal(torch.ones(size), torch.zeros(size) + 0.02)).cuda()
        # fake_label = Variable(torch.normal(torch.zeros(size), torch.zeros(size) + 0.02)).cuda()

        real_label = Variable(torch.ones(size)).cuda()
        fake_label = Variable(torch.zeros(size)).cuda()

        loss = torch.mean(F.binary_cross_entropy_with_logits(real_logit, real_label, size_average=False) + \
                          F.binary_cross_entropy_with_logits(fake_logit, fake_label, size_average=False))
        # loss = -(torch.mean(torch.log(score_real + 1e-6)) - torch.mean(torch.log(.5 + score_fake / 2 + 1e-6)))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


if __name__ == '__main__':
    d = Discriminator(8, 256, 128, 3)
    in_ = Variable(torch.rand(8, 3, 256, 128))
    res = d(in_)
    print()
