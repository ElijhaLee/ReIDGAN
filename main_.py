import torch
from model.network import Network

net = Network()
net.cuda()
data = []

for i in range(10000):
    score_real, score_fake, prob = net(data)

    loss_select = net.loss_select(score_real, prob)
    loss_dis = net.loss_dis(score_real, score_fake)
    print('loss_select: %.4f\t loss_dis: %.4f' % (loss_select, loss_dis))

    net.bp(net.optmzr_select, loss_select)
    net.bp(net.optmzr_dis, loss_dis)
