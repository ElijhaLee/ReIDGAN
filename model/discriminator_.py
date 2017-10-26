import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
import numpy as np

df_dim = 64


class Discriminator(nn.Module):
    def __init__(self, batch_size, c_in, h_in, w_in):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.h_in = h_in
        self.w_in = w_in
        self.c_in = c_in
        self.df_dim = df_dim

        # 128-->64
        self.block1 = self.__block(self.c_in, self.df_dim, 5, 2, False)
        # 64-->32
        self.block2 = self.__block(1 * self.df_dim, 2 * self.df_dim, 4, 2, True)
        # 32-->16
        self.block3 = self.__block(2 * self.df_dim, 4 * self.df_dim, 3, 2, True)
        # 16-->8
        self.block4 = self.__block(4 * self.df_dim, 8 * self.df_dim, 3, 2, True)
        # 8-->4
        self.block5 = self.__block(8 * self.df_dim, 16 * self.df_dim, 3, 2, False)
        # h_in * w_in * c_in --> 1
        self.linear = nn.Linear(self.h_in // 32 * self.w_in // 32 * 16 * self.df_dim, 1)

    def forward(self, tensor_in):
        res = self.__forward(tensor_in, self.block1)
        res = self.__forward(res, self.block2)
        res = self.__forward(res, self.block3)
        res = self.__forward(res, self.block4)
        res = self.__forward(res, self.block5)
        res = self.linear(res.view(self.batch_size, -1))
        return res

    def __block(self, c_in, c_out, k_size, stride, is_bn):
        m = []
        m.append(nn.Conv2d(c_in, c_out, k_size, stride))
        if is_bn:
            m.append(nn.BatchNorm2d(c_out))
        m.append(nn.ReLU(True))
        return nn.Sequential(*m)

    def __forward(self, tensor_in: Variable, block: nn.Sequential):
        conv = block[0]
        k, _ = conv.kernel_size
        s, _ = conv.stride
        [_, _, h, w] = tensor_in.data.size()
        h_, w_ = np.ceil(h / s), np.ceil(w / s)
        p_h = s * (h_ - 1) + k - h
        p_w = s * (w_ - 1) + k - w
        tensor_padded = F.pad(tensor_in,
                              (int(np.floor(p_w / 2)), int(np.ceil(p_w / 2)),
                               int(np.floor(p_h / 2)), int(np.ceil(p_h / 2))))
        return block(tensor_padded)


if __name__ == '__main__':
    d = Discriminator(8, 256, 128, 3)
    in_ = Variable(torch.rand(8, 3, 256, 128))
    res = d(in_)
    print()
