import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from PIL import Image
import skimage.io as io
import torchvision.transforms as transforms
from data_input.my_preprocess import generate_person_table, mars_train_root
import numpy as np
import torch

MAX_SELECT_NUM = 16


class Img:
    def __init__(self, path, label):
        self.path = path
        self.label = label


def get_data(data_dir):
    person_list = os.listdir(data_dir)
    img_label = []

    for per in person_list:
        img_list = os.listdir(os.path.join(data_dir, per))
        for img in img_list:
            img_label.append(Img(os.path.join(data_dir, per, img), int(per)))

    return img_label


def get_person_table(data_dir):
    return generate_person_table(data_dir)


class CustomDataset(Dataset):
    def __init__(self, batch_size: int, data_dir=mars_train_root, transform_=None):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.data = list(get_person_table(data_dir).table.items())
        self.transform_ = transform_

    def __getitem__(self, index):
        '''
        :param index:index of a item in self.data ( list(PersonTable.items()) )
        :return:an anchor and a candidate batch, all stack into one batch. REMEMBER TO SPLIT!
        '''
        per_id, person = self.data[index][0], self.data[index][1]
        anchor_and_candi = person.select(  # img name list from one person's different seq
            self.batch_size // 2 + 1)  # max to half of batch_size plus an anchor
        anchor_and_candi_path = [os.path.join(self.data_dir, str(per_id), img_name) for img_name in anchor_and_candi]
        selected_num = len(anchor_and_candi) - 1  # sub 1 anchor
        n_sample = selected_num  # num of right answer
        per_total_num = len(self.data)  # total person num
        while selected_num < self.batch_size:
            i = int(np.random.randint(0, per_total_num, ()))
            if i == index:
                continue
            anchor_and_candi_path.append(
                os.path.join(self.data_dir, str(self.data[i][0]), self.data[i][1].select(1)[0]))
            selected_num += 1

        # img = Image.open(img_path)

        img_list = [Image.open(img_path) for img_path in anchor_and_candi_path]

        if self.transform_ is not None:
            img_list = [self.transform_(img) for img in img_list]

        ret = torch.stack(img_list, 0)

        return ret[0:1], ret[1:], n_sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    return batch[0]


# def get_data_loader(batch_size):
#     dataset = CustomDataset(batch_size,
#                             transform_=transforms.Compose(
#                                 [transforms.Scale([224, 224]),
#                                  transforms.ToTensor()])
#                             )
#     # batch_size OF DATALOADER MUST BE 1, since each __getitem()__ return a batch.
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn, drop_last=True)
#     return dataloader


class CustomIterator:
    def __init__(self, batch_size, is_cuda):
        self.batch_size = batch_size
        self.dataset = CustomDataset(self.batch_size,
                                     transform_=transforms.Compose(
                                         [transforms.Scale([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])]),
                                     )
        self.is_cuda = is_cuda
        self.dataiter = None
        self.iteration = 0
        self.epoch = 0

    def build(self):
        # dataloader = get_data_loader(batch_size=self.batch_size)
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn,
                                drop_last=True)
        self.dataiter = DataLoaderIter(dataloader)

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iteration += 1

            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2]]
            return batch

        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration = 1  # reset and return the 1st batch

            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2]]
            return batch


if __name__ == '__main__':
    print()
