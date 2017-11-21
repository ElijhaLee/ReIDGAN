import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from PIL import Image
import torchvision.transforms as transforms
from data_input.my_preprocess import generate_person_table, mars_train_root, PersonTable
import numpy as np
import torch
from collections import defaultdict

MAX_SELECT_NUM = 16
HHH = 4

default_transform = transforms.Compose(
    [transforms.Scale([299, 299]),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])


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


class DatasetFeatureHHH(Dataset):
    def __init__(self):
        # self.per_imgs_list = list()
        self.per_imgs_list = defaultdict(list)
        self.query = []
        self.doc_part = []
        self.doc = []

    def insert(self, key, value):
        for i in range(len(key)):
            self.per_imgs_list[key[i]].append(value[i])
        print(len(self.per_imgs_list))

    def build(self):
        for per_id, imgs in self.per_imgs_list.items():
            q_idx = np.random.randint(0, len(imgs))
            q = imgs.pop(q_idx)
            self.query.append((per_id, q))
            self.doc_part.extend(tuple(zip([per_id] * len(imgs), imgs)))

    def get_query(self, idx):
        query = self.query.pop(idx)
        self.doc = self.doc_part + self.query
        self.query.insert(idx, query)
        return query

    def __getitem__(self, index):
        return self.doc[index]

    def __len__(self):
        return len(self.doc)


class Dataset_origin(Dataset):
    def __init__(self, data_dir=mars_train_root, transform_=default_transform):
        self.data_dir = data_dir
        self.per_table = list(get_person_table(data_dir).table.items())
        self.pair_list = []
        for per_id, per in self.per_table:
            pics = per.select_all()
            per_ids = [per_id] * len(pics)
            self.pair_list.extend(list(zip(per_ids, pics)))

        self.data_len = len(self.pair_list)
        self.transform_ = transform_

    def __getitem__(self, index):
        '''
        :param index:index of a item in self.data ( list(PersonTable.items()) )
        :return:an anchor and a candidate batch, all stack into one batch. REMEMBER TO SPLIT!
        '''
        per_id = self.pair_list[index][0]
        img = Image.open(os.path.join(self.data_dir, per_id, self.pair_list[index][1]))
        # img_names = person.select_all()

        if self.transform_ is not None:
            img = self.transform_(img)

        return per_id, img

    def __len__(self):
        return self.data_len

    def get_person_img(self, idx):
        per_id, per = self.per_table[idx]
        per_img = per.select(1)[0]
        img = Image.open(os.path.join(self.data_dir, per_id, per_img))
        if self.transform_ is not None:
            img = self.transform_(img)
        return per_id, img


def get_gallery(dataset: Dataset):
    """
    get a dict {per_id: str, imgs:torch.FloatTensor}
    :param dataset:
    :return:
    """
    res = {}
    for i in range(len(dataset)):
        per_id, imgs = dataset[i]
        res[per_id] = imgs

    return res


class Dataset_triple(Dataset):
    def __init__(self, batch_size: int, data_dir=mars_train_root, transform_=default_transform):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.data = list(get_person_table(data_dir).table.items())
        self.data_len = len(self.data)
        self.transform_ = transform_

    def __getitem__(self, index):
        '''
        :param index:index of a item in self.data ( list(PersonTable.items()) )
        :return:an anchor and a candidate batch, all stack into one batch. REMEMBER TO SPLIT!
        '''
        per_id, person = self.data[index][0], self.data[index][1]
        wrong_index = np.random.randint(0, self.data_len)
        while wrong_index == index:
            wrong_index = np.random.randint(0, self.data_len)

        [wrong_id, wrong_person] = self.data[wrong_index]

        [anchor, real_img] = person.select(2)
        [wrong_img] = wrong_person.select(1)

        anchor = Image.open(os.path.join(self.data_dir, str(per_id), anchor))
        real_img = Image.open(os.path.join(self.data_dir, str(per_id), real_img))
        wrong_img = Image.open(os.path.join(self.data_dir, str(wrong_id), wrong_img))

        if self.transform_ is not None:
            anchor = self.transform_(anchor)
            real_img = self.transform_(real_img)
            wrong_img = self.transform_(wrong_img)

        return anchor, real_img, wrong_img

    def __len__(self):
        return self.data_len


class DatasetMini(Dataset):
    def generate_subset(self):
        res = []
        for person in self.person_table.table.values():
            img_list = person.select(HHH)
            ids = [person.id] * HHH
            idAndImgPath = list(zip(ids, [os.path.join(self.imgRoot, person.id, img) for img in img_list]))
            res.extend(idAndImgPath)
        self.idAndImgPathList = res

    def __init__(self, personTable: PersonTable, imgRoot, transforms_=default_transform):
        self.imgRoot = imgRoot
        self.person_table = personTable
        self.idAndImgPathList = None
        self.transforms_ = transforms_

    def __getitem__(self, idx):
        [id, imgPath] = self.idAndImgPathList[idx]
        # queryPath = self.query_gallery[idx % HHH][1]
        img = Image.open(imgPath)
        # query_img = Image.open(queryPath)
        if self.transforms_ is not None:
            img = self.transforms_(img)
            # query_img = self.transforms_(query_img)
        return id, img  # , query_img

    def __len__(self):
        return len(self.idAndImgPathList)


class DatasetFeature(Dataset):
    def __init__(self, feature_id: list, feature_tensor: torch.FloatTensor):
        self.feature_id = feature_id
        self.feature_tensor = feature_tensor
        self.query_gallery = None

    def __getitem__(self, idx):
        return self.feature_id[idx], self.feature_tensor[idx], self.query_gallery[idx % len(self.query_gallery)]

    def __len__(self):
        return len(self.feature_id)

    def random_query_gallery(self):
        query_idx = []
        id = self.feature_id[np.random.randint(0, len(self.feature_id))]
        for i in range(len(self.feature_id)):
            if self.feature_id[i] == id:
                query_idx.append(i)
        query_idx = torch.LongTensor(query_idx).cuda()
        gallery = torch.index_select(self.feature_tensor, 0, query_idx)
        self.query_gallery = gallery


class DataProvider:
    def __init__(self, batch_size, is_cuda):
        self.batch_size = batch_size
        self.dataset = Dataset_triple(self.batch_size,
                                      transform_=default_transform)
        self.is_cuda = is_cuda
        self.dataiter = None
        self.iteration = 0
        self.epoch = 0

    def build(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.dataiter = DataLoaderIter(dataloader)

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iteration += 1

            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch

        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration = 1  # reset and return the 1st batch

            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch


if __name__ == '__main__':
    print()
