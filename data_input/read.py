import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage.io as io
import torchvision.transforms as transforms

mars_train_dir = '/home/elijha/Documents/Data/PersonReID/MARS/bbox_train'


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


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform_=None):
        self.data = get_data(data_dir)
        self.transform_ = transform_

    def __getitem__(self, index):
        img_path, label = self.data[index].path, self.data[index].label
        # img = Image.open(img_path)
        img = io.imread(img_path)

        if self.transform_ is not None:
            img = self.transform_(img)

        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = CustomDataset(mars_train_dir, transforms.ToTensor())
    dl = DataLoader(ds, 4)
    for d in dl:
        print(d)
        print()
