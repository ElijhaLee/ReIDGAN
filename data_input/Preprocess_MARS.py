import os
import numpy as np
from collections import Counter
from PIL import Image
import random
from Configuration import FLAGS

LEAST_SHOT = 2
MIN_SHOT_IMAGES = 20
MAX_SHOT_IMAGES = 1000000
PATH = FLAGS.root_path


def create_seq_sample(root_path):
    """
    
    :param root_path: 
    :return: total_path
    [(0, 'D:\\Data\\PersonReID\\MARS\\bbox_train/0001', [['0001C1T0002F001.jpg', ..., '0001C1T0002F079.jpg'], 
                                                         ['0001C1T0003F001.jpg', ..., '0001C1T0003F023.jpg'],
                                                                                .....
                                                         ['0001C1T0055F001.jpg', ..., '0001C1T0055F048.jpg']])
                                                 ............
     (631, 'D:\\Data\\PersonReID\\MARS\\bbox_train/0632', [['0632C1T0002F001.jpg', ..., '0632C1T0002F019.jpg'], 
                                                           ['0632C1T0003F001.jpg', ..., '0632C1T0003F066.jpg'],
                                                                                .....
                                                           ['0632C1T0044F001.jpg', ..., '0632C1T0044F035.jpg'])
    ]
    """
    total_path = []
    label = 0
    menus_1 = os.listdir(root_path)
    menus_1 = sorted(menus_1)
    for one_person in menus_1:
        one_person_path = os.path.join(root_path, one_person)
        imgs = os.listdir(one_person_path)
        imgs = sorted(imgs)
        shot_path = []
        for img in imgs:
            if img[:11] in shot_path:
                pass
            else:
                shot_path.append(img[:11])

        shot_path = sorted(shot_path)
        one_person_imgs = []
        for shot_name in shot_path:
            one_shot_img = sorted([k for k in imgs if shot_name in k])
            if MIN_SHOT_IMAGES < len(one_shot_img) < MAX_SHOT_IMAGES:
                one_person_imgs.append(one_shot_img)
        if LEAST_SHOT < len(one_person_imgs):
            total_path.append((label, one_person_path, one_person_imgs))
            label += 1

    return total_path


# ============================================================================
def count_per_shot(person_list):

    shot_count = []
    shot_img_count = []
    for person in person_list:
        shot_count.append(len(person[-1]))
        for shot in person[-1]:
            shot_img_count.append(len(shot))

    return shot_count, shot_img_count


def load_shot_seq_image(path, shot_name, max_img, height, width):
    seq_img = np.zeros((max_img, height, width, 3))
    seq_length = np.zeros(max_img)
    for i, image_name in enumerate(shot_name):
        if i < max_img:
            seq_img[i, :, :, :] = _load_img(os.path.join(path, image_name), target_size=(height, width))
            seq_length[i] = 1.0
        else:
            pass
    return seq_img, seq_length


def _load_img(path, crop_tuple=None, grayscale=False, target_size=None):

    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    x = np.asarray(img)
    if crop_tuple is None:
        x_crop = x
    else:
        x_crop = x[crop_tuple[1]:crop_tuple[1]+crop_tuple[3], crop_tuple[0]:crop_tuple[0]+crop_tuple[2], :]

    x_img = Image.fromarray(x_crop.astype('uint8'), 'RGB')
    if target_size:
        x_img = x_img.resize((target_size[1], target_size[0])) # shape = [width, height]
    x_end = np.asarray(x_img, dtype='float32')

    return x_end


# ---------------------------------------------------------------------------------------------------------------------
def create_person_list_txt(total_path_list):
    person_list = []
    person_writer = open(file=os.path.join(PATH, 'train.txt'), mode='w')
    for (label, path, shot_record) in total_path_list:
        person_list.append((path, label))
        person_writer.write(path+" %d" % label + "\n")

    person_writer.close()
    return person_list


def create_prob_gallery_txt(test_path_list):
    prob_list = []
    gallery_list = []
    recall_anchor = []
    prob_label = []
    gallery_label = []

    prob_writer = open(file=os.path.join(PATH, 'prob.txt'), mode='w')
    gallery_writer = open(file=os.path.join(PATH, 'gallery.txt'), mode='w')

    for (label, path, shot_record) in test_path_list:
        recall_anchor.append(len(shot_record)-1)
        for i, one_shot_imgs in enumerate(shot_record):
            if i == 0:
                prob_list.append((os.path.join(path, one_shot_imgs[0]), one_shot_imgs[0], label))
                prob_label.append(label)
                prob_writer.write(os.path.join(path, one_shot_imgs[0])+" %d" % label + "\n")
            else:
                gallery_list.append((os.path.join(path, one_shot_imgs[0]), one_shot_imgs[0], label))
                gallery_label.append(label)
                gallery_writer.write(os.path.join(path, one_shot_imgs[0])+" %d" % label + "\n")

    prob_writer.close()
    gallery_writer.close()
    return prob_list, gallery_list, recall_anchor, prob_label, gallery_label


def create_total_image_path(person_path):

    total_image_path = []
    for path in person_path:
        pics_path = os.listdir(path[0])
        random.shuffle(pics_path)
        total_image_path.append(pics_path)
    return total_image_path


def count_per_person_pic(total_image_path):
    count = []
    for person_pic in total_image_path:
        count.append(len(person_pic))
    pic_counter = Counter(count)
    return count, pic_counter


def image_minibatch(persons_path, image_names_in_path, width, height, channel=3):

    batch_size = len(persons_path)
    images_set = np.zeros((batch_size, len(image_names_in_path[0]), height, width, channel))

    if len(persons_path)!=len(image_names_in_path):
        raise ValueError("len(persons_path) and len(image_names_in_path) must be same!")

    for j, person_ID in enumerate(persons_path):
        for i, pic_name in enumerate(image_names_in_path[j]):
            # image_path = os.path.join(person_ID, pic_name)
            x = _load_img(os.path.join(person_ID, pic_name), grayscale=False, target_size=(height, width))
            images_set[j,i,:,:,:] = x
    return images_set

# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    root_path = r'D:\Data\PersonReID\MARS\bbox_train'


    total = create_seq_sample(root_path)
    print(total[0])
    shot_count, shot_img_count = count_per_shot(total)

    print('================data information==================')
    print('total satisfactory person number:', len(total))
    print('total shot number:', sum(shot_count))
    print('=================================================')

    shot_counter = Counter(shot_count)
    print('shot_counter length:', len(shot_counter))
    print('shot_counter statistic:')
    print(sorted(shot_counter.items(), key=lambda item:item[0]))

    shot_img_counter = Counter(shot_img_count)
    print('shot_img_counter length:', len(shot_img_counter))
    print('shot_img_counter statistic:')
    print(sorted(shot_img_counter.items(), key=lambda item: item[0]))


