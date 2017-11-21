"""
trp
epoch   train   test
399     45.97
499     55.39   26.88
599             31.73
799     61.06   32.7
1999    70.01   40.29
2200    74.23   40.56
2400    73.63   38.75
3000    76.26   38.69
6000            42.10

trp-fc (no eval)
400     64.04   36.45
1600    77.31   40.30
2200            40.26
3000            40.33
4000            40.71
5000            43.41

adv-fc-0
3-92            33.47
4-52            31.74
5-1     49.36   33.1

adv-fc (pretrained on fc no eval, train with eval)
        eval    no eval
0-90    40.28   45.65
=================================================
trp-fc
3999    42.02
adv-0
4-49    26.20
7-76    26
=================================================
trp-fc-new (new net struct, with margin)
3099    55.95
4599    56.77
"""

import torch
from torch.autograd.variable import Variable
import numpy as np
from model.discriminator import Discriminator
from model.gv2model import Inceptionv2
from model.selector import Selector
from data_input.read_D import Dataset_origin, DatasetFeatureHHH
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
time.sleep(1.5)


def cos_similarity(q, d):
    return F.cosine_similarity(q, d, dim=1)


def AP(q_id, doc_id_list, doc_score_tensor):
    sorted_score, sorted_index = torch.sort(doc_score_tensor, dim=0, descending=True)

    eql = np.array([q_id] * len(doc_id_list)) == np.array(doc_id_list)
    eql = np.array(eql).astype(np.float32)
    eql_tensor = torch.FloatTensor(eql).cuda()
    ranked_eql_tensor = torch.gather(eql_tensor, 0, sorted_index)

    numerator = torch.cumsum(ranked_eql_tensor, 0)
    denominator = torch.cumsum(torch.ones(ranked_eql_tensor.size()).cuda(), 0)
    AP = torch.sum(numerator / denominator * ranked_eql_tensor) / torch.sum(ranked_eql_tensor)

    # if sorted_score.is_cuda:
    #     sorted_score = sorted_score.cpu()
    # sorted_score = sorted_score.numpy()
    #
    # [length, ] = sorted_score.shape
    # hit_cnt = 0
    # total_hit = doc_id_list.count(q_id)
    # img_index = 0
    # point = []
    # for i in range(length):
    #     img_index += 1
    #     if q_id == doc_id_list[sorted_index[i]]:
    #         hit_cnt += 1
    #         point.append(hit_cnt / img_index)
    #         if hit_cnt >= total_hit:
    #             break
    #
    # AP = np.sum(np.array(point)) / total_hit
    return AP


def mAP(dis: Discriminator, dataset_fea: DatasetFeatureHHH, batch_size):
    AP_list = []

    for idx in range(len(dataset_fea.query)):
        [q_id, q_fea] = dataset_fea.get_query(idx)  # prop person
        doc_id_list = []
        doc_score_tensor = torch.FloatTensor().cuda()
        # test all persons
        dl = DataLoader(dataset_fea, batch_size, shuffle=False)
        for doc_ids, doc in dl:
            # query
            query = torch.unsqueeze(q_fea, 0)
            query = Variable(query.cuda(), volatile=True)
            # doc
            doc = Variable(doc.cuda(), volatile=True)

            doc_scores = dis(query, doc)          # metric learning similarity
            # doc_scores = cos_similarity(query, doc)  # cosine similarity

            doc_scores = doc_scores.data
            doc_id_list.extend(doc_ids)
            doc_score_tensor = torch.cat([doc_score_tensor, doc_scores])
            # doc_score_tensor.append(doc_scores.detach().cpu().data.numpy())
            # print(len(doc_id_list))
        AP_ = AP(q_id, doc_id_list, doc_score_tensor)
        AP_list.append(AP_)
        print(idx, AP_ * 100)

    AP_list = np.array(AP_list)
    mAP = np.mean(AP_list)
    print(mAP * 100)
    return mAP


def extract_all_feature(batch_size, feature_extractor):
    # train_or_test = 'train'
    train_or_test = 'test'
    dataset = Dataset_origin(data_dir="/home/nhli/SharedSSD/PersonReID/MARS/bbox_%s/" % train_or_test)
    dl = DataLoader(dataset, batch_size, shuffle=False, num_workers=1, drop_last=False)
    dataset_fea = DatasetFeatureHHH()
    for doc_ids, doc in dl:
        doc = Variable(doc.cuda(), volatile=True)
        feature = feature_extractor(doc)
        dataset_fea.insert(doc_ids, feature.data)
    dataset_fea.build()
    return dataset_fea


# class Net(torch.nn.Module):
#     def __init__(self, fea_ex, dis):
#         super(Net, self).__init__()
#         self.fea_ex = fea_ex
#         self.dis = dis
#
#     def forward(self, query_: torch.FloatTensor, doc_: torch.FloatTensor):
#         query_ = Variable(query_, volatile=True)
#         doc_ = Variable(doc_, volatile=True)
#         fea_q = self.fea_ex(query_)  # bs =1
#         fea_d = self.fea_ex(doc_)  # bs>1
#
#         res = self.dis(fea_q, fea_d)  # expand fea_q inside
#         return res.data  # return Tensor


if __name__ == '__main__':
    # config
    batch_size = 512
    is_cuda = True
    save_step = 100

    gv2_model_path = "/home/nhli/PycharmProj/ReIDGAN_/params/record-step-12685-model.pkl"
    # trp-fc-new
    dis_model_path = "/home/nhli/PycharmProj/ReIDGAN_/workdir/trp-fc-new/save-dis-4599"
    # adv-0
    # dis_model_path = "/home/nhli/PycharmProj/ReIDGAN_/workdir/adv-0/save-sel_4-49"

    # build graph
    feature_extractor = Inceptionv2()
    fea_ext_dict = torch.load(gv2_model_path)
    fea_ext_dict.pop('classifier.weight')
    fea_ext_dict.pop('classifier.bias')
    fea_ext_dict.pop('criterion2.center_feature')
    fea_ext_dict.pop('criterion2.all_labels')
    feature_extractor.load_state_dict(fea_ext_dict)

    dis = Discriminator()
    # dis = Selector()
    dis.load_state_dict(torch.load(dis_model_path))

    # eval
    dis.eval()
    feature_extractor.eval()

    if is_cuda:
        dis.cuda()
        feature_extractor.cuda()

    dataset_feature = extract_all_feature(batch_size, feature_extractor)

    mAP = mAP(dis, dataset_feature, batch_size)
