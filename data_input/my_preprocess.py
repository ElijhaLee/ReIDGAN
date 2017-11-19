import pickle as pkl
import os as os
from collections import defaultdict
import numpy as np

mars_train_root = "/home/nhli/SharedSSD/PersonReID/MARS/bbox_train/"
MIN_SEQ_NUM_PER_PER = 4


class SeqTable:
    def __init__(self):
        self.table = defaultdict(list)

    def add_img_to_seq(self, img):
        seq_id = img[6:11]
        self.table[seq_id].append(img)

    def clean_seq(self, min_img_num):
        for item in self.table.items():
            img_cnt = len(item[1])
            if img_cnt < min_img_num:
                self.table.pop(item[0])

    def select(self, max_seq):
        if max_seq <= 2:
            res = []
            for seq_idx in range(max_seq):
                seq_index = np.random.randint(0, len(self.table))
                item = list(self.table.items())[seq_index]
                total_img = len(item[1])
                img_idx = int(np.random.randint(0, total_img, []))
                res.append(item[1][img_idx])
            return res

        seq_index = np.random.permutation(len(self.table))[:min(max_seq, len(self.table))]

        res = []
        seqs = list(self.table.values())
        for seq_idx in seq_index:
            total_img = len(seqs[seq_idx])
            img_idx = int(np.random.randint(0, total_img))
            res.append(seqs[seq_idx][img_idx])
        return res

    def select_all(self):
        max_seq = len(self.table)
        return self.select(max_seq)


class Person:
    def __init__(self, id: str):
        self.id = id
        self.seqTable = None

    def set_seq_table(self, seqTable: SeqTable):
        self.seqTable = seqTable

    def get_seq_num(self):
        return len(self.seqTable.table.items())

    def select(self, max_seq_num):
        return self.seqTable.select(max_seq_num)

    def select_all(self):
        return self.seqTable.select_all()


class PersonTable:
    def __init__(self):
        self.table = dict()

    def add_person_to_table(self, per: Person):
        per_id = per.id
        self.table[per_id] = per

    def clean_person(self, min_seq_num):
        id_to_clean = {}
        for item in self.table.items():
            seq_cnt = item[1].get_seq_num()
            if seq_cnt < min_seq_num:
                id_to_clean[item[0]] = seq_cnt
        for id in id_to_clean.items():
            self.table.pop(id[0])
            # print('Person: %s is cleaned: %d' % (id[0], id[1]))


def generate_person_table(mars_root=mars_train_root):
    per_table = PersonTable()

    per_dir_list = os.listdir(mars_root)

    per_num = len(per_dir_list)
    per_cnt = 0

    for per_dir in per_dir_list:
        if not os.path.isdir(os.path.join(mars_root, per_dir)):
            continue
        img_list = os.listdir(os.path.join(mars_root, per_dir))
        per = Person(per_dir)
        seqTable = SeqTable()
        for img in img_list:
            seqTable.add_img_to_seq(img)
        per.set_seq_table(seqTable)
        per_table.add_person_to_table(per)
        per_cnt += 1
        # print('%d / %d' % (per_cnt, per_num))

    per_table.clean_person(MIN_SEQ_NUM_PER_PER)
    return per_table


def dump_person_table(person_table: PersonTable, out_path):
    out_file = open(os.path.join(out_path, 'person_table.pkl'), 'wb')
    pkl.dump(person_table, out_file)

    print('Person_table dumped! Total person: %d' % len(per_table.table))


if __name__ == '__main__':
    per_table = generate_person_table(mars_train_root)
    dump_person_table(per_table, mars_train_root)

    print()
