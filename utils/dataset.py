import dill
import numpy as np
import torch
import os

class PKLSet(object):
    def __init__(self, batch_size, dataset):
        self.eval_path = os.path.join('datasets', dataset + '/data_eval.pkl')
        self.voc_path = os.path.join('datasets', dataset + '/voc_final.pkl')
        self.ddi_adj_path = os.path.join('datasets', dataset + '/ddi_A_final.pkl')
        self.ddi_adj = dill.load(open(self.ddi_adj_path, 'rb'))
        # self.ddi_adj = 0
        self.sym_train, self.drug_train, self.data_eval = self.check_file(batch_size, dataset)
        self.sym_sets, self.drug_multihots = self.mat_train_data(dataset)
        self.similar_sets_idx = self.find_similae_set_by_ja(self.sym_train)
        # self.sym_counts = self.count_sym(self.sym_train)

    def check_file(self, batch_size, dataset):
        sym_path = os.path.join('datasets', '{}/sym_train_{}.pkl'.format(dataset, batch_size))
        drug_path = os.path.join('datasets', '{}/drug_train_{}.pkl'.format(dataset, batch_size))
        if not os.path.exists(sym_path):
            self.gen_batch_data(batch_size, dataset)
        return self.load_data(sym_path, drug_path)

    def load_data(self, sym_path, drug_path):
        sym_train, drug_train = dill.load(open(sym_path, 'rb')), dill.load(open(drug_path, 'rb'))
        data_eval = dill.load(open(self.eval_path, 'rb'))
        voc = dill.load(open(self.voc_path, 'rb'))
        sym_voc, pro_voc, med_voc = voc['sym_voc'], voc['diag_voc'], voc['med_voc']

        self.n_sym, self.n_drug = len(sym_voc.idx2word), len(med_voc.idx2word)
        print("num symptom: {}, num drug: {}".format(self.n_sym, self.n_drug))
        return sym_train, drug_train, data_eval

    def count_sym(self, dataset):
        train_path = os.path.join('datasets', dataset + '/data_train.pkl')
        data = dill.load(open(train_path, 'rb'))
        countings = np.zeros(self.n_sym)
        for adm in data:
            syms, drugs = adm[0], adm[2]
            countings[syms] += 1
        return countings

    def mat_train_data(self, dataset):
        train_path = os.path.join('datasets', dataset + '/data_train.pkl')
        data_train = dill.load(open(train_path, 'rb'))
        sym_sets, drug_sets_multihot = [], []
        for adm in data_train:
            syms, drugs = adm[0], adm[2]
            sym_sets.append(syms)
            drug_multihot = np.zeros(self.n_drug)
            drug_multihot[drugs] = 1
            drug_sets_multihot.append(drug_multihot)
        return sym_sets, drug_sets_multihot

    def gen_batch_data(self, batch_size, dataset):
        voc = dill.load(open(self.voc_path, 'rb'))
        sym_voc, pro_voc, med_voc = voc['sym_voc'], voc['diag_voc'], voc['med_voc']

        self.n_sym, self.n_drug = len(sym_voc.idx2word), len(med_voc.idx2word)
        sym_count = self.count_sym(dataset)
        size_dict, drug_dict = {}, {}
        sym_sets, drug_sets = [], []
        s_set_num = 0

        train_path = os.path.join('datasets', dataset + '/data_train.pkl')
        data = dill.load(open(train_path, 'rb'))
        for adm in data:
            syms, drugs = adm[0], adm[2]
            sym_sets.append(syms)
            drug_sets.append(drugs)
            s_set_num += 1

        for adm in data:
            syms, drugs = adm[0], adm[2]
            drug_multihot = np.zeros(self.n_drug)
            drug_multihot[drugs] = 1
            if size_dict.get(len(syms)):
                size_dict[len(syms)].append(syms)
                drug_dict[len(syms)].append(drug_multihot)
            else:
                size_dict[len(syms)] = [syms]
                drug_dict[len(syms)] = [drug_multihot]

        keys, count = list(size_dict.keys()), 0
        keys.sort()
        new_s_set, new_d_set = [], []
        for size in keys:
            if size <= 2: continue
            for (syms, drugs) in zip(size_dict[size], drug_dict[size]):
                syms = np.array(syms)
                cnt, del_nums = torch.from_numpy(sym_count[syms]), int(max(1, len(syms) * 0.2))
                if del_nums == 1:
                    del_idx = torch.multinomial(cnt, len(syms) - del_nums)
                    remained = syms[del_idx.numpy()]
                    remained = remained.tolist()
                    new_s_set.append(remained)
                    new_d_set.append(drugs)
                else:
                    for _ in range(min(del_nums, 3)):
                        del_num = np.random.randint(1, del_nums)
                        del_idx = torch.multinomial(cnt, len(syms) - del_num)
                        remained = syms[del_idx.numpy()]
                        remained = remained.tolist()
                        new_s_set.append(remained)
                        new_d_set.append(drugs)

        for (remained, drugs) in zip(new_s_set, new_d_set):
            if size_dict.get(len(remained)) is None:
                count += 1
                size_dict[len(remained)] = [remained]
                drug_dict[len(remained)] = [drugs]
            elif remained not in size_dict[len(remained)]:
                count += 1
                size_dict[len(remained)].append(remained)
                drug_dict[len(remained)].append(drugs)

        sym_train, drug_train = [], []
        keys = list(size_dict.keys())
        keys.sort()
        for size in keys:
            num_size = len(size_dict[size])
            batch_num, start_idx = num_size // batch_size, 0
            if num_size % batch_size != 0: batch_num += 1
            for i in range(batch_num):
                if i == batch_num:
                    syms, drugs = size_dict[size][start_idx:], drug_dict[size][start_idx:]
                else:
                    syms, drugs = size_dict[size][start_idx:start_idx + batch_size], drug_dict[size][
                                                                                     start_idx:start_idx + batch_size]
                    start_idx += batch_size
                sym_train.append(syms)
                drug_train.append(drugs)

        with open(os.path.join('datasets', '{}/sym_train_{}.pkl'.format(dataset, batch_size)), 'wb') as f:
            dill.dump(sym_train, f)

        with open(os.path.join('datasets', '{}/drug_train_{}.pkl'.format(dataset, batch_size)), 'wb') as f:
            dill.dump(drug_train, f)

    def find_similae_set_by_ja(self, sym_train):
        similar_sets = [[] for _ in range(len(sym_train))]
        for i in range(len(sym_train)):
            for j in range(len(sym_train[i])):
                similar_sets[i].append(j)

        for idx, sym_batch in enumerate(sym_train):
            if len(sym_batch) <= 2 or len(sym_batch[0]) <= 2: continue
            batch_sets = [set(sym_set) for sym_set in sym_batch]
            for i in range(len(batch_sets)):
                max_intersection = 0
                for j in range(len(batch_sets)):
                    if i == j: continue
                    if len(batch_sets[i] & batch_sets[j]) > max_intersection:
                        max_intersection = len(batch_sets[i] & batch_sets[j])
                        similar_sets[idx][i] = j

        return similar_sets
