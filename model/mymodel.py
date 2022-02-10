import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.sparse as sp
from model.aggregation import Attention
import time

class Model(nn.Module):
    def __init__(self, n_sym, n_drug, ddi_adj, sym_sets, drug_multihots, embed_dim=64, dropout=0.4):
        super(Model, self).__init__()
        self.n_sym, self.n_drug = n_sym, n_drug
        self.embed_dim, self.dropout = embed_dim, dropout
        self.sym_sets, self.drug_multihots = sym_sets, drug_multihots
        self.sym_embeddings = nn.Embedding(self.n_sym, self.embed_dim)
        self.drug_embeddings = nn.Embedding(self.n_drug, self.embed_dim)
        self.sym_agg = Attention(self.embed_dim)
        self.sym_counts = None
        self.tensor_ddi_adj = ddi_adj
        self.sparse_ddi_adj = sp.csr_matrix(ddi_adj.detach().cpu().numpy())
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, syms, drugs, similar_idx, device="cpu"):
        '''
        :param syms: [batch_size, sym_set_size]
        :param drugs: [batch_size, num_drugs]
        :param device: 'cpu' or 'gpu
        :param similar_idx: [batch_size]
        :return:
        '''

        all_drugs = torch.tensor(range(self.n_drug)).to(device)
        sym_embeds, all_drug_embeds = self.sym_embeddings(syms.long()), self.drug_embeddings(all_drugs)
        s_set_embeds = self.sym_agg(sym_embeds)
        # s_set_embeds = torch.mean(sym_embeds, dim=1)
        all_drug_embeds = all_drug_embeds.unsqueeze(0).repeat(s_set_embeds.shape[0], 1, 1)

        scores = torch.bmm(s_set_embeds.unsqueeze(1), all_drug_embeds.transpose(-1, -2)).squeeze(-2)  # [batch_size, n_drug]
        scores_aug, batch_neg = 0.0, 0.0

        neg_pred_prob = torch.sigmoid(scores)
        neg_pred_prob = torch.mm(neg_pred_prob.transpose(-1, -2), neg_pred_prob)  # (voc_size, voc_size)
        batch_neg = 0.00001 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        if syms.shape[0] > 2 and syms.shape[1] > 2:
            scores_aug = self.intraset_augmentation(syms, drugs, all_drug_embeds, similar_idx, device)
            batch_neg += self.intersect_ddi(syms, s_set_embeds, drugs, all_drug_embeds, similar_idx, device)

        return scores, scores_aug, batch_neg

    def evaluate(self, syms, device='cpu'):
        sym_embeds, drug_embeds = self.sym_embeddings(syms.long()), self.drug_embeddings(torch.arange(0, self.n_drug).long().to(device))
        s_set_embed = self.sym_agg(sym_embeds).unsqueeze(0)
        # s_set_embed = torch.mean(sym_embeds, dim=0).unsqueeze(0)
        scores = torch.mm(s_set_embed, drug_embeds.transpose(-1, -2)).squeeze(0)

        return scores

    def intraset_augmentation(self, syms, drugs, all_drug_embeds, similar_idx, device='cpu'):
        selected_drugs = drugs[similar_idx]
        r = torch.tensor(range(drugs.shape[0])).to(device).unsqueeze(1)
        sym_multihot, selected_sym_multihot = torch.zeros((drugs.shape[0], self.n_sym)).to(device), \
                                              torch.zeros((drugs.shape[0], self.n_sym)).to(device)
        sym_multihot[r, syms], selected_sym_multihot[r, syms[similar_idx]] = 1, 1

        common_sym = sym_multihot * selected_sym_multihot
        common_sym_sq = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds = self.sym_embeddings(torch.tensor(range(self.n_sym)).to(device)).unsqueeze(0).expand_as(common_sym_sq)
        common_sym_embeds = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        common_drug, diff_drug = drugs * selected_drugs, drugs - selected_drugs
        diff_drug[diff_drug == -1] = 1

        common_drug_sum, diff_drug = torch.sum(common_drug, -1, True), torch.sum(diff_drug, -1, True)
        common_drug_sum[common_drug_sum == 0], diff_drug[diff_drug == 0] = 1, 1

        scores = torch.bmm(common_set_embeds.unsqueeze(1), all_drug_embeds.transpose(-1, -2)).squeeze(1)
        scores = F.binary_cross_entropy_with_logits(scores, common_drug)

        return scores

    def intersect_ddi(self, syms, s_set_embed, drugs, all_drug_embeds, similar_idx, device='cpu'):
        selected_drugs = drugs[similar_idx]
        r = torch.tensor(range(drugs.shape[0])).to(device).unsqueeze(1)
        sym_multihot, selected_sym_multihot = torch.zeros((drugs.shape[0], self.n_sym)).to(device), \
                                              torch.zeros((drugs.shape[0], self.n_sym)).to(device)
        sym_multihot[r, syms], selected_sym_multihot[r, syms[similar_idx]] = 1, 1

        common_sym = sym_multihot * selected_sym_multihot
        common_sym_sq = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds = self.sym_embeddings(torch.tensor(range(self.n_sym)).to(device)).unsqueeze(0).expand_as(
            common_sym_sq)
        common_sym_embeds = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        diff_drug = drugs - selected_drugs
        diff_drug_2 = torch.zeros_like(diff_drug)
        diff_drug_2[diff_drug == -1], diff_drug[diff_drug == -1] = 1, 0

        diff_drug_exp, diff2_exp = diff_drug.unsqueeze(1), diff_drug_2.unsqueeze(1)
        diff_drug = torch.sum(diff_drug, -1, True)
        diff_drug_2 = torch.sum(diff_drug_2, -1, True)
        diff_drug[diff_drug == 0] = 1
        diff_drug_2[diff_drug_2 == 0] = 1
        diff_drug_embed = torch.bmm(diff_drug_exp.float(), all_drug_embeds).squeeze() / diff_drug
        diff2_embed = torch.bmm(diff2_exp.float(), all_drug_embeds).squeeze() / diff_drug_2

        diff_score = torch.sigmoid(common_set_embeds * diff_drug_embed.float())
        diff2_score = torch.sigmoid(common_set_embeds * diff2_embed.float())
        score_aug = 0.0001 * torch.sum(diff2_score * diff_score)

        return score_aug
