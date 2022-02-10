import torch
import torch.nn.functional as F
from eval.metrics import multi_label_metric, ddi_rate_score
import numpy as np
from utils.dataset import PKLSet
from tqdm import trange, tqdm
from model.mymodel import Model
from model.radm import RAdam
import argparse
import os
import time

if torch.cuda.is_available():
    torch.cuda.set_device(0)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=64)

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--score_threshold', type=float, default=0.5)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='MIMIC3')

    return parser.parse_known_args()


def evaluate(model, test_loader, n_drugs, device="cpu"):
    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    for step, adm in enumerate(test_loader):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        syms, drugs = torch.tensor(adm[0]).to(device), torch.tensor(adm[2]).to(device)
        # print(syms, drugs)
        # print(syms.shape, drugs.shape)
        scores = model.evaluate(syms, device=device)
        # scores = 2 * torch.softmax(scores, dim=-1) - 1

        y_gt_tmp = np.zeros(n_drugs)
        y_gt_tmp[drugs.cpu().numpy()] = 1
        y_gt.append(y_gt_tmp)

        result = torch.sigmoid(scores).detach().cpu().numpy()
        y_pred_prob.append(result)
        y_pred_tmp = result.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred.append(y_pred_tmp)

        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1
        med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt),
                                                                                 np.array(y_pred),
                                                                                 np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
    # print(y_pred_label)
    ddi_rate = ddi_rate_score(smm_record, path='datasets/MIMIC3/ddi_A_final.pkl')
    # ddi_rate = 0
    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), 1.0 * med_cnt / visit_cnt, ddi_rate


if __name__ == '__main__':
    args, unknown = parse_args()
    print(args)
    # config = Config()
    pklSet = PKLSet(args.batch_size, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    # device = torch.device("cpu")

    model = Model(pklSet.n_sym, pklSet.n_drug, torch.FloatTensor(pklSet.ddi_adj).to(device), pklSet.sym_sets,
                  torch.tensor(pklSet.drug_multihots).to(device), args.embedding_dim).to(device)
    # model.load_state_dict(torch.load('best_ja_at15.pt', map_location=device))
    optimizer = RAdam(model.parameters(), lr=args.lr)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("total number of parameters: ", tot_params)

    best_ja = -np.inf
    for epoch in trange(args.epoch):
        losses, set_idx = 0.0, 0
        model.train()

        for step, (syms, drugs, similar_idx) in tqdm(enumerate(zip(pklSet.sym_train, pklSet.drug_train, pklSet.similar_sets_idx))):
            syms, drugs, similar_idx = torch.tensor(syms).to(device), torch.tensor(drugs).to(device), torch.tensor(similar_idx).to(device)
            # print(syms)
            model.zero_grad()
            optimizer.zero_grad()
            scores, bpr, loss_ddi = model(syms, drugs, similar_idx, device)
            # scores = 2 * torch.softmax(scores, dim=-1) - 1

            sig_scores = torch.sigmoid(scores)
            scores_sigmoid = torch.where(sig_scores == 0, torch.tensor(1.0).to(device), sig_scores)

            bce_loss = F.binary_cross_entropy_with_logits(scores, drugs)
            entropy = -torch.mean(sig_scores * (torch.log(scores_sigmoid) - 1))
            loss = bce_loss + 0.5 * entropy + args.alpha * bpr + args.beta * loss_ddi
            losses += loss.item() / syms.shape[0]
            loss.backward()
            optimizer.step()
            set_idx += 1

        if (epoch + 1) % 5 == 0:
            start = time.time()
            ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = evaluate(model, pklSet.data_eval, pklSet.n_drug, device)
            print('-' * 89)
            print(
                '| end of epoch{:3d}| training time{:5.4f} | ja {:5.4f} | prauc {:5.4f} | avg_p {:5.4f} | avg_recall {:5.4f} | '
                'avg_f1 {:5.4f} | avg_med {:5.4f} | ddi_rate {:5.4f}'.format(epoch, time.time() - start, ja, prauc, avg_p, avg_r,
                                                                             avg_f1, avg_med, ddi_rate))
            print('-' * 89)

            if ja > best_ja:
                torch.save(model.state_dict(), 'best_ja_at15.pt')
                best_ja = ja

    model.load_state_dict(torch.load('best_ja_at15.pt', map_location=device))
    ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = evaluate(model, pklSet.data_eval, pklSet.n_drug, device)
    print('-' * 89)
    print(
        '| best ja {:5.4f} | prauc {:5.4f} | avg_p {:5.4f} | avg_recall {:5.4f} | '
        'avg_f1 {:5.4f} | avg_med {:5.4f} | ddi_rate {:5.4f}'.format(ja, prauc, avg_p,
                                                                     avg_r,
                                                                     avg_f1, avg_med, ddi_rate))
    print('-' * 89)
