import argparse
import numpy as np
from sklearn.metrics import accuracy_score

def min_max_norm(input, output):
    out_file = open(output, 'a')
    input_score = {}
    qpp_norm = ''
    with open(input) as f:
        content = f.readlines()
        for x in content:
            l = x.strip().split('\t')
            input_score[l[0]] = float(l[1])
        print(input_score)
        for qid, score in input_score.items():
            norm = round((float(score) - min(input_score.values())) /
                         (max(input_score.values()) - min(input_score.values())), 5)
            qpp_norm += str(qid) + '\t' + str(norm) + '\n'
        print(qpp_norm)
        out_file.write(qpp_norm)
        out_file.close()

def make_relative_measure(qpp_file, qid_pairs,outfile):
    out_file = open(outfile, 'a')
    qpp_score = {}
    res = ''
    with open(qpp_file) as f:
        content = f.readlines()
        for x in content:
            l = x.strip().split('\t')
            qpp_score[l[0]] = float(l[1])
        print(qpp_score)
    with open(qid_pairs) as f:
        content = f.readlines()
        for x in content:
            l = x.strip().split('\t')
            if qpp_score.get(l[0]) >= qpp_score.get(l[1]):
                res += str(l[0]) + '\t' + str(l[1]) + '\t1\n'
            else: res += str(l[0]) + '\t' + str(l[1]) + '\t0\n'
    out_file.write(res)
    out_file.close()

def measure_accuracy(gt, pred):
    gt_score = np.genfromtxt(gt, delimiter='\t')
    actual = gt_score[:, 2:]
    pred_score = np.genfromtxt(pred, delimiter='\t')
    predict = pred_score[:, 2:]
    score = accuracy_score(actual, predict)
    print('Accuracy : ', round(score, 4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-scores', default='../results/baseline/trec7.uef_wig')
    parser.add_argument('--norm-scores', default='../results/baseline/trec7.uef_wig.norm')
    parser.add_argument('--input-qid-pair', default='../data/test_pair_ndcg20_trec7') # 2-col tsv <qid-a \t qid-b>
    parser.add_argument('--out-qid-pair-pred', default='../results/baseline/trec7_pair.ndcg20.uef_wig') # 3-col tsv <qid-a \t qid-b \t 0/1>
    parser.add_argument('--qid-pair-gt', default='../data/test_pair_gt_ndcg20_trec7') # 3-col tsv <qid-a \t qid-b \t 0/1>
    args = parser.parse_args()

    # for min-max normalization of the computed qpp scores
    min_max_norm(args.input_scores, args.norm_scores)

    # for measuring pairwise accuracy of baselines
    make_relative_measure(args.norm_scores, args.input_qid_pair, args.out_qid_pair_pred)

    # measure accuracy
    measure_accuracy(args.qid_pair_gt, args.out_qid_pair_pred)

if __name__ == '__main__':
    main()