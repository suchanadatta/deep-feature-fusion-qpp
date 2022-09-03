from scipy import stats
import sys, math


def reportRankCorr(x, y):
    corr, _ = stats.pearsonr(x, y)
    rho, pval = stats.spearmanr(x, y)
    tau, pval = stats.kendalltau(x, y)
    print('P-r = {:,.4f}, S-rho = {:.4f}, K-tau = {:.4f}\n'.format(corr, rho, tau))

    return corr, rho, tau

#also compute the avg shift in ranks
def RMSE(x, y):
    i = 1
    rmse = 0
    for x_i in x:
        j = y.index(x_i)
        rmse += abs(i-j)
        i+= 1
    print('RMSE = {:.4f}'.format(rmse/len(x)))

    return rmse/len(x)


# if len(sys.argv) < 1:
#     print ('usage: python eval_rank_corr.py <values file>')
#     sys.exit(0)
# values_file = open(sys.argv[1], 'r')
# lines = values_file.readlines()
# x = []
# y = []
# for line in lines:
#     tokens = line.split()
#     x.append(tokens[0])
#     y.append(tokens[1])
# values_file.close()
# reportRankCorr(x, y)
# RMSE(x, y)