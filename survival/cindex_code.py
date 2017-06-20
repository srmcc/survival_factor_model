"""
Survival Factor Model, Copyright (C) 2017,  Shannon McCurdy and Regents of the University of California.

Contact:  Shannon McCurdy, smccurdy@berkeley.edu.

    This file is part of Survival Factor Model.

    Survival Factor Model is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Survival Factor Model is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Survival Factor Model program.  If not, see <http://www.gnu.org/licenses/>.
"""


from __future__ import division
import numpy as np
import pandas as pd
import scipy
import scipy.stats


def ordergraph(tE, Delta):
    data = pd.DataFrame(
        np.vstack([tE, Delta, scipy.stats.rankdata(tE)]).T,
        columns=['tE', 'Delta', 'rank'])
    ordg = np.empty((tE.shape[1], tE.shape[1]))
    for i in list(data.index):
        ordg[i, :] = (1 * (data.loc[:, 'rank'] > data.loc[i, 'rank']) *
                      data.loc[i, 'Delta'])
    return ordg


def cindex(tE, Delta, tEp, Deltap):
    ordg = ordergraph(tE, Delta)
    ordgp = ordergraph(tEp, Deltap)
    c = np.sum(ordg * (ordg == ordgp)) / np.sum(ordg)
    return c


    ##cindex with ties kang 2015
def ksign(yi, yj):
    ##this is kcsign with delta y=1   (the estimate)
    return 1 * (yi >= yj) - 1 * (yi <= yj)


def kcsign(xi, deltai, xj, deltaj):
    ###the truth (has censoring)
    return 1 * (xi >= xj) * deltaj - 1 * (xi <= xj) * deltai


def ktxy(xi, deltai, xj, deltaj, yi, deltayi, yj, deltayj):
    ##for ktxx put in ktxy(xi, deltai, xj, deltaj, xi, deltai, xj, deltaj)
    ##for ktxy put in ktxy(xi, deltai, xj, deltaj, yi, 1, yj, 1)
    return kcsign(xi, deltai, xj, deltaj) * kcsign(yi, deltayi, yj, deltayj)


def cindex_ties(tE, Delta, tE_est):
    ##no censoring on the estimated tE_est
    nsamp = tE.shape[1]
    ktxyav = 0
    ktxxav = 0
    # hmm= np.zeros((nsamp, nsamp))
    for i in range(nsamp):
        for j in range(nsamp):
            if j != i:
                #print(i, j)
                ktxyav = ktxyav + ktxy(tE[:, i], Delta[:, i], tE[:, j],
                                       Delta[:, j], tE_est[:, i], 1,
                                       tE_est[:, j], 1)
                ktxxav = ktxxav + ktxy(tE[:, i], Delta[:, i], tE[:, j],
                                       Delta[:, j], tE[:, i], Delta[:, i],
                                       tE[:, j], Delta[:, j])
                # hmm[i, j] = ktxy(tE[:, i] , Delta[:, i] , tE[:, j] , Delta[:, j],
                #                      tE[:, i] , Delta[:, i] , tE[:, j] , Delta[:, j])
                # print(ktxyav, ktxxav)
    txy = ktxyav / (nsamp * (nsamp - 1))
    txx = ktxxav / (nsamp * (nsamp - 1))
    vartxy = covtxytuz(tE, Delta, tE_est, np.ones(tE.shape), tE, Delta, tE_est,
                       np.ones(tE.shape))
    vartxx = covtxytuz(tE, Delta, tE, Delta, tE, Delta, tE, Delta)
    covtxytxx = covtxytuz(tE, Delta, tE_est, np.ones(tE.shape), tE, Delta, tE,
                          Delta)
    var = (vartxy / (txx**2) - 2 * txy * covtxytxx / (txx**3) +
           (txy**2) * vartxx / (txx**4)) / 4
    return 1 / 2 * (
        ktxyav / ktxxav + 1), var, txx, txy, vartxx, covtxytxx  #, hmm


def covtxytuz(tE, Delta, tE_est, Delta_tE_est, tE_U, Delta_U, tE_est_Z,
              Delta_tE_est_Z):
    """
    first two arguementsare for x, next two are y, next two are u, last two are z.
    y and z should not be censored (they are predictions), therefore those deltas should be ones.
    var txx:  covtxytuz(tE, Delta, tE, Delta, tE, Delta, tE, Delta )
    var txy:  covtxytuz(tE, Delta, tE_est, Delta_tE_est, tE, Delta, tE_est, Delta_tE_est)
    cov (txx, txy) : covtxytuz(tE, Delta, tE, Delta, tE, Delta,  tE_est, Delta_tE_est)
    cov (txy, txz) :covtxytuz(tE, Delta, tE_est, Delta_tE_est, tE, Delta, tE_est_Z, Delta_tE_est_Z)

    """
    nsamp = tE.shape[1]
    kcovtxytuz_1 = 0
    kcovtxytuz_2 = 0
    kcovtxytuz_3 = 0
    kcovtxytuz_4 = 0
    for i in range(nsamp):
        for j in range(nsamp):
            kcovtxytuz_1 = (kcovtxytuz_1 + ktxy(
                tE[:, i], Delta[:, i], tE[:, j], Delta[:, j], tE_est[:, i],
                Delta_tE_est[:, i], tE_est[:, j], Delta_tE_est[:, j]) * ktxy(
                    tE_U[:, i], Delta_U[:, i], tE_U[:, j], Delta_U[:, j],
                    tE_est_Z[:, i], Delta_tE_est_Z[:, i], tE_est_Z[:, j],
                    Delta_tE_est_Z[:, j]))
            kcovtxytuz_3 = (kcovtxytuz_3 + ktxy(
                tE[:, i], Delta[:, i], tE[:, j], Delta[:, j], tE_est[:, i],
                Delta_tE_est[:, i], tE_est[:, j], Delta_tE_est[:, j]))
            kcovtxytuz_4 = (kcovtxytuz_4 + ktxy(
                tE_U[:, i], Delta_U[:, i], tE_U[:, j], Delta_U[:, j],
                tE_est_Z[:, i], Delta_tE_est_Z[:, i], tE_est_Z[:, j],
                Delta_tE_est_Z[:, j]))
            for jp in range(nsamp):
                kcovtxytuz_2 = (kcovtxytuz_2 + ktxy(
                    tE[:, i], Delta[:, i], tE[:, j], Delta[:, j], tE_est[:, i],
                    Delta_tE_est[:, i], tE_est[:, j], Delta_tE_est[:, j]) *
                                ktxy(tE_U[:, i], Delta_U[:, i], tE_U[:, jp],
                                     Delta_U[:, jp], tE_est_Z[:, i],
                                     Delta_tE_est_Z[:, i], tE_est_Z[:, jp],
                                     Delta_tE_est_Z[:, jp]))
    return ((4 * kcovtxytuz_2 - 2 * kcovtxytuz_1 - 2 * (2 * nsamp - 3) /
             (nsamp * (nsamp - 1)) * kcovtxytuz_3 * kcovtxytuz_4) /
            (nsamp * (nsamp - 1) * (nsamp - 2) * (nsamp - 3)))


def compare_two_cindex(tE, Delta, tE_est1, tE_est2, alpha):
    c1, var1, txx1, txy1, vartxx1, covtxytxx1 = cindex_ties(tE, Delta, tE_est1)
    c2, var2, txx2, txy2, vartxx2, covtxytxx2 = cindex_ties(tE, Delta, tE_est2)
    covtxytxz = covtxytuz(tE, Delta, tE_est1, np.ones(tE.shape), tE, Delta,
                          tE_est2, np.ones(tE.shape))
    cov = (covtxytxz / (txx1**2) + (-txy1 / (txx1**3)) * covtxytxx2 +
           (-txy2 /
            (txx1**3)) * covtxytxx1 + txy1 * txy2 * vartxx1 / (txx1**4))
    print(cov, (covtxytxz / (txx1**2), (-txy1 / (txx1**3)) * covtxytxx2,
                (-txy2 /
                 (txx1**3)) * covtxytxx1, txy1 * txy2 * vartxx1 / (txx1**4)))
    var = var1 + var2 - 2 * cov / 4
    z = (c1 - c2) / np.sqrt(var)
    p = 2 * scipy.stats.norm.sf(np.abs(z), loc=0, scale=1)
    ##two tailed p - value
    return c1, var1, c2, var2, var, p


def c_index_sig_no_censoring(c, n1, n2):
    ##with no censoring, the c - index is equivalent to the u statistic of man - witney (Koziol and Jia, 2009)
    ## with large (>20) sample sizes, the u - statistic distribution under the null is well - approximated by the normal distribution
    ##this code calcluates the p - value under the normal approximation.
    mu = (n1 * n2) / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    u = c * n1 * n2
    ##c is the area under the receiver operating characteristic curve (auc, roc)  wiki
    z = (u - mu) / sigma
    p = 2 * scipy.stats.norm.sf(np.abs(z), loc=0, scale=1)
    ##sf measures sf = 1 - CDF
    print('mu, sigma, u, z', mu, sigma, u, z)
    return p
    ##two tailed p - value


def random_expected_cindex(nsamp, ntrain):
    expected_c_test = 0
    tE_test = np.random.choice(range(nsamp), size=nsamp, replace=False)
    tE_test = np.reshape(tE_test, (1, nsamp))
    for i in range(0, ntrain):
        est_test = np.random.choice(range(nsamp), size=nsamp, replace=False)
        est_test = np.reshape(est_test, (1, nsamp))
        estdelta_test = np.ones((1, nsamp))
        Delta_test = np.ones((1, nsamp))
        c_test = cindex(tE_test, Delta_test, est_test, estdelta_test)
        expected_c_test = expected_c_test + c_test
        print('expected c test', expected_c_test, c_test)
    expected_c_test = expected_c_test / ntrain
    return expected_c_test
