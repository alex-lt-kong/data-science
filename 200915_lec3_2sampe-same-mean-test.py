#!/usr/bin/python3

# refer to page 221 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD
# two-sample independent t-test


import numpy as np
from scipy.stats import ttest_ind # ind stands for independence

import importlib

#moduleName = input('Enter module name:')
importlib.import_module('200920-asgmt1_q2_2samp-wilcoxon-rank-sum-and-moods-same-distribution-tests')
importlib.import_module('200920-asgmt1_q1-d_1samp-population-mean-ttest-confidence-interval-for-mean')


def ttest_2samp_for_same_mean(dataset1, dataset2, significance_level: float, equal_variance_assumption: bool):
    """
    null hypothesis that 2 independent samples have identical average (expected) values.
    """
    
    assert significance_level < 0.2
    alpah = significance_level
    # How to determine equal_variance_assumption? Use moods_2sample_same_distribution_test!
    stat, p = ttest_ind(dataset1, dataset2, equal_var = equal_variance_assumption)
    print('Assuming that two datasets are normally distributed and independent, result of ttest_ind:\n p == {}'.format(p), end='')
    if p > significance_level:
        print('  > {}, fail to reject H0 (that the population means of two datasets are equal)'.format(significance_level))
    else:
        print(' <= {}, reject H0 (that the population means of two datasets are equal)'.format(significance_level))

def alternative_2samp_test_for_same_mean(dataset1, dataset2, confidence_level: float):
    # Test if the population means are equal
    
    assert confidence_level > 0.8
    
    import statsmodels.stats.api as sms
    cm = sms.CompareMeans(sms.DescrStatsW(dataset1), sms.DescrStatsW(dataset2))
    #note sms.DescrStatsW().tconfint_mean() and sms.DescrStatsW() are DIFFERENT!
    stat, p = cm.tconfint_diff(alpha = 1 - confidence_level, usevar = 'pooled')
    print('Assuming that two datasets are normally distributed and independent, result of an alternative test:\n p == {}'.format(p), end='')
    if p > significance_level:
        print('  > {}, fail to reject H0 (that the population means of two datasets are equal)'.format(significance_level))
    else:
        print(' <= {}, reject H0 (that the population means of two datasets are equal)'.format(significance_level))


if __name__ == '__main__':
    
    x = np.array([12, 11, 7, 13, 8, 9, 10, 13])
    y = np.array([13, 11, 10, 6, 7, 4, 10])
    moods_2sample_same_distribution_test(x, y, 0.05)
#    ttest_2samp_for_same_mean(x, y, 0.05, True)
    print('\n\n')
    alternative_2samp_test_for_same_mean(x, y, 0.97)
