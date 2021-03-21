#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:32:06 2020

@author: mamsds
"""
# question 2
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, anderson
import statsmodels.stats.api as sms
import statsmodels.api as sm
from scipy.stats import wilcoxon

import os
import pathlib

def moods_2sample_same_distribution_test(dataset1, dataset2, significance_level):
    """
    Mood’s two-sample test for scale parameters is a test for the null hypothesis that two
    samples are drawn from the same distribution with the same scale parameter.
    Wikipedia: It tests the null hypothesis that the medians of the populations from which two or more samples are drawn are identical.

    H0: σ1^2 / σ2^2 == 1

    """

    z, p = scipy.stats.mood(dataset1, dataset2)

    assert significance_level < 0.2

    print('Assuming that both datasets are from normal distributions, result from Mood’s two-sample test: p == {}, '.format(p), end='')
    if p > significance_level:
        print('variances of two dataset look insignificantly different (fail to reject H0)')
    else:
        print('variances of two dataset look significantly different (reject H0)')
    print('A more detailed null hypothesis: two samples are drawn from the same distribution with the same scale parameter')

def wilcoxon_rank_sum_2sample_median_test(dataset1, dataset2, significance_level):
    """
    My understanding is that the key difference between wilcoxon_rank_sum_2samples_test and moods_2samples_median_test is that
    # mood's test assumes normal distribution (pages 218 and 220)
    while wilcoxon's rank sum test does not need this assumption
    (page 222 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf).
    """

    assert significance_level < 0.2

    from scipy.stats import ranksums
    stat, p = ranksums(dataset1, dataset2)
    print('WithOUT the assumption that both datasets are from normal distributions, result from Wilcoxon rank-sum test: p == {}'.format(p), end='')
    if p > significance_level:
        print(' > {}, medians from two samples look INsignificantly different (fail to reject H0)'.format(significance_level))
    else:
        print(' <= {}, medians from two samples look significantly different (reject H0)'.format(significance_level))
    print('Note: It tests whether two samples are likely to derive from the same population. Some investigators interpret this test as comparing the medians between the two populations.')

def students_t_test_for_population_mean(dataset, population_mean: float, significance_level: float):
    """ One-sample T Test (also called a two-sided test):
    SciPy Manual: This is a two-sided test for the null hypothesis that the expected value (mean) of
    a sample of independent observations a is equal to the given population mean, popmean.

    According to 191 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD, ttest_1samp requires the
    assumption of normal distribution

    refer to pages 190 - 192 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD

    """
    assert np.isnan(dataset).any() == False
    assert significance_level < 0.2

    print('Assuming that the population is normally distributed, Student\'s t-test result:')
    alpha = significance_level
    confidence_level = 1 - significance_level
    assert alpha == significance_level
    assert confidence_level == 1 - significance_level

    from scipy.stats import ttest_1samp
    stat, p = ttest_1samp(a=dataset, popmean=population_mean)
    print('p == {}'.format(round(p, 10)), end='')
    if p > significance_level:
        print(' > {}, do not reject H0 (that population mean (μ) == {})'.format(population_mean, population_mean))
    else:
        print(' <= {} reject H0 (that population mean (μ) == {})'.format(population_mean, population_mean))
    print('sample mean: {}'.format(np.array(dataset).mean()))

def calculate_confidence_interval_for_weight_mean(dataset, confidence_level: float):
    """
    arguments:
    confidence_level -- In plain English, a Confidence Interval is a range of values we are fairly sure our true value lies in.
     The level of "fair surety" is called confidence level significance level (alpha) + confidence level = 1
     alpha is also the threshold of pvalue.
    """

    assert np.isnan(dataset).any() == False
    assert confidence_level > 0.8

    ci_lower_bound, ci_upper_bound = sms.DescrStatsW(dataset).tconfint_mean(alpha=(1 - confidence_level))
    print('Assuming that the population is normally distributed, ', end = '')
    print('C.I. with {}% confidence: [{}, {}]'.format(confidence_level * 100, round(ci_lower_bound, 10), round(ci_upper_bound, 10)))
    return ci_lower_bound, ci_upper_bound

def shapiro_wilk_test(significance_level: float, dataset: np.ndarray):
    """
    Refer to pages 228-229 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    """

    assert significance_level < 0.2
    assert len(dataset) >= 3 and len(dataset) <= 5000
    assert np.isnan(dataset).any() == False

    '''WARNING: According to page 228 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf,
        the allowed dataset length for Shapiro-Wilk test is [3, 5000]
    '''
    alpha = significance_level
    confidence_level = 1 - significance_level

    assert alpha == significance_level and confidence_level == 1 - significance_level

    stat, p = shapiro(dataset)
    print('Result from Shairo-Wilk test: p == {}'.format(p), end='')
    if p > significance_level:
        print(' (p > {}), sample looks Gaussian (fail to reject H0)'.format(significance_level))
    else:
        print(' (P <= {}), sample does NOT look Gaussian (reject H0)'.format(significance_level))


def kolmogorov_smirnov_test(significance_level: float,
                            dataset: np.ndarray,
                            standardized: bool):
    """
    Refer to pages 228-229 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    """
    alpha = significance_level

    assert significance_level < 0.2
    assert np.isnan(dataset).any() == False
    assert alpha == significance_level
#    assert standardized == True
    # According to page 229 of lec2-3_ch1_introduction_with_mjd.pdf, dataset for KS test has to be standardized.
    if standardized == False:
        dataset = (dataset - np.mean(dataset))/np.std(dataset)
        print('IMPORTANT: dataset will be STANDARDIZED if it has not been done!')

    stat, p = kstest(dataset, cdf = 'norm')
    print('Results from Kolmogorov-Smirnov test: stat == {}, p == {},'.format(stat, p), end='')
    if p > significance_level:
        print(' (p > {}), sample looks Gaussian (fail to reject H0)'.format(significance_level))
    else:
        print(' (p <= {}), sample does NOT look Gaussian (reject H0)'.format(significance_level))

def anderson_darling_test(dataset: np.ndarray):
    """
    Refer to pages 228-229 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf

    anderson-darling test does not provide a concrete p value
    """
    assert np.isnan(dataset).any() == False
    assert len(dataset) > 7

    stat, critical_values, significance_levels = anderson(x = dataset, dist = 'norm')
    print('Results from Anderson-Darling test:\nstats: {}\ncritical_values: {}\nsignificance_levels: {}'.format(stat, critical_values, significance_levels))

    print('Results interpretation:')
    for i in range(len(critical_values)):
        if stat > critical_values[i]:
            print('sample does NOT look Gaussian (reject H0) (p < alpha == {})'.format(significance_levels[i] / 100))
        else:
            print('sample looks Gaussian (fail to reject H0) (p > alpha == {})'.format(significance_levels[i] / 100))

def wilcoxon_signed_rank_median_test(dataset, hypothesized_median: float, significance_level: float, correction: bool):
    """
    Basically you try to determine the probability that the median of a dataset is hypothesized_median.
    """
    stat1, p1 = wilcoxon(dataset - hypothesized_median, correction = correction)
    stat2, p2 = wilcoxon(dataset, np.array([hypothesized_median] * len(dataset)), correction = correction)

    assert stat1 == stat2 and p1 == p2

    print('wihtOUT the normal distribution assumption, Wilcoxon signed-rank test result:')
    print('p == {}, '.format(p1), end = '')
    if p1 > significance_level:
        print('do NOT reject H0 (that median == {})'.format(hypothesized_median))
    else:
        print('reject H0 (that median == {})'.format(hypothesized_median))


def question_1_ab(da):
    #df = pd.DataFrame(da['return'])
    da['log_rt'].plot.box(vert = True)

    import matplotlib.pyplot as plt
    plt.show()

    print('sample mean: {}'.format(da['log_rt'].mean()))
    print('sample variance: {}'.format(da['log_rt'].var()))
    print('sample skew: {}'.format(da['log_rt'].skew()))
    print('sample kurtosis: {}'.format(da['log_rt'].kurtosis()))

def question_1_c(da_sp, da_ge):
    sm.qqplot(data=da_sp.log_rt, dist=scipy.stats.distributions.norm, line='45')
    plt.title("""qqplot - da_sp.log_rt
    This is wrong because statsmodels.apiqqplot() scipy.stats.distributions.norm is defined as
    a standard normal distribution, which has a mean of zero and standard deviation of 1""")
    plt.show()
    sm.qqplot(data=da_sp.log_rt_standardized, dist=scipy.stats.distributions.norm, line='45')
    plt.title("qqplot - da_sp.log_rt_standardized")
    plt.show()
    sm.qqplot(data=da_ge.log_rt, dist=scipy.stats.distributions.norm, line='45')
    plt.title("qqplot - da_ge.log_rt")
    plt.show()
    sm.qqplot(data=da_ge.log_rt_standardized, dist=scipy.stats.distributions.norm, line='45')
    plt.title("qqplot - da_ge.log_rt_standardized")
    plt.show()

def question_1_d(da):


    students_t_test_for_population_mean(da.log_rt[1:], 0, 0.05)
    students_t_test_for_population_mean(da.log_rt_standardized[1:], 0, 0.05)
    print('\n\n')

    calculate_confidence_interval_for_weight_mean(dataset = da.log_rt[1:], confidence_level = 0.95)

def question_1_e(da):

    shapiro_wilk_test(significance_level = 0.05, dataset = da.log_rt[1:])
    shapiro_wilk_test(significance_level = 0.05, dataset = da.log_rt_standardized[1:])

    print('\n')
    kolmogorov_smirnov_test(significance_level = 0.05, dataset = da.log_rt_standardized[1:], standardized = True)
    kolmogorov_smirnov_test(significance_level = 0.05, dataset = da.log_rt[1:], standardized = False)
 #   kolmogorov_smirnov_test(significance_level = 0.05, dataset = da.log_rt[1:], standardized = True)
    
    print('\n')
    anderson_darling_test(dataset = da.log_rt_standardized[1:])
    anderson_darling_test(dataset = da.log_rt[1:])

    print('\n\n\n')
    nd = np.random.normal(loc = 0, scale = 3, size = 1000)
    shapiro_wilk_test(significance_level = 0.05, dataset = nd)
    kolmogorov_smirnov_test(significance_level = 0.05, dataset = nd, standardized = True)
    anderson_darling_test(dataset = nd)

    print('\n\n\n')
    ud = np.random.uniform(low = 0, high = 100, size = 1000)
    shapiro_wilk_test(significance_level = 0.05, dataset = ud)
    kolmogorov_smirnov_test(significance_level = 0.05, dataset = ud, standardized = True)
    anderson_darling_test(dataset = ud)

def question_1_f(da):

    rd = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    rd_standardized = (rd - rd.mean())/rd.std()
    print(rd)
    print(rd_standardized) # note that standardization can change mean to zero, but it cannot change median to zero.

    wilcoxon_signed_rank_median_test(dataset = rd, hypothesized_median = 2, significance_level = 0.05, correction = True)
    wilcoxon_signed_rank_median_test(dataset = rd, hypothesized_median = 1, significance_level = 0.05, correction = True)
    wilcoxon_signed_rank_median_test(dataset = rd, hypothesized_median = 7, significance_level = 0.05, correction = True)
    wilcoxon_signed_rank_median_test(dataset = rd, hypothesized_median = 8, significance_level = 0.05, correction = False)
    wilcoxon_signed_rank_median_test(dataset = rd_standardized, hypothesized_median = 0, significance_level = 0.05, correction = True)
    wilcoxon_signed_rank_median_test(dataset = rd_standardized, hypothesized_median = 0, significance_level = 0.05, correction = False)
    wilcoxon_signed_rank_median_test(dataset = da.log_rt, hypothesized_median = 0, significance_level = 0.05, correction = True)
    wilcoxon_signed_rank_median_test(dataset = da.log_rt, hypothesized_median = 0, significance_level = 0.05, correction = False)
    wilcoxon_signed_rank_median_test(dataset = da.log_rt[1:], hypothesized_median = 0, significance_level = 0.05, correction = True)
    wilcoxon_signed_rank_median_test(dataset = da.log_rt[1:], hypothesized_median = 0, significance_level = 0.05, correction = False)

def question_2(da_dj, da_sp):
    alpha = 0.05
    significance_level = alpha
    confidence_level = 1 - significance_level
    assert confidence_level == 1 - significance_level


    plt.plot(list(range(len(da_dj['close']))), da_dj.log_rt, color = 'red', linewidth=0.5)
    plt.plot(list(range(len(da_sp['close']))), da_sp.log_rt, color = 'green', linewidth=0.5)
    plt.show()

    moods_2sample_same_distribution_test(da_dj.log_rt, da_sp.log_rt, 0.05)
    print('\n')
    print('\n')

    wilcoxon_rank_sum_2sample_median_test(da_sp.log_rt[1:], da_dj.log_rt[1:], 0.05)
    print('\n')
    wilcoxon_rank_sum_2sample_median_test(da_dj.log_rt, da_sp.log_rt, 0.05)
    print('\n')
    wilcoxon_rank_sum_2sample_median_test(da_dj.log_rt_standardized, da_sp.log_rt_standardized, 0.05)
    print('\n')
    wilcoxon_rank_sum_2sample_median_test(da_dj.log_rt_standardized[1:], da_sp.log_rt_standardized[1:], 0.05)

def main():

    parent_dir = pathlib.Path(__file__).parent.absolute()

    da_dj = pd.read_csv(os.path.join(parent_dir, 'geDJ_with-headers.txt'), sep='\s+')
    rt = np.diff(np.log(da_dj['close']))
    rt = np.append(np.nan, rt)
    da_dj['log_rt'] = rt
    da_dj['log_rt_standardized'] = (da_dj.log_rt - da_dj.log_rt.mean())/da_dj.log_rt.std()

    da_sp = pd.read_csv(os.path.join(parent_dir, 'sp500_with-headers.txt'), sep='\s+')
    rt = np.diff(np.log(da_sp['close']))
    rt = np.append(np.nan, rt)
    da_sp['log_rt'] = rt
    da_sp['log_rt_standardized'] = (da_sp.log_rt - da_sp.log_rt.mean())/da_sp.log_rt.std()

    print('question 1(ab)\n')
    question_1_ab(da_sp)
    print('\n\n\n\nquestion 1(c)\n')
    question_1_c(da_sp, da_dj)
    print('\n\n\n\nquestion 1(d)\n')
    question_1_d(da_dj)
    print('\n\n\n\nquestion 1(e)\n')
    question_1_e(da_dj)
    print('\n\n\n\nquestion 1(f)\n')
    question_1_f(da_dj)
    print('\n\n\n\nquestion 2\n')
    question_2(da_dj, da_sp)

if __name__ == '__main__':
    main()
