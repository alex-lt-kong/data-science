#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:51:38 2020

@author: mamsds
"""

from typing import Dict
from typing import List
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

import importlib
asgmt1 = importlib.import_module('200920_asgmt1_normality-and-other-tests')

# NOTE:
# Some core functions demostrated on Dr. Yu's lecture have been greatly modified in 201018_asgmt3.py
# Refer to that source code file for more updated versions!

def multicollinearity_detector(dataset: np.ndarray, verbosity: int):

    if verbosity >= 1:
        print('Detecting multicollinearity using variance inflation factor:')
    for i in range(dataset.shape[1]):
        vif = variance_inflation_factor(dataset.values, i)
        if verbosity >= 1:
            print('[{}] vif == {}: '.format(dataset.columns[i], vif), end = '')
        if verbosity >= 2:
            if vif > 10:
                print('multicollinearity is SERIOUS')
            else:
                print('multicollinearity is NOT that serious')
        elif verbosity >= 1:
            print('')

    if verbosity >= 3:
        print('Notes:')
        print('1. The term multicollinearity refers to the effect, on the precision of the LS estimators of the regression coefficients, of two or more of the explanatory variables being highly correlated;')
        print('2. Pearson correlation matrix among all explanatory variables ONLY shows the association between any two variables, ignoring other explanatory variables;')
        print('3. We can use  variance inflation factor (VIF) to measure the level of multicollinearity. A large VIF indicates a sign of serious multicollinearity. No rule of thumbs on numerical values is foolproof , but it is generally believed that if any VIF exceeds 10 , there may be a serious problem with multicollinearity .')

def durbin_watson_autocorrelation_test(model):

    dw = np.sum(np.diff(model.resid.values) ** 2.0) / model.ssr
    print('Durbin-Watson statistic: {}'.format(dw), end = '')
    if dw == 4:
        print(', implying a PERFECT NEGATIVE autocorrelation!')
    elif dw < 4 and dw > 2:
        print(', implying a negative autocorrelation')
    elif dw == 2:
        print(', implying NO autocorrelation AT ALL!')
    elif dw > 0 and dw < 2:
        print(', implying a positive autocorrelation')
    elif dw == 0:
        print(', implying a PERFECT POSITIVE autocorrelation')
    else:
        print(', WTF??!! This is IMPOSSIBLE!!!')
    print('Notes:')
    print('1. Durbin-Watson statistic should be between 0 to 4. 0 means a perfect positive autocorrelation, 2 means no autocorrelation and 4 means a perfect negative autocorrelation.')
    print('2. Independence requires both no autocorrelation and normality. Durbin-Watson statistic is only about autocorrelation.')

    # Residual
    et = model.resid
    # 1-lagged et
    et_1 = et.shift(1)

    plt.scatter(et_1[1:], et[1:], s = 1)
    plt.show()

    return dw

def assumption_6_checker(model):

    fig1 = plt.figure(figsize=(20, 10))
    sm.graphics.plot_ccpr_grid(model, fig=fig1)
    plt.show()
    print('Notes:')
    print('1. The above plots are used to check if the true relationship between the mean of dependent variables and independent variables is linear (i.e. if they are linear then it is good);');
    print('2. Given the nature of this checker, no natural language interpretation is generated;')
    print('3. The slope of each plot is more or less similar to the beta value in model\'s summary()')
    print('4. Dr. Yu introduced the mathematial foundation of a remedy called Box-Tidwell transformation. But no code is provided for this method.')

def box_cox_transformer(lambda1: float, x1: np.ndarray, y1: np.ndarray):

    if lambda1 != 0:
        BCy = (y1 ** lambda1 - 1) / lambda1
    else:
        BCy = np.log(y1)

    BCfmodel = ols_model_helper(patsy_formula = 'BCy ~ x1',
                     data = {'BCy': BCy, 'x1': x1},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    #BCfmodel = sm.OLS(BCy, x1).fit()
    # This statement is used by Dr. Yu. My understanding is that both statements should be fine.

    SSE_lambda = ((BCfmodel.predict() - BCy) ** 2).sum()

    loglf = (lambda1 - 1) * ((np.log(y1)).sum()) - len(y1) * (np.log(SSE_lambda / len(y1))) / 2

    return { 'lambda': lambda1, 'loglf': loglf }

def assumptions_12346_checker(model):

    extresid = model.get_influence().resid_studentized_external
    pred = model.predict()

    print('Externally Studentized Residual Plots:')
    plt.scatter(pred, extresid, s = 1)
    plt.xlabel('ŷ (the predicted dependent variable)')
    plt.ylabel('ri (the externally studentized residual(i.e. y[i] - ŷ[i]))')
    plt.axhline(y=2, color='g', linestyle='-.', linewidth = 0.5)
    plt.axhline(y=0, color='r', linestyle='-.', linewidth = 0.5)
    plt.axhline(y=-2, color='g', linestyle='-.', linewidth = 0.5)
    plt.show()
    print('Interpretation of the plot:\nIf:')
    print('(1) The plot has no pattern (for possible patterns and their meanings, refer to pages 28 - 30 of lec5_ch2-variable-selection-and-model-diagnostics.pdf);')
    print('(2) Points are around 0; and')
    print('(3) Most of the points are inside the band |r[i]| <= 2')
    print('We say that Assumptions A1 - A4 are likely to be valid.')


    print('\nNormality Check:')
    sm.qqplot(data=extresid, dist=sp.stats.distributions.norm, line='45')
    plt.show()
    print('If many points do not fall near the red line then it is likely that assumption 4 is invalid.')

    asgmt1.shapiro_wilk_test(significance_level = 0.01, dataset = extresid)
    print('\n')
   # z = (extresid - np.mean(extresid))/np.std(extresid)
    asgmt1.kolmogorov_smirnov_test(significance_level = 0.01, dataset = extresid, standardized = False)
    print('NOTE: For some reasons, the result of KS-Test is not exactly the same as that given by Dr. Yu. Consider round it to 3 decimal places when necessary!\n')
    asgmt1.anderson_darling_test(dataset = extresid)

def assumption_5_checker(model):
    pass

def predict_y_with_model(model,
                         intercept_included: bool,
                         independent_variables: List[List[float]],
                         iv_names: List[str],
                         significance_level = 0.1):
    """
    Format of independent_variables:
      x1  [[1, 2, 3, 4, 5],
      x2   [5, 4, 3, 2, 1],
      x3   [6, 6, 6, 6, 6]]
    """
    ivs = independent_variables
    assert significance_level < 0.2

    print('Predicting y with model (significance_level == {}):'.format(significance_level))

    model_parameters = {}
    if intercept_included:
        model_parameters['intercept'] = 1
    for i in range(len(ivs)):
        model_parameters[iv_names[i]] = ivs[i]
    print(model_parameters)
    predicted_values = model.predict(pd.DataFrame(model_parameters))

    results = []
    for i in range(len(independent_variables[0])):
        tmp = []
        for j in range(len(iv_names)):
            tmp.append(independent_variables[j][i])
        tmp.append(predicted_values[i])
        results.extend([tmp])

    results_pd = pd.DataFrame(results)
    headers = []
    for i in range(len(iv_names)):
        headers.append(iv_names[i])
    headers.append('Predicted Value')

    results_pd.columns = headers

    if intercept_included:
        exogenous_parameters = []

        for i in range(len(ivs[0])):
            temp_iv = [1]
            for j in range(len(ivs)):
                temp_iv.append(ivs[j][i])
            exogenous_parameters.append(temp_iv)
       # print(f'exogenous_parameters: {exogenous_parameters}')
        results_pd['Prediction Std'], results_pd['Lowers'], results_pd['Uppers'] = wls_prediction_std(model, exog = exogenous_parameters, weights = 1, alpha = significance_level)
    else:
        print('Interval prediction not available if intercept is not included.')
    print(results_pd)

def test_betas_at_specific_values(model, matrix_a, vector_b, significance_level = 0.03):
    """
    Formula of H0: ax = b
    Example I (with intercept):
        to test β0 == 0, β1 == 0, β2 == 0
        [[1, 0, 0]
         [0, 1, 0]  * [β0, β1, β2]^T = [0, 0, 0]^T
         [0, 0, 1]]
    Example II (with intercept):
        to test β1 == 1
        [0, 1] * [β0, β1]^T = 1
    Example III (with intercept):
        to test β1 == β2
        [0, 1, -1]* [β0, β1, β2]^T = 0
    Example IV (withOUT intercept):
        to test β1 == β2
        [1, -1]* [β1, β2]^T = 0

    """
    print('Beta values hypothesis test:')
    print('Matrix A: {}'.format(matrix_a))

    print('Vector b: {}'.format(vector_b))
    B = (matrix_a, vector_b)
    p = model.f_test(B).pvalue

    print('Results from F-Test: p == {} '.format(p), end='')
    if p > significance_level:
        print('(p > {}), fail to reject H0 '.format(significance_level))
    else:
        print('(p <= {}), reject H0'.format(significance_level))
    print('NOTE: Due to the complexity of this test, no natural language interpretation is generated. Use these keywords to draw a conclusion: significant/insignificantly different, reject/fail to reject H0.')


def ols_model_helper(patsy_formula: str,
                     data: Dict[str, np.ndarray],
                     with_intercept: bool,
                     verbosity: int,
                     significance_level = 0.05):

    patsy_formula = patsy_formula.replace(' ', '')
    assert (('-1' not in patsy_formula) == with_intercept)
    # This assertion is not 100% reliable. But for the purpose of this course it should be enough


    model = smf.ols(formula = patsy_formula, data = data).fit()
    # https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html
    if verbosity > 0:
        print(model.summary())
        print('\n')

    if verbosity > 1:
        print('                        OLS Regression Results Interpretation')
        print('R-squared: {} (Can also be interpreted as the square of Pearson correlation)\nAdjusted R-squared: {} (Used to compare nested models)'.format(model.rsquared, model.rsquared_adj))
        print('σ_hat: {}'.format(model.mse_resid ** 0.5))

        ci_results = model.conf_int(alpha=significance_level, cols=None)
        print('Confidence Interval at {}% confidence level:'.format((1 - significance_level) * 100))
        for i in range(len(ci_results[0])):
            print('beta[{}]: [{}, {}]'.format(i if with_intercept else i + 1, ci_results[0][i], ci_results[1][i]))

        for i in range(len(model.params)):
            print('beta[{}] == {}'.format(i if with_intercept else i + 1, model.params[i]))


        for i in range(len(model.pvalues)):
            print('p-value[{}] (H0: beta[{}] == 0) == {} ({})'.format(i if with_intercept else i + 1, i if with_intercept else i + 1, model.pvalues[i],
              'do NOT reject H0' if model.pvalues[i] > significance_level else 'reject H0'))
        if with_intercept:
            if model.pvalues[0] > significance_level:
                print('IMPORTANT NOTE: since beta[0] is INsignificantly different from 0, we should DROP the intercept!')
            else:
                print('NOTE: since beta[0] is significantly different from 0, we canNOT drop the intercept!')

    if verbosity > 0:
        print('\n\n')
    return model

def scatter_plot_and_subplot(dataset1: List[float], ds1_name: str, dataset2: List[float], ds2_name: str):

    fig = plt.figure(figsize = (4 * 3, 3 * 3))

    fig.add_subplot(2, 2, 1)
    fig.canvas.set_window_title('scatter_plot_and_subplot')

    # "2,2,1" means "2x2 grid, 1st subplot".
    plt.hist(dataset1, bins = 20, color = 'green', alpha = 0.9)
    plt.title(ds1_name)

    fig.add_subplot(2, 2, 3)
    plt.scatter(dataset1, dataset2, marker = '+', alpha = 0.9)

    fig.add_subplot(2, 2, 4)
    plt.hist(dataset2, bins = 20, orientation = 'horizontal', color = 'red', alpha = 0.9)
    plt.title(ds2_name)

    plt.show()

def spearmans_rank_correlation_and_kendalls_rank_correlation(
        dataset1: List[float],
        ds1_name: str,
        dataset2: List[float],
        ds2_name: str):

    assert np.isnan(dataset1).any() == False
    assert np.isnan(dataset2).any() == False

    significance_level = 0.05

    correlation_coefficient, p = sp.stats.spearmanr(dataset1, dataset2)
    print('Results from Spearman\'s rank test (in particular for ordinal variables): p == {}, '.format(p), end='')
    if p > significance_level:
        print(' (p > {}), fail to reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    else:
        print(' (p <= {}), reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    print('correlation_coefficient == {}, '.format(correlation_coefficient), end = '')
    if correlation_coefficient > 0.5:
        print('indicating a strong POSITIVE correlation between [{}] and [{}]'.format(ds1_name, ds2_name))
    elif correlation_coefficient < -0.5:
        print('indicating a strong NEGATIVE correlation between [{}] and [{}] '.format(ds1_name, ds2_name))
    else:
        print('indicating NO significant correlation between [{}] and [{}]'.format(ds1_name, ds2_name))

    correlation_coefficient, p = sp.stats.kendalltau(dataset1, dataset2)
    print('\nResults from Kendall\'s rank test (in particular for ordinal variables): p == {}'.format(p), end = '')
    if p > significance_level:
        print(' (p > {}), fail to reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    else:
        print(' (p <= {}), reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    print('correlation_coefficient == {}, '.format(correlation_coefficient), end = '')
    if correlation_coefficient > 0.5:
        print('indicating a strong POSITIVE linear between [{}] and [{}]'.format(ds1_name, ds2_name))
    elif correlation_coefficient < -0.5:
        print('indicating a strong NEGATIVE linear between [{}] and [{}] '.format(ds1_name, ds2_name))
    else:
        print('indicating NO significant correlation between [{}] and [{}]'.format(ds1_name, ds2_name))

    print('\nNOTES:')
    print('1. Both tests do NOT need the assumption of bivariate normality')
    print('2. A monotonic relationship is a relationship that does one of the following: (1) as the value of one variable increases, so does the value of the other variable; or (2) as the value of one variable increases, the other variable value decreases. (but the relationship can be linear or non-linear)')

def process_subset(feature_set: np.ndarray):

    warnings.warn("This method has been greatly modified in 201018_asgmt3.py", DeprecationWarning)

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df0 = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    df = df0[df0.iloc[:, 6] > 0]

    aaa_diff = np.diff(df.iloc[:, 9])
    tbond10_diff = np.diff(df.iloc[:, 5])
    tbond30_diff = np.diff(df.iloc[:, 6])

    fedfund_diff = np.diff(df.iloc[:, 3])
    prime_diff = np.diff(df.iloc[:, 8])

    y = aaa_diff

    X = pd.concat([pd.DataFrame(tbond10_diff),
                   pd.DataFrame(tbond30_diff),
                   pd.DataFrame(fedfund_diff),
                   pd.DataFrame(prime_diff)], axis = 1)
    X.columns = ['tbond10_diff', 'tbond30_diff', 'fedfund_diff', 'prime_diff']

    model = sm.OLS(y, sm.add_constant(X[list(feature_set)]))
    regr = model.fit()
    return {'model': regr,
            'variables': list(feature_set),
            'AIC': regr.aic,
            'BIC': regr.bic,
            'R2': regr.rsquared,
            'AdjR2': regr.rsquared_adj}

def forward(explanatory, measure: str):

    assert measure == 'AIC' or measure == 'BIC' or measure == 'R2' or measure == 'AdjR2'
    columns = ['tbond10_diff', 'tbond30_diff', 'fedfund_diff', 'prime_diff']

    remaining_exp = [p for p in columns if p not in explanatory]
    results = []

    for p in remaining_exp:
        results.append(process_subset(explanatory + [p]))
        models = pd.DataFrame(results)
        if measure == 'AIC' or measure == 'BIC':
            model_each = models.loc[models[measure].idxmin()]
        else:
            model_each = models.loc[models[measure].idxmax()]
    return model_each

def backward_helper():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df0 = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    df = df0[df0.iloc[:, 6] > 0]

    aaa_diff = np.diff(df.iloc[:, 9])
    tbond10_diff = np.diff(df.iloc[:, 5])
    tbond30_diff = np.diff(df.iloc[:, 6])

    fedfund_diff = np.diff(df.iloc[:, 3])
    prime_diff = np.diff(df.iloc[:, 8])

    y = aaa_diff

    X = pd.concat([pd.DataFrame(tbond10_diff),
                   pd.DataFrame(tbond30_diff),
                   pd.DataFrame(fedfund_diff),
                   pd.DataFrame(prime_diff)], axis = 1)
    X.columns = ['tbond10_diff', 'tbond30_diff', 'fedfund_diff', 'prime_diff']

    bmodel1 = pd.DataFrame(columns = ['AIC', 'variables'])
    # WARNING: This 'AIC' parameter cannot be changed directly! It is hard-coded below!
    exp = X.columns

    bmodel1.loc[len(exp)] = [sm.OLS(y, sm.add_constant(X)).fit().aic, list(X.columns)]

    while len(exp) > 1:
        bmodel1.loc[len(exp) - 1] = backward(exp)
        exp = bmodel1.loc[len(exp) - 1]['variables']
    print(bmodel1)

def backward(explanatory):

    results = []
    for combo in itertools.combinations(explanatory, len(explanatory) - 1):
        results.append(process_subset(combo))
        models = pd.DataFrame(results)
        model_each = models.loc[models['AIC'].idxmin()]
    return model_each

def get_best_overall(measure: str):
    warnings.warn("This method has been greatly modified in 201018_asgmt3.py", DeprecationWarning)
    assert measure == 'AIC' or measure == 'BIC' or measure == 'R2' or measure == 'AdjR2'

    # Note this method and process_subset are highly coupled.
    # The sequence of the columns must be the same as those data in process_subset.
    columns = ['tbond10_diff', 'tbond30_diff', 'fedfund_diff', 'prime_diff']

    best_models = [None] * (len(columns) + 1)
    for i in range(1, len(columns) + 1):
        results = []
        for combo in itertools.combinations(columns, i):
            results.append(process_subset(combo))


        if measure == 'BIC' or measure == 'AIC':
            # AIC and BIC can be considered as measures of error.
            # Therefore, the smaller the better.
            temp_min = 23333
            for j in range(len(results)):
                if results[j][measure] < temp_min:
                    best_models[i] = results[j]
                    temp_min = results[j][measure]
        else:
            temp_max = -23333
            for j in range(len(results)):
                if results[j][measure] > temp_max:
                    best_models[i] = results[j]
                    temp_max = results[j][measure]
    best_models.pop(0)

    best_models = pd.DataFrame(best_models)

    if measure == 'BIC' or measure == 'AIC':
        best_models = best_models.sort_values(by = measure, ascending = True)
    else:
        best_models = best_models.sort_values(by = measure, ascending = False)
    print(best_models[[measure, 'variables']])
    return best_models


def pearsons_correlation(dataset1: List[float], ds1_name: str, dataset2: List[float], ds2_name: str):

    assert np.isnan(dataset1).any() == False
    assert np.isnan(dataset2).any() == False

    retval = np.corrcoef(dataset1, dataset2)
    r2, _ = sp.stats.stats.pearsonr(dataset1, dataset2)
    assert retval[0][0] == retval[1][1]
    assert retval[1][0] - retval[0][1] < 10 ** -10
    r1 = retval[1][0]

    assert r1 - r2 < 10 ** -10

    print('Pearson\'s correlation == {}, '.format(retval[1][0]), end = '')
    if r1 > 0.7:
        print('indicating a strong POSITIVE linear correlation between [{}] and [{}]'.format(ds1_name, ds2_name))
    elif r1 < -0.7:
        print('indicating a strong NEGATIVE linear correlation between [{}] and [{}] '.format(ds1_name, ds2_name))
    else:
        print('indicating NO significant correlation between [{}] and [{}]'.format(ds1_name, ds2_name))
    print('r^2 == {}%, meaning that aroung {}% of the variation of [{}] can be explained by the variation of [{}]'.format(round(r1 ** 2 * 100, 1), round(r1 ** 2 * 100, 1), ds2_name, ds1_name))
    print('Note: Pearson\'s correlation test performs well ONLY IF the joint probability distribution is a bivariate normal')


def read_data():

    parent_dir = pathlib.Path(__file__).parent.absolute()

    df_ge = pd.read_csv(os.path.join(parent_dir, 'geDJ_with-headers.txt'), sep='\s+')
    rt = np.diff(np.log(df_ge['close']))
    rt = np.append(np.nan, rt)
    df_ge['log_rt'] = rt
    df_ge['log_rt_standardized'] = (df_ge.log_rt - df_ge.log_rt.mean())/df_ge.log_rt.std()

    df_sp = pd.read_csv(os.path.join(parent_dir, 'sp500_with-headers.txt'), sep='\s+')
    rt = np.diff(np.log(df_sp['close']))
    rt = np.append(np.nan, rt)
    df_sp['log_rt'] = rt
    df_sp['log_rt_standardized'] = (df_sp.log_rt - df_sp.log_rt.mean())/df_sp.log_rt.std()

    return df_sp, df_ge


def lec4_application_one_interest_rates():

    parent_dir = pathlib.Path(__file__).parent.absolute()

    df_wi = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    aaa_bond_yield_diff = np.diff(df_wi.iloc[:,9])
    ten_year_tbond_rate_diff = np.diff(df_wi.iloc[:,5])

    scatter_plot_and_subplot(dataset1 = ten_year_tbond_rate_diff,
                             ds1_name = '10-Year Treasury Bond Rate',
                             dataset2 = aaa_bond_yield_diff,
                             ds2_name = 'AAA Bond Yield')


    ols_model_helper(patsy_formula = 'y ~ x',
                    data = {'y': aaa_bond_yield_diff, 'x' : ten_year_tbond_rate_diff},
                    with_intercept = True,
                    verbosity = 0,
                    significance_level = 0.1)

    ols_model_helper(patsy_formula = 'y ~ x - 1',
                     data = {'y': aaa_bond_yield_diff, 'x' : ten_year_tbond_rate_diff},
                    with_intercept = False,
                    verbosity = 0,
                    significance_level = 0.1)



def lec4_application_two_capm():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'charAB.txt'), sep='\t')
    df.month = pd.to_datetime(df.month, format = '%m/%d/%Y')


    ols_model_helper(patsy_formula = 'y ~ x',
                     data = {'y': df['excess_return_fund_A'], 'x' : df['market_excess_return']},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    # Here we are using two datasets to estimate the beta of the CAPM model
    ols_model_helper(patsy_formula = 'y ~ x',
                     data = {'y': df['excess_return_fund_B'], 'x' : df['market_excess_return']},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    model =ols_model_helper(patsy_formula = 'y ~ x - 1',
                     data = {'y': df['excess_return_fund_B'], 'x' : df['market_excess_return']},
                     with_intercept = False,
                     verbosity = 0,
                     significance_level = 0.05)
    predict_y_with_model(model = model, intercept_included = False, independent_variables = [[7, 8]], iv_names = ['x'])
    print('Note: According to page 102 of lec4_ch2-linear-regression-models.pdf, the final answer should be the predicted value + risk free rate')


def lec4_application_three_characteristic_line():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'capm.txt'), sep='\t', header = None)

    sp_diff = np.diff(np.log(df.iloc[:, 2]))
    msft_diff = np.diff(np.log(df.iloc[:,1]))

    tbillp = df.iloc[1:, 0] / (100 * 253)

    excess_return_sp500 = sp_diff - tbillp
    excess_return_msft = msft_diff - tbillp

    scatter_plot_and_subplot(dataset1 = excess_return_sp500,
                             ds1_name = 'Excess Return of SP500',
                             dataset2 = excess_return_msft,
                             ds2_name = 'Excess Return of Microsoft')

    ols_model_helper(patsy_formula = 'y ~ x',
                     data = {'y': excess_return_msft, 'x' : excess_return_sp500},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    ols_model_helper(patsy_formula = 'y ~ x - 1',
                     data = {'y': excess_return_msft, 'x' : excess_return_sp500},
                     with_intercept = False,
                     verbosity = 0,
                     significance_level = 0.05)

#    test_h0_for_beta01_at_specific_value(model = model, specific_values_for_betas =[1])


def lec5_application_one_interest_rates():

    parent_dir = pathlib.Path(__file__).parent.absolute()

    df_wi = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    df_wi = df_wi[df_wi.iloc[:, 6] > 0]

    aaa_bond_yield_diff = np.diff(df_wi.iloc[:,9])
    tbond_10year_rate_diff = np.diff(df_wi.iloc[:,5])
    tbond_30year_rate_diff = np.diff(df_wi.iloc[:,6])

    model = ols_model_helper(patsy_formula = 'y ~ x1 + x2',
                     data = {'y': aaa_bond_yield_diff,
                             'x1': tbond_10year_rate_diff,
                             'x2': tbond_30year_rate_diff},
                     with_intercept = True,
                     verbosity = 2,
                     significance_level = 0.05)
    test_betas_at_specific_values(model, [[0, 1, 0]], [0.5], 0.03)
    test_betas_at_specific_values(model, [[0, 1, 0], [0, 0, 1]], [0.5, 0.5], 0.03)
    test_betas_at_specific_values(model, [[0, 1, -1]], 0, 0.03)

    predict_y_with_model(model, True, [[-0.02, 0.12], [0.13, 0.07]], ['x1', 'x2'])

    model = ols_model_helper(patsy_formula = 'y ~ x1 + x2 - 1',
                 data = {'y': aaa_bond_yield_diff,
                         'x1': tbond_10year_rate_diff,
                         'x2': tbond_30year_rate_diff},
                 with_intercept = False,
                 verbosity = 2,
                 significance_level = 0.05)
    test_betas_at_specific_values(model = model,
                                  matrix_a = [1, -1],
                                  vector_b = 0,
                                  significance_level = 0.03)

    scatter_plot_and_subplot(dataset1 = tbond_10year_rate_diff + tbond_30year_rate_diff,
                         ds1_name = 'Total TBond Diff',
                         dataset2 = aaa_bond_yield_diff,
                         ds2_name = 'AAA Bond Yield Diff')

    ols_model_helper(patsy_formula = 'y ~ x',
                 data = {'y': aaa_bond_yield_diff,
                         'x': tbond_10year_rate_diff + tbond_30year_rate_diff},
                 with_intercept = True,
                 verbosity = 0,
                 significance_level = 0.05)

    ols_model_helper(patsy_formula = 'y ~ x - 1',
                 data = {'y': aaa_bond_yield_diff,
                         'x': tbond_10year_rate_diff + tbond_30year_rate_diff},
                 with_intercept = False,
                 verbosity = 0,
                 significance_level = 0.05)

def lec5_application_two_fama_french_3f_model():

    parent_dir = pathlib.Path(__file__).parent.absolute()

    df_ibm = pd.read_csv(os.path.join(parent_dir, 'IBM_GE_Jan69_Dec98.txt'), sep=',')
    df_ff0 = pd.read_csv(os.path.join(parent_dir, 'FF_Data_Jan69_Dec98.txt'), sep='\s+')

    df_ff0.columns = ['data', 'rm', 'smb', 'hml', 'rf']

    df_ff = df_ff0.iloc[510: 870]


    ribm = pd.DataFrame(df_ibm.iloc[:, 2] * 100 - np.array(df_ff['rf']))
    ribm.columns = ['return_ibm']


    df_ibm.reset_index(drop = True, inplace = True)
    df_ff.reset_index(drop = True, inplace = True)
    ribm.reset_index(drop = True, inplace = True)

    ibm_ff = pd.concat([df_ibm, df_ff, ribm], axis = 1)


    # Fama-French 3-Factor Model
    ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3',
                 data = {'y': ibm_ff['return_ibm'],
                         'x1': ibm_ff['rm'],
                         'x2': ibm_ff['smb'],
                         'x3': ibm_ff['hml']},
                 with_intercept = True,
                 verbosity = 0,
                 significance_level = 0.05)
    # Fama-French 3-Factor Model
    ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 - 1',
                 data = {'y': ibm_ff['return_ibm'],
                         'x1': ibm_ff['rm'],
                         'x2': ibm_ff['smb'],
                         'x3': ibm_ff['hml']},
                 with_intercept = False,
                 verbosity = 0,
                 significance_level = 0.05)

    # Capital Asset Pricing Model
    ols_model_helper(patsy_formula = 'y ~ x - 1',
                 data = {'y': ibm_ff['return_ibm'],
                         'x': ibm_ff['rm']},
                 with_intercept = False,
                 verbosity = 0,
                 significance_level = 0.05)

def lec5_application_three_exchange_rate():

    parent_dir = pathlib.Path(__file__).parent.absolute()

    df_exrate = pd.read_csv(os.path.join(parent_dir, 'HKExchange.txt'), sep='\t')
    time_index = np.arange(len(df_exrate.index)) + 1

    scatter_plot_and_subplot(dataset1 = time_index,
                             ds1_name = 'Time Series',
                             dataset2 = df_exrate['EXHKUS'],
                             ds2_name = 'Exchange Rate')
    # I() (Identity function) simply returns its input.
    model1 = ols_model_helper(patsy_formula = 'y ~ x + I(x**2)',
                 data = {'y': df_exrate['EXHKUS'],
                         'x': time_index},
                 with_intercept = True,
                 verbosity = 0,
                 significance_level = 0.05)

    plt.scatter(time_index, df_exrate['EXHKUS'], s = 1) # s: scalar
    plt.plot(time_index, model1.predict(), 'r--', label = 'Quadratic Regression', alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

    # I() (Identity function) simply returns its input.
    model2 = ols_model_helper(patsy_formula = 'y ~ x + I(x**2) + I(x**3)',
                 data = {'y': df_exrate['EXHKUS'],
                         'x': time_index},
                 with_intercept = True,
                 verbosity = 0,
                 significance_level = 0.05)

    plt.figure(figsize=(4*3,3*3))
    plt.plot(time_index, df_exrate['EXHKUS'], linewidth = 0.5, label = 'USD-HKD Exchange Rate')
    plt.plot(time_index, model1.predict(), 'r--',
             label = 'Quadratic Regression (AdjR^2=={})'.format(round(model1.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model2.predict(), 'g--',
             label = 'Cubic Regression(AdjR^2=={})'.format(round(model2.rsquared_adj, 3)), alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

    print(model1.predict({'x': 503}))
    print(model2.predict({'x': 503}))

    x1 = time_index
    x2 = time_index ** 2
    x3 = time_index ** 3
    x4 = (time_index > 100) * (time_index - 100) ** 3
    x5 = (time_index > 210) * (time_index - 210) ** 3
    x6 = (time_index > 390) * (time_index - 390) ** 3

    model = ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4+ x5+ x6',
                 data = {'y': df_exrate['EXHKUS'],
                         'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6},
                 with_intercept = True,
                 verbosity = 0,
                 significance_level = 0.05)

    plt.plot(time_index, df_exrate['EXHKUS'], linewidth = 0.5, label = 'USD-HKD Exchange Rate')
    plt.plot(time_index, model.predict(), 'r--', label = '', alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

    # My personal extra-curricular practice
    parent_dir = pathlib.Path(__file__).parent.absolute()

    df_myexrate = pd.read_csv(os.path.join(parent_dir, 'usdhkd_exrate.csv'))
    my_time_index = np.arange(len(df_myexrate['Price'].index)) + 1

    my_model = smf.ols(formula = 'y ~ np.sin(I(x + 1))',
                       data = {'y': df_myexrate['Price'], 'x': my_time_index}).fit()
    plt.plot(my_time_index, df_myexrate['Price'], linewidth = 0.5, label = 'USD-HKD Exchange Rate')
    plt.plot(my_time_index, my_model.predict(), 'r--', label = '', alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()


def lec5_assumptions_12346_checker():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hprice.txt'), sep='\t')

    fmodel = ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4',
                     data = pd.DataFrame({'x1': df['X1'], 'x2': df['X2'], 'x3': df['X3'], 'x4': df['X4'], 'y': df['Y']}),
                     with_intercept = True,
                     verbosity = 2,
                     significance_level = 0.05)

    assumptions_12346_checker(fmodel)
    print('IMPORTANT NOTE: The plot from the original data may look like pattern (d), implying that CONSTANT VARIANCE assumption does not satisfy')

def lec5_box_cox_transformation_remedy_non_normality_non_constant_variance():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hprice.txt'), sep='\t')

    y = df['Y']
    X = pd.concat([pd.DataFrame(df['X1']), pd.DataFrame(df['X2']), pd.DataFrame(df['X3']), pd.DataFrame(df['X4'])], axis = 1)

  #  X = sm.add_constant(X)
  #  print(X)
    BC = pd.DataFrame(columns = ['lambda', 'loglf'])
    lambda1 = np.arange(-2, 2, 0.01)
    # this range should be related to the [-2, +2] mentioned in the interpretation of assumptions_12346_checker()
    lambda1[200] = 0
    for i in range(1, len(lambda1)):
        BC.loc[i] = box_cox_transformer(lambda1[i], X, y)

    plt.plot(BC['lambda'], BC['loglf'])
    plt.show()
    print(BC['lambda'][BC['loglf'] == BC['loglf'].max()])

    mylambda = 0.11
    # this 0.11 comes from the above statement
    BCy = (y ** mylambda - 1) / mylambda

    dataBC = pd.DataFrame({'x1': df['X1'], 'x2': df['X2'], 'x3' : df['X3'], 'x4': df['X4'], 'BCy': BCy})

    model = ols_model_helper(patsy_formula = 'BCy ~ x1 + x2 + x3 + x4',
                     data = dataBC,
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)
    assumptions_12346_checker(model)
    warnings.warn("This method has been greatly modified in 201018_asgmt3.py", DeprecationWarning)

def lec6_linear_relationship():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    df1 = df[df.iloc[:, 6] > 0]

    aaa_diff = np.diff(df1.iloc[:, 9])
    cm10_diff = np.diff(df1.iloc[:, 5])
    cm30_diff = np.diff(df1.iloc[:, 6])

    ff_diff = np.diff(df1.iloc[:, 3])
    prime_diff = np.diff(df1.iloc[:, 8])

    model = ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4',
                     data = {'x1': cm10_diff, 'x2': cm30_diff, 'x3': ff_diff, 'x4': prime_diff, 'y': aaa_diff},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    assumption_6_checker(model)

def lec6_independence():
    print('No autocorrelation + normality -> independence')

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hprice.txt'), sep='\t')

    model = ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4',
         data = pd.DataFrame({'x1': df['X1'], 'x2': df['X2'], 'x3': df['X3'], 'x4': df['X4'], 'y': df['Y']}),
         with_intercept = True,
         verbosity = 0,
         significance_level = 0.05)

    dw = durbin_watson_autocorrelation_test(model)

    # now we try to correct the autocorrelation problem.

    p = 1 - dw / 2

    new_y = df['Y'] - p * df['Y'].shift(1)
    new_x1 = df['X1'] - p * df['X1'].shift(1)
    new_x2 = df['X2'] - p * df['X2'].shift(1)
    new_x3 = df['X3'] - p * df['X3'].shift(1)
    new_x4 = df['X4'] - p * df['X4'].shift(1)

    model_after = ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4',
         data = pd.DataFrame({'x1': new_x1[1:], 'x2': new_x2[1:], 'x3': new_x3[1:], 'x4': new_x4[1:], 'y': new_y[1:]}),
         with_intercept = True,
         verbosity = 0,
         significance_level = 0.05)

    durbin_watson_autocorrelation_test(model_after)

def lec6_multicollinearity_detection_and_correction():


    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    # reading non-missing data only
    df1 = df[df.iloc[:, 6] > 0]

    aaa_diff = np.diff(df1.iloc[:, 9])
    cm10_diff = np.diff(df1.iloc[:, 5])
    cm30_diff = np.diff(df1.iloc[:, 6])

    ff_diff = np.diff(df1.iloc[:, 3])
    prime_diff = np.diff(df1.iloc[:, 8])

    xx = pd.concat([pd.DataFrame(cm10_diff), pd.DataFrame(cm30_diff), pd.DataFrame(ff_diff), pd.DataFrame(prime_diff), pd.DataFrame(aaa_diff)], axis = 1)

    xx = sm.add_constant(xx)
    xx.columns = ['constant', 'cm10_diff', 'cm30_diff', 'ff_diff', 'prime_diff', 'aaa_diff']

    multicollinearity_detector(dataset = xx, verbosity = 1)
    print('')
    xx1 = xx.drop('aaa_diff', 1)
    multicollinearity_detector(dataset = xx1, verbosity = 1)

    print('')
    xx2 = xx.drop('cm10_diff', 1)
    multicollinearity_detector(dataset = xx2, verbosity = 1)

    # Dr. Yu introduced another method to remedy the multicollinearity problem withOUT droppin an independent variable. But I think this method sounds odd: why don't we just drop it??? As a result, it is not implemented here.

def main():

    print('\n\n===== Lec4 Application I: Interest Rates =====')
    lec4_application_one_interest_rates()

    print('\n\n===== Lec4 Application II: CAPM =====')
    lec4_application_two_capm()

    print('\n\n===== Lec4 Application III: Characteristic Line =====')
    lec4_application_three_characteristic_line()

    print('\n\n===== Lec5 Application I: Interest Rates =====')
    lec5_application_one_interest_rates()

    print('\n\n===== Lec5 Application II: Fama-French Three-Factor Model =====')
    lec5_application_two_fama_french_3f_model()

    print('\n\n===== Lec5 Application III: HKD-USD Exchange Rate =====')
    lec5_application_three_exchange_rate()

    print('\n\n===== Lec5: Best subset selection method  =====')
    measure = 'AdjR2'
    get_best_overall(measure)

    print('\n\n===== Lec5: Sequential selection methods (Forward) =====')
    fmodel = pd.DataFrame(columns = [measure, 'variables'])
    exp = []

    for i in range(1, 5):
        fmodel.loc[i] = forward(exp, measure = measure)
        exp = fmodel.loc[i]['variables']
    print(fmodel)

    print('\n\n===== Lec5: Sequential selection methods (Backward) =====')
    backward_helper()

    print('\n\n===== Lec5: Assumptions 1-4 and 6 =====')
    lec5_assumptions_12346_checker()

    print('\n\n===== Lec5: Box-Cox Transformation--remedy non-normality or non -constant variance =====')
    lec5_box_cox_transformation_remedy_non_normality_non_constant_variance()

    print('\n\n===== Lec6: Assumption 5 -- Linear relationship =====')
    lec6_linear_relationship()

    print('\n\n===== Lec6: Assumption 4 -- Independent =====')
    lec6_independence()

    print('\n\n===== Lec6: Assumption 4 -- Independent =====')
    lec6_multicollinearity_detection_and_correction()

if __name__ == '__main__':

    main()