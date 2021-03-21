#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:56:13 2020

@author: mamsds
"""
from typing import List
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf

import importlib
lec45 = importlib.import_module('200922_ch2_linear-regression')
#
# Refer to page 103 of lec4-5_ch2-linear-regression-models.pdf
#

def q3():


    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hw2q3.txt'), sep='\t')
    df['market_excess_return']
    df['excess_return_fund_A']




    lec45.scatter_plot_and_subplot(dataset1 = df['market_excess_return'],
                             ds1_name = 'market_excess_return',
                             dataset2 = df['excess_return_fund_A'],
                             ds2_name = 'excess_return_fund_A')
    lec45.scatter_plot_and_subplot(dataset1 = df['market_excess_return'],
                             ds1_name = 'market_excess_return',
                             dataset2 = df['excess_return_fund_B'],
                             ds2_name = 'excess_return_fund_B')
    print('Fund A')
    lec45.ols_model_helper(patsy_formula = 'y ~ x',
                     data = {'y': df['excess_return_fund_A'], 'x' : df['market_excess_return']},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    print('Fund B')
    lec45.ols_model_helper(patsy_formula = 'y ~ x',
                     data = {'y': df['excess_return_fund_B'], 'x' : df['market_excess_return']},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)
    model = lec45.ols_model_helper(patsy_formula = 'y ~ x - 1',
                     data = {'y': df['excess_return_fund_B'], 'x' : df['market_excess_return']},
                     with_intercept = False,
                     verbosity = 2,
                     significance_level = 0.05)
    lec45.test_betas_at_specific_values(model = model,
                                        matrix_a = [1],
                                        vector_b = [1],
                                        significance_level = 0.05)

    plt.figure(figsize=(4*3,3*3))
    plt.scatter(df['market_excess_return'], df['excess_return_fund_B'], marker = '+', alpha = 0.9)
    plt.plot(df['market_excess_return'], model.predict(), 'r--', label = '', alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

    lec45.predict_y_with_model(model = model,
                         intercept_included = False,
                         independent_variables = [[7, ]],
                         iv_names = ['x'],
                         significance_level = 0.1)


def q4():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hw2q4.txt'), sep='\s+')
    time_index = np.arange(len(df.index)) + 1

    lec45.scatter_plot_and_subplot(dataset1 = time_index,
                         ds1_name = 'Time Series',
                         dataset2 = df['EXEUROUS'],
                         ds2_name = 'USD-EUR Exrage')


    model1 = lec45.ols_model_helper(patsy_formula = 'y ~ x + I(x**2)',
                     data = {'y': df['EXEUROUS'],
                             'x': time_index},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    plt.scatter(time_index, df['EXEUROUS'], s = 1) # s: scalar
    plt.plot(time_index, model1.predict(), 'r--', label = 'Quadratic Regression', alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

        # I() (Identity function) simply returns its input.
    model2 = lec45.ols_model_helper(patsy_formula = 'y ~ x + I(x**2) + I(x**3)',
                 data = {'y': df['EXEUROUS'],
                         'x': time_index},
                 with_intercept = True,
                 verbosity = 2,
                 significance_level = 0.05)

    plt.figure(figsize=(4*3,3*3))
    plt.plot(time_index, df['EXEUROUS'], linewidth = 0.5, label = 'USD-EUR Exchange Rate')
    plt.plot(time_index, model1.predict(), 'r--',
             label = 'Quadratic Regression (AdjR^2=={})'.format(round(model1.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model2.predict(), 'g--',
             label = 'Cubic Regression(AdjR^2=={})'.format(round(model2.rsquared_adj, 3)), alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

    # Refer to page 147 of lec4-5_ch2-linear-regression-models.pdf
    x1 = time_index
    x2 = time_index ** 2
    x3 = (time_index > 167) * (time_index - 167) ** 2
    x4 = (time_index > 334) * (time_index - 334) ** 2


    model3 = lec45.ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4',
                 data = {'y': df['EXEUROUS'],
                         'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4},
                 with_intercept = True,
                 verbosity = 2,
                 significance_level = 0.05)

    plt.figure(figsize=(4*3,3*3))
    plt.plot(time_index, df['EXEUROUS'], linewidth = 0.5, label = 'USD-EUR Exchange Rate')
    plt.plot(time_index, model1.predict(), 'r--',
             label = 'Quadratic Regression (AdjR^2=={})'.format(round(model1.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model2.predict(), 'g--',
             label = 'Cubic Regression(AdjR^2=={})'.format(round(model2.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model3.predict(), 'g--',
             label = 'Quadratic Regression with knots(AdjR^2=={})'.format(round(model2.rsquared_adj, 3)), alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

    # Refer to page 147 of lec4-5_ch2-linear-regression-models.pdf
    x1 = time_index
    x2 = time_index ** 2
    x3 = time_index ** 3
    x4 = (time_index > 125) * (time_index - 125) ** 3
    x5 = (time_index > 251) * (time_index - 251) ** 3
    x6 = (time_index > 377) * (time_index - 377) ** 3

    model4 = lec45.ols_model_helper(patsy_formula = 'y ~ x1 + x2 + x3 + x4 + x5 + x6',
                 data = {'y': df['EXEUROUS'],
                         'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6},
                 with_intercept = True,
                 verbosity = 2,
                 significance_level = 0.05)

    plt.figure(figsize=(4*3,3*3))
    plt.plot(time_index, df['EXEUROUS'], linewidth = 0.5, label = 'USD-EUR Exchange Rate')
    plt.plot(time_index, model1.predict(), 'r--',
             label = 'Quadratic Regression (AdjR^2=={})'.format(round(model1.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model2.predict(), 'g--',
             label = 'Cubic Regression(AdjR^2=={})'.format(round(model2.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model3.predict(), 'b--',
             label = 'Quadratic Regression with knots(AdjR^2=={})'.format(round(model3.rsquared_adj, 3)), alpha = 0.9)
    plt.plot(time_index, model4.predict(), 'g--',
             label = 'Cubic Regression with knots(AdjR^2=={})'.format(round(model4.rsquared_adj, 3)), alpha = 0.9)
    plt.legend(loc = 'best')
    plt.show()

def main():

    print('\n\n ========== Question 3(a), 3(b) ==========')
    q3()

    print('\n\n ========== Question 3(a), 3(b) ==========')
    q4()

if __name__ == '__main__':

    main()