#!/usr/bin/env python3

#  monte carlo stimulation?
# P 141 of Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
# refer P139 in Ch1_MFIT5003_Fall2020-21_with_MJD.pdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy

import os
import pathlib

def monte_carlo_simulation_merton_jump_diffusion():

    T = 2
    #mu = 0.2  # without drift
    sigma = 0.3
    SO = 20
    dt = 0.01 # step side?
    N = round(T/dt)
    
    t = np.linspace(0, T, N) # return evenly spaced numbers over a specified inteval
    
    risk_free_rate = 0.05 # 5% inrerest rate
    # Poisson intensity
    lamb = 0.95 
    
    # jump size
    mu_jump = -0.6
    sigma_jump = 0.25
    
    k = np.exp(mu_jump + 0.5*sigma_jump**2) -1
    mu = risk_free_rate - lamb*k
    
    B = {}
    for sim in range(1, 2000):
        S = np.zeros(N+1)
        S[0] = SO
        for t in range (1, N+1):
            GBM_part = S[t-1]*(np.exp((mu - 0.5*sigma**2)*dt + sigma*np.random.standard_normal(size=1) * np.sqrt(dt)))
            jump_part = S[t-1]*(np.exp(mu_jump+sigma_jump*np.random.standard_normal(size=1))-1)*np.random.poisson(lamb*dt, size=1)
            S[t] = GBM_part + jump_part
        B['stimulation_'+str(sim)] = S
    
    B = pd.DataFrame(B)
    B.head()
    #B = pd.DataFrame(S)
    B.plot(figsize=(10, 7), grid = True, legend = False)
    plt.title('Stimulation from Merton Jump Diffusion Model')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.draw()
    #plt.show()
    
    plt.figure(2)    
    plt.hist(B.iloc[-1], bins=50,edgecolor='black', linewidth=0.5)
    plt.axvline(np.percentile(B.iloc[-1], 5), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(B.iloc[-1], 95), color='r', linestyle='dashed', linewidth=1)
    plt.draw()
    #plt.show()
    
    plt.figure(3)    
    plt.hist(np.log(B.iloc[-1]), bins=50,edgecolor='black', linewidth=0.5)
    plt.show()

def merton_jump_diffusion():
    # Let's JUMP!
    # Merton Jump Diffusion Model (MDJ) Model UNDER risk-neutral measure
    # P160 of Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    
    T = 5
    #mu = 0.2  # without drift
    # mu is replace by r below
    sigma = 0.3 # The same as GBM Model
    S0 = 20
    dt = 0.01 # step side?
    N = round(T/dt)
    
    t = np.linspace(0, T, N) # return evenly spaced numbers over a specified inteval
    
    risk_free_rate = 0.07 # risk-free interest rate
    # According to page 156, dNt denotes the number of jumps. dNt cannot be predicted
    # accurately, we assume it follows Poisson distribution. Nt ~ Possion(λt)
    # λ is also called "poisson intensity"
    lamb = 0.95 
    
    # jump size. According to page 150 of Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    # Suppose price jumps from St to Jt*St, Js is called an absolute price jump size Merton assumes that Jt 
    # is a positive random variable from lognormal distribution logJt ~ N(mu, sigma^2).
    # This assumption, however, is more or less a subjective one. Users are free to use any other distributions instead.
    mu_jump = 0.6 
    sigma_jump = 0.25
    
    # In MJD Model we have two more random variables compared with Geometric Brownian Motion (GBM) model, namely
    # jump_size (assumed to follow normal distribution) and number_of_jump (assumed to follow Possion distribution)
    
    k = np.exp(mu_jump + 0.5*sigma_jump**2) - 1
    # k is defined as the mean of (Jt - 1), (not Jt -1 itself)
    
    mu = risk_free_rate - lamb * k
    print('mu == {} (it could be negative!)'.format(mu))
    # In this case, interest rate is not equal to mu.
    
    S = np.zeros(N+1)
    S[0] = S0
    
    for t in range (1, N+1):
        # Geometric Brownian Motion
        GBM_part = S[t-1] * (np.exp((mu - 0.5*sigma**2) * dt + sigma * np.random.standard_normal(size=1) * np.sqrt(dt)))
        MDJ_part = S[t-1] * (np.exp(mu_jump + sigma_jump * np.random.standard_normal(size=1))-1) * np.random.poisson(lamb*dt, size=1)
        S[t] = GBM_part + MDJ_part
    
    S = pd.DataFrame(S)
    S.plot(figsize=(10, 7), grid = True, legend = False)
    plt.title('Stimulation from Merton Jump Diffusion Model')
    plt.xlabel('Time')
    plt.ylabel('Price')

def monte_carlo_simulation_with_gbm_histogram2():
    T = 2
    mu = 0.2  # without drift
    sigma = 0.3
    SO = 20
    dt = 0.01 # step side?
    N = round(T/dt)
    
    t = np.linspace(0, T, N) # return evenly spaced numbers over a specified inteval
    
    S = {}
    for sim in range(1, 50000):
    
        dZ = np.random.standard_normal(size = N) * np.sqrt(dt)
        Z = np.cumsum(dZ)
    
        X = (mu - 0.5 * sigma ** 2) * T + sigma * Z # upper T or lower t here? subject to further research!
        S["Simulation_"+ str(sim)] = np.append(SO, SO*np.exp(X))
        
    S = pd.DataFrame(S)
    plt.hist(S.iloc[-1], bins=100,edgecolor='black', linewidth=0.5)
    plt.axvline(np.percentile(S.iloc[-1], 5), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(S.iloc[-1], 95), color='r', linestyle='dashed', linewidth=1)
    plt.show()
    
    S.iloc[-1].describe()

def monte_carlo_simulation_with_gbm_histogram():

    T = 5
    mu = 0.05  # without drift
    sigma = 0.3
    SO = 20
    dt = 0.01 # step side?
    N = round(T/dt)
    print(N)
    t = np.linspace(0, T, N) # return evenly spaced numbers over a specified inteval
    
    S = {}
    for sim in range(1, 5000):
    
        dZ = np.random.standard_normal(size = N) * np.sqrt(dt)
        Z = np.cumsum(dZ)
    
        X = (mu - 0.5 * sigma ** 2) * t + sigma * Z
        S["Simulation_"+ str(sim)] = np.append(SO, SO*np.exp(X))
        
    S = pd.DataFrame(S)
    plt.hist(x=S.iloc[-1], bins=50, histtype='bar', orientation='vertical', edgecolor='black', linewidth=1.2)  
    # hist stands for histogram here. However, the real origin of the name "histogram" is not clear.
    plt.show()
    S.iloc[-1].describe()

def monte_carlo_simulation_with_gbm():
    #  monte carlo stimulation?
    # P 141 of Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    # refer P139 in Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    
    T = 5
    risk_free_rate = 0.05
    mu = risk_free_rate
    sigma = 0.3
    SO = 20
    dt = 0.01 # step side?
    N = round(T/dt)
    
    t = np.linspace(0, T, N) # return evenly spaced numbers over a specified inteval
    
    S = {}
    for sim in range(1, 100):
    
        dZ = np.random.standard_normal(size = N) * np.sqrt(dt)
        Z = np.cumsum(dZ)
    
        X = (mu - 0.5 * sigma ** 2) * t + sigma * Z
        S["Simulation_"+ str(sim)] = np.append(SO, SO*np.exp(X))
        
    S = pd.DataFrame(S)
    
    

   # plt.subplot(111)
    S.plot(figsize=(10, 7), grid = True, legend = False)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.draw()
    #plt.show()
    
    plt.figure(2)
   # plt.subplot(222)
    plt.hist(x=S.iloc[-1], bins=20, histtype='bar', orientation='horizontal', edgecolor='black', linewidth=1.2)  # hist stands for histogram here. However, the real origin of the name "histogram" is not clear.
    # list[-1] refers to the last element
    # bin can be a verb, to bin a historgram means to divide it into a series of intervals
    plt.draw()  # Draws, but does not block
   # raw_input()  # This shows the first figure "separately" (by waiting for "enter").
    plt.show()

def quantile_quantile_plot(da):

    rt = np.diff(np.log(da['close']))
    plt.figure(1)
 #   plt.subplot(111)
    plt.plot(rt, color='green', linewidth=0.1)
    # plot the points before standardization.
  #  plt.show()
    
    z = (rt - rt.mean())/rt.std() # This is used to standardize the values.
    plt.figure(2)
  #  plt.subplot(111)
    plt.plot(z, color='red', linewidth=0.1) # plot the points after standardization.
  #  plt.show()
    
    # In the example from Dr. Yu, the dist parameter is omitted and the default will be norm(al distribution)
    # A list of possible distribution parameters can be found here:
    # https://docs.scipy.org/doc/scipy/reference/stats.html
    sm.qqplot(data=rt, dist=scipy.stats.distributions.norm, line='45')
    sm.qqplot(data=z, dist=scipy.stats.distributions.norm, line='45')
    sm.qqplot(data=z, dist=scipy.stats.distributions.uniform, line='45')
    
    # Basically, a quantile-quantile plot is used to check whether or not a given
    # set of data is sampled from a certain type of distribution (e.g. normal distribution)
    
  #  import matplotlib.pyplot as plt
    plt.show()

def geometric_brownian_motion_without_drift():
    
    T = 2 # time period, the real interval is generated below as t
    risk_free_rate = 0
    mu = risk_free_rate  # 0 means without drift. 
    # This is very important: If the stock price is driven by the risk-neutral measure,
    # then it will have an expeccted value only discounted by the risk-free rate r.
    # For risk-neutral measure, mu is euqla to risk-free rate r.
    # However, for Merton Jump Diffusion Model (MDJ) Model, interest rate is not equal to mu
    sigma = 0.3 # means volatility
    S0 = 20 # initial stock price
    dt = 0.01 # step side
    N = round(T/dt)
    
    t = np.linspace(0, T, N) 
    # np.linspace is not about vector space, it just 
    # returns evenly spaced numbers over a specified interval.
    print(N)
    
    dZ = np.random.standard_normal(size = N) * np.sqrt(dt)
    Z = np.cumsum(dZ) # the summation operation seen in the formula.
    
    X = (mu - 0.5 * sigma ** 2) * t + sigma * Z
    print('len(Z):{}, len(X): {}'.format(len(Z), len(X)))
    #S = np.append(S0, S0 * np.exp(X)) # This line will include S0 in the final result
    S = S0 * np.exp(X) # This line will NOT include S0 in the final result.
    # np.exp(x) just means e^x 
    plt.figure(7)
  #  plt.subplot(111)
    plt.plot(t, S)
 #   plt.show()
    
    rt = np.log(S)

    sm.qqplot(data=rt, dist=scipy.stats.distributions.norm, line='45')
    # A list of possible distribution parameters can be found here:
    # https://docs.scipy.org/doc/scipy/reference/stats.html
  #  plt.show()

def histogram():
    mu = 0
    sigma = 1
    nd = np.random.normal(loc=mu, scale=sigma, size=10000)
    
    plt.hist(x=nd, bins=50, histtype='bar', orientation='vertical', edgecolor='black', linewidth=1.2)  
    # hist stands for histogram here. However, the real origin of the name "histogram" is not clear.
    plt.show()

def read_dataset():
    
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

    return da_sp, da_dj

def main():
    
    da_sp, da_dj = read_dataset()
    
    switch = 6
    # For unknown reasons, plt does not work well across functions. Therefore a switch is implemented.
    if switch == 0:
        quantile_quantile_plot(da_sp)
    elif switch == 1:
        geometric_brownian_motion_without_drift()
    elif switch == 2:
        monte_carlo_simulation_with_gbm()
    elif switch == 3:
        monte_carlo_simulation_with_gbm_histogram()
    elif switch == 4:
        monte_carlo_simulation_with_gbm_histogram2()
    elif switch == 5:
        monte_carlo_merton_jump_diffusion()
    elif switch == 6:
        monte_carlo_simulation_merton_jump_diffusion()

    
if __name__ == '__main__':
    main()