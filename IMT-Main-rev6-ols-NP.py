import numpy as np
from scipy import optimize
from sympy.solvers import solve
from scipy.optimize import fsolve
from scipy.optimize import root
from sympy import Symbol
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import bisect
import six 
from six.moves import zip
import csv
from numpy import genfromtxt
import xlsxwriter as xlwt
from matplotlib.ticker import FuncFormatter
import random
import pandas as pd
# ========================================== parameters ========================================== #
global PMs, PMu, gammaN, mc, buM, bsM
global tsN, tuN, N, Aa, Am, Da, Dm, Dr, Ar
 
# ========================================== parameters ========================================== #
Tstart = 2000
Tend = 2100
Tstep = 20
T = int((Tend - Tstart)/Tstep + 1)                  # Time horizonreg
nreg = 160                                          # number of countries #161
nssp = 5                                            # number of SSP scenarios = 5
nrcp = 4                                            # number of RCP scenarios = 4
alpha = 0.55/2                                      # Agricultural share in Consumption function
eps = 0.75                                          # Elasticity of substitution in Consumption function
theta = 1.0                                         # labor power in production function
nsol = np.zeros((nreg*2, T, nrcp, nssp))

# ========================================== Damages =========================================== #
# D = g0 + g1 * T + g2 * T^2
 
# Agricultural parameters
g0a = -2.24
g1a = 0.308
g2a = -0.0073
 
# Manufacturing parameters
g0m = 0.3
g1m = 0.08
g2m = -0.0023
 
# ========================================== Variables =========================================== #                    
 
# == temperature == #
Temp = np.zeros((nreg, T, nrcp, nssp))                    # Temperature
 
# == child-rearing time == #
gamma0 = 0.45                                        # Share of children's welbeing in Utility function of parents in 1980
gamma = np.zeros((nreg, T, nrcp, nssp))
gammaN = [0] * nreg
 
# == Age matrix == #
nu = np.zeros((nreg, T, nrcp, nssp))                       # number of unskilled children
ns = np.zeros((nreg, T, nrcp, nssp))                       # number of skilled children
L = np.zeros((nreg, T, nrcp, nssp))                        # Number of unskilled parents
H = np.zeros((nreg, T, nrcp, nssp))                        # Number of skilled parents
h = np.zeros((nreg, T, nrcp, nssp))                        # Ratio of skilled to unskilled labor h=H/L
hn = np.zeros((nreg, T, nrcp, nssp))                       # Ratio of skilled to unskilled children h=ns/nu
N = np.zeros((nreg, T, nrcp, nssp))                        # Adult population
Ng = np.zeros((nreg, T, nrcp, nssp))                       # Gross adult population
Pop = np.zeros((nreg, T, nrcp, nssp))                      # total population
 
tu = np.zeros((nreg, T, nrcp, nssp))                       # time spent on raising an unskilled child
ts = np.zeros((nreg, T, nrcp, nssp))                       # time spent on raising an skilled child
tr = np.zeros((nreg, T, nrcp, nssp))                       # relative time spent on raising an skilled child to an unskilled child

tuN = [0] * nreg
tsN = [0] * nreg
tug = [0.05] * nreg                                 # 0.01
tsg = [0.05] * nreg                                 # 0.05
 
# == Prices == #
pa = np.zeros((nreg, T, nrcp, nssp))                       # Pice of AgricuLtural good
pm = np.zeros((nreg, T, nrcp, nssp))                       # Pice of Manufacturing good
pr = np.zeros((nreg, T, nrcp, nssp))                       # Relative pice of Manufacturing to Agricultural goods
 
# == Wages == #
wu = np.zeros((nreg, T, nrcp, nssp))                       # Wage of unskilled labor
ws = np.zeros((nreg, T, nrcp, nssp))                       # Wage of skilled labor
wr = np.zeros((nreg, T, nrcp, nssp))                       # Wage ratio of skilled to unskilled labor
 
# == Technology == #
Aa = np.zeros((nreg, T, nrcp, nssp))                       # Technological growth function for Agriculture
Am = np.zeros((nreg, T, nrcp, nssp))                       # Technological growth function for Manufacurng
Ar = np.zeros((nreg, T, nrcp, nssp))                       # ratio of Technology in Manufacurng to Agriculture
Aag = np.zeros((nreg, nrcp, nssp))                         # growth rate of Agricultural productivity
Amg = np.zeros((nreg, nrcp, nssp))                         # growth rate of Manufacturing productivity
Amgr = [0.01] * nreg                                       # annual growth rate of Manufacturing productivity
 
# == Output == #
Y = np.zeros((nreg, T, nrcp, nssp))                        # Total output
Ya = np.zeros((nreg, T, nrcp, nssp))                       # AgricuLtural output
Ym = np.zeros((nreg, T, nrcp, nssp))                       # Manufacturing output
Yr = np.zeros((nreg, T, nrcp, nssp))                       # Ratio of Manufacturing output to Agricultural output
Yp = np.zeros((nreg, T, nrcp, nssp))                       # Output per capita
 
# == Output == #
Da = np.zeros((nreg, T, nrcp, nssp))                       # AgricuLtural damage
Dm = np.zeros((nreg, T, nrcp, nssp))                       # Manufacturing damage
Dr = np.zeros((nreg, T, nrcp, nssp))                       # Ratio of Manufacturing damages to Agricultural damages
 
# == Consumption == #
cau = np.zeros((nreg, T, nrcp, nssp))                      # consumption of agricultural good unskilled
cas = np.zeros((nreg, T, nrcp, nssp))                      # consumption of agricultural good skilled
cmu = np.zeros((nreg, T, nrcp, nssp))                      # consumption of manufacturing good unskilled
cms = np.zeros((nreg, T, nrcp, nssp))                      # consumption of manufacturing good skilled
cu = np.zeros((nreg, T, nrcp, nssp))                       # consumption of all goods unskilled
cs = np.zeros((nreg, T, nrcp, nssp))                       # consumption of all goods skilled
 
# == Migration == #
PMs = np.zeros((nreg, nreg, T, nrcp, nssp))                # migration probability for skilled labor
PMu = np.zeros((nreg, nreg, T, nrcp, nssp))                # migration probability for unskilled labor
 
Mout_s = np.zeros((nreg, T, nrcp, nssp))                   # total outflow of skilled labor
Mout_u = np.zeros((nreg, T, nrcp, nssp))                   # total outflow of unskilled labor
Mout = np.zeros((nreg, T, nrcp, nssp))                     # total outflow of labor
 
Min_s = np.zeros((nreg, T, nrcp, nssp))                    # total inflow of skilled labor
Min_u = np.zeros((nreg, T, nrcp, nssp))                    # total inflow of unskilled labor
Min = np.zeros((nreg, T, nrcp, nssp))                      # total inflow of labor
 
Mtotal_s = np.zeros((T, nrcp, nssp))                       # total flow of skilled labor
Mtotal_u = np.zeros((T, nrcp, nssp))                       # total flow of unskilled labor
Mtotal = np.zeros((T, nrcp, nssp))                         # total flow of labor

Ms = np.zeros((nreg, nreg, T, nrcp, nssp))                 # migration flow level of skilled labor
Mu = np.zeros((nreg, nreg, T, nrcp, nssp))                 # migration flow level of unskilled labor
Mt = np.zeros((nreg, nreg, T, nrcp, nssp))                 # total migration flow
 
MCs = np.zeros((nreg, nreg, T, nrcp, nssp))                # Migration cost of skilled labor
MCu = np.zeros((nreg, nreg, T, nrcp, nssp))                # Migration cost of unskilled labor

#MCs[:, :, :, 2] = 0.5                               # SSP3 Migration cost of skilled labor
#MCu[9, 16, :, :] = 1.0                               # SSP3 Migration cost of unskilled labor
#
#MCs[:, :, 1:T, :] = 0.0                              #  Migration cost of skilled labor
#MCu[:, :, 1:T, :] = 0.0

Ltotal = np.zeros((T, nrcp, nssp))                         # Number of unskilled parents in the world
Htotal = np.zeros((T, nrcp, nssp))                         # Number of skilled parents in the world
htotal = np.zeros((T, nrcp, nssp))                         # Ratio of skilled to unskilled labor h=H/L in the world

# log(prob) =  C1 x log(pop_orig) +  C2 x log(pop_dest) +  C3 x log(wage_dest/wage_orig) + C4 x log(distance)
Coef_u = [0.474256, -0.37707, 0.413186, -0.06192]                   # Parameters of historical migration probabilities of unskilled labor
Coef_s = [0.173871, -0.38297, 0.168784, -0.06268]                   # Parameters of historical migration probabilities of skilled labor

# ============================================== Input Data for Calibration ============================================== #

SSP_name = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
RCP_name = ['RCP26', 'RCP45', 'RCP60', 'RCP85']
Year_name = [i for i in range(Tstart, Tend + 1, Tstep)]
Country_name = [0] * nreg

y2000_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/iso3_dataset_wide.csv'
historical_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/new_dataset_iso3_for_soheil.csv'
projection_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/proj_dataset_wide.csv'
migprob_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/migration_probability_calibration_1980_2000_ss.csv'
migflow_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/migration_flow_1980_2000_ss.csv'
migflow0_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/migration_flow_1960_1980_ss.csv'
feunskilled_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/coefficients_unskilled_ols_ss.csv'
feskilled_add = 'C:/Users/Shayegh/Dropbox/Research/Papers/2_Submitted/5-IMT_International Migration/witch-econometrics/coefficients_skilled_ols_ss.csv'

y2000_data = pd.read_csv(y2000_add)
hist_data = pd.read_csv(historical_add) #, header=None
proj_data = pd.read_csv(projection_add)
migprob_data = pd.read_csv(migprob_add)
migflow_data = pd.read_csv(migflow_add)
migflow0_data = pd.read_csv(migflow0_add)
feunskilled_data = pd.read_csv(feunskilled_add)
feskilled_data = pd.read_csv(feskilled_add)

Ydata = np.zeros((nreg, T, nrcp, nssp))
Pdata = np.zeros((nreg, T, nrcp, nssp))
Ndata = np.zeros((nreg, T, nrcp, nssp))
Xdata = np.zeros((nreg, T, nrcp, nssp))
Ldata = np.zeros((nreg, T, nrcp, nssp))
Hdata = np.zeros((nreg, T, nrcp, nssp))
hdata = np.zeros((nreg, T, nrcp, nssp))
Tempdata = np.zeros((nreg, T, nrcp, nssp))

Iudata = [0] * nreg
Isdata = [0] * nreg 

N0data = [0] * nreg
H0data = [0] * nreg
L0data = [0] * nreg
Iu0data = [0] * nreg
Is0data = [0] * nreg
Fuodata = [0] * nreg 
Fuddata = [0] * nreg
Fsodata = [0] * nreg
Fsddata = [0] * nreg

Dist =np.zeros((nreg, nreg))
prob0 = np.zeros((nreg, nreg))
M0 = np.zeros((nreg, nreg))
M00 = np.zeros((nreg, nreg))

Country_name = migflow0_data['country']

for k in range(nreg):
    Iu = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[0])]['I_u']
    Is = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[0])]['I_s']    
    
    Nd = hist_data.loc[(hist_data['origin'] == Country_name[k])&(hist_data['year'] == 1980)]['adults_origin']    
    Hd = hist_data.loc[(hist_data['origin'] == Country_name[k])&(hist_data['year'] == 1980)]['adults_s_origin']    
    Ld = hist_data.loc[(hist_data['origin'] == Country_name[k])&(hist_data['year'] == 1980)]['adults_u_origin']    
    Isd = hist_data.loc[(hist_data['origin'] == Country_name[k])&(hist_data['year'] == 1980)]['I_s_origin']    
    Iud = hist_data.loc[(hist_data['origin'] == Country_name[k])&(hist_data['year'] == 1980)]['I_u_origin']
    Fuod = feunskilled_data.loc[(feunskilled_data['country'] == Country_name[k])]['origin']
    Fudd = feunskilled_data.loc[(feunskilled_data['country'] == Country_name[k])]['destination']    
    Fsod = feskilled_data.loc[(feskilled_data['country'] == Country_name[k])]['origin']    
    Fsdd = feskilled_data.loc[(feskilled_data['country'] == Country_name[k])]['destination']    


    if np.any(Nd) != 0 and Nd.values[0] > 0:
        N0data[k] = Nd.values[0]
    if np.any(Hd) != 0 and Hd.values[0] > 0:
        H0data[k] = Hd.values[0]
    if np.any(Ld) != 0 and Ld.values[0] > 0:
        L0data[k] = Ld.values[0]
    if np.any(Isd) != 0 and Isd.values[0] > 0:
        Is0data[k] = Isd.values[0]
    if np.any(Iud) != 0 and Iud.values[0] > 0:
        Iu0data[k] = Iud.values[0]
    if Iu.values[0] > 0:
        Iudata[k] = Iu.values[0]
    if Is.values[0] > 0:
        Isdata[k] = Is.values[0]
    if np.any(Fuod) != 0:
        Fuodata[k] = Fuod.values[0]
    if np.any(Fudd) != 0:
        Fuddata[k] = Fudd.values[0]
    if np.any(Fsod) != 0:
        Fsodata[k] = Fsod.values[0]
    if np.any(Fsdd) != 0:
        Fsddata[k] = Fsdd.values[0]
        
    for t in range(T):
        for i in range(nrcp):
            for j in range(nssp):
                if t == 0:
                    pop = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[t])]['population']
                    gdp = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[t])]['gdp']
                    low = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[t])]['adults_u']
                    high = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[t])]['adults_s']
                    temp = y2000_data.loc[(y2000_data['iso3'] == Country_name[k])&(y2000_data['year'] == Year_name[t])]['temperature_mean']
                else:
                    pop = proj_data.loc[(proj_data['n'] == Country_name[k])&(proj_data['year'] == Year_name[t])&(proj_data['RCP'] == RCP_name[i])&(proj_data['SSP'] == SSP_name[j])]['population']
                    gdp = proj_data.loc[(proj_data['n'] == Country_name[k])&(proj_data['year'] == Year_name[t])&(proj_data['RCP'] == RCP_name[i])&(proj_data['SSP'] == SSP_name[j])]['gdp']
                    low = proj_data.loc[(proj_data['n'] == Country_name[k])&(proj_data['year'] == Year_name[t])&(proj_data['RCP'] == RCP_name[i])&(proj_data['SSP'] == SSP_name[j])]['adults_u']
                    high = proj_data.loc[(proj_data['n'] == Country_name[k])&(proj_data['year'] == Year_name[t])&(proj_data['RCP'] == RCP_name[i])&(proj_data['SSP'] == SSP_name[j])]['adults_s']
                    temp = proj_data.loc[(proj_data['n'] == Country_name[k])&(proj_data['year'] == Year_name[t])&(proj_data['RCP'] == RCP_name[i])&(proj_data['SSP'] == SSP_name[j])]['temperature_mean']

                Ydata[k, t, i, j] = gdp.values[0]
                Ldata[k, t, i, j] = low.values[0]
                Hdata[k, t, i, j] = high.values[0]
                hdata[k, t, i, j] = Hdata[k, t, i, j]/Ldata[k, t, i, j]
                Ndata[k, t, i, j] = Hdata[k, t, i, j] + Ldata[k, t, i, j]
                Pdata[k, t, i, j] = pop.values[0]
                Tempdata[k, t, i, j] = temp.values[0]

for k in range(nreg):
    for t in range(T - 1):
        for i in range(nrcp):
            for j in range(nssp):
                Xdata[k, t, i, j] = Pdata[k, t, i, j] - (Ndata[k, t, i, j] + Ndata[k, t + 1, i, j])
                #######################################################
                # Migration policy adjustments based on SSP assumptions #
                #######################################################
                # if j == 2:
                #     MCs[k, :, t, i, j] = 0.9
                #     MCu[k, :, t, i, j] = 0.9
                # else:
                #     if j == 4:
                #         MCs[k, :, t, i, j] = 0.1
                #         MCu[k, :, t, i, j] = 0.1
                #     else:
                #         MCs[k, :, t, i, j] = 0.5
                #         MCu[k, :, t, i, j] = 0.5  
                #######################################################
                
# Migration probability fixed effect   
betas = np.zeros((nreg, nreg))  
betau = np.zeros((nreg, nreg))

for k in range(nreg):
    #######################################################
    # Different growth rate of non-agri sector in developed/develping countries #
    #######################################################
    if hdata[k, 0, 0, 0]<1:
        Amgr[k] = 0.01 * Amgr[k]
    else:
        Amgr[k] = 10 * Amgr[k]
    #######################################################
    if N0data[k] == 0:
        N0data[k] = Ndata[k, 0, 0, 0] * (np.sum(N0data)/np.count_nonzero(N0data))/np.average(Ndata[:, 0, 0, 0])
    if Iudata[k] == 0:
        Iudata[k] = np.sum(Iudata)/np.count_nonzero(Iudata)
    if Isdata[k] == 0:
        Isdata[k] = np.sum(Isdata)/np.count_nonzero(Isdata)
    for kk in range(nreg):
        M00[k, kk] =  migflow0_data.loc[(migflow_data['country'] == Country_name[k])][Country_name[kk]]/1e6
        M0[k, kk] =  migflow_data.loc[(migflow_data['country'] == Country_name[k])][Country_name[kk]]/1e6
        prob0[k, kk] = migprob_data.loc[(migprob_data['country'] == Country_name[k])][Country_name[kk]]
        if k != kk:
            distance = hist_data.loc[(hist_data['origin'] == Country_name[k])&(hist_data['destination'] == Country_name[kk])&(hist_data['year'] == Year_name[0])]['distance']
            Dist[k, kk] = distance.values[0]
            if prob0[k, kk] == 0:
                prob0[k, kk] = min(1/(1000 * Ndata[k, 0, 0, 0]), 1e-6)
    for kk in range(nreg): 
        if k == kk:
            prob0[k, kk] = 1 - sum(prob0[k, :])
            M0[k, kk] = Ndata[k, 0, 0, 0]
    Ng[k, 0, 0, 0] = sum(M0[k, :])
    for kk in range(nreg):
        PMs[k, kk, 0, 0, 0] = prob0[k, kk]
        PMu[k, kk, 0, 0, 0] = prob0[k, kk]
        Ms[k, kk, 0, 0, 0] = PMs[k, kk, 0, 0, 0] * Ng[k, 0, 0, 0] * Hdata[k, 0, 0, 0]/Ndata[k, 0, 0, 0]
        Mu[k, kk, 0, 0, 0] = PMu[k, kk, 0, 0, 0] * Ng[k, 0, 0, 0] * Ldata[k, 0, 0, 0]/Ndata[k, 0, 0, 0]
        Mt[k, kk, 0, 0, 0] = Mu[k, kk, 0, 0, 0] + Ms[k, kk, 0, 0, 0]
        
        if kk != k:
            Mout_s[k, 0, 0, 0] = Mout_s[k, 0, 0, 0] + Ms[k, kk, 0, 0, 0]
            Min_s[kk, 0, 0, 0] = Min_s[kk, 0, 0, 0] + Ms[k, kk, 0, 0, 0]
            Mout_u[k, 0, 0, 0] = Mout_u[k, 0, 0, 0] + Mu[k, kk, 0, 0, 0]
            Min_u[kk, 0, 0, 0] = Min_u[kk, 0, 0, 0] + Mu[k, kk, 0, 0, 0]
            Min[kk, 0, 0, 0] = Min_u[kk, 0, 0, 0] + Min_s[kk, 0, 0, 0]
            Mout[k, 0, 0, 0] = Mout_u[k, 0, 0, 0] + Mout_s[k, 0, 0, 0]

    Mtotal_s[0, 0, 0] = Mtotal_s[0, 0, 0] + Mout_s[k, 0, 0, 0]
    Mtotal_u[0, 0, 0] = Mtotal_u[0, 0, 0] + Mout_u[k, 0, 0, 0]

Mtotal[0, 0, 0] = Mtotal_s[0, 0, 0] + Mtotal_u[0, 0, 0]

H0avg = (np.sum(H0data)/np.count_nonzero(H0data))/np.average(N0data)
L0avg = (np.sum(L0data)/np.count_nonzero(L0data))/np.average(N0data)
Iu0avg = (np.sum(Iu0data)/np.count_nonzero(Iu0data))/(np.sum(Iudata)/np.count_nonzero(Iudata))
Is0avg = (np.sum(Is0data)/np.count_nonzero(Is0data))/(np.sum(Isdata)/np.count_nonzero(Isdata))

for k in range(nreg):
    if H0data[k] == 0:
        H0data[k] = N0data[k] * H0avg
    if L0data[k] == 0:
        L0data[k] = N0data[k] * L0avg
    if Iu0data[k] == 0:
        Iu0data[k] = Iudata[k] * Iu0avg
    if Is0data[k] == 0:
        Is0data[k] = Isdata[k] * Is0avg
    for kk in range(nreg):        
        for i in range(nrcp):
            for j in range(nssp):
                Pop[k, 0, i, j] = Pdata[k, 0, 0, 0]
                Ng[k, 0, i, j] = Ng[k, 0, 0, 0]
                PMs[k, kk, 0, i, j] = PMs[k, kk, 0, 0, 0]
                PMu[k, kk, 0, i, j] = PMu[k, kk, 0, 0, 0]
                Ms[k, kk, 0, i, j] = Ms[k, kk, 0, 0, 0]
                Mu[k, kk, 0, i, j] = Mu[k, kk, 0, 0, 0]
                Mt[k, kk, 0, i, j] = Mt[k, kk, 0, 0, 0]
                Mout_s[k, 0, i, j] = Mout_s[k, 0, 0, 0]
                Mout_u[k, 0, i, j] = Mout_u[k, 0, 0, 0]
                Mout[k, 0, i, j] = Mout[k, 0, 0, 0]
                Min_s[k, 0, i, j] = Min_s[k, 0, 0, 0]
                Min_u[k, 0, i, j] = Min_u[k, 0, 0, 0]
                Min[k, 0, i, j] = Min[k, 0, 0, 0]
                Mtotal_s[0, i, j] = Mtotal_s[0, 0, 0]
                Mtotal_u[0, i, j] = Mtotal_u[0, 0, 0]
                Mtotal[0, i, j] = Mtotal[0, 0, 0]
for i in range(nrcp):
    for j in range(nssp):
        Htotal[0, i, j] = sum(Hdata[:, 0, i, j])
        Ltotal[0, i, j] = sum(Ldata[:, 0, i, j])
        htotal[0, i, j] = Htotal[0, i, j]/Ltotal[0, i, j]
        
### ============================================== Model Calibration ============================================== #
def Calib(hdata0, Ndata0, Ydata0, Amgrx, tugx, tsgx, Iu0, Is0, Temp0, Nd00, kx, tx, ix, jx):

    hdatax = hdata[kx, tx, ix, jx]
    
    popg0 = Ndata0/Nd00
    nu0 = popg0 / (1 + hdata0)
    ns0 = nu0 * hdata0
    
    Ir0 = Is0/Iu0
    tu0 = gamma0 / (Ir0 * ns0 + nu0)
    ts0 = tu0 * Ir0
    tr0 = Ir0
     
    tux = tu0 * (1 + tugx)**((Year_name[tx] - Tstart)/Tstep)
    tsx = ts0 * (1 + tsgx)**((Year_name[tx] - Tstart)/Tstep)
    trx = tsx/tux
             
    Da0 = max(0.1, g0a + g1a * Temp0 + g2a * Temp0**2)
    Dm0 = max(0.1, g0m + g1m * Temp0 + g2m * Temp0**2)
    Dr0 = Dm0/Da0
     
    L0 = nu0 * Nd00
    H0 = ns0 * Nd00
  
    Ar0 = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(tr0) - np.log(hdata0)) /(1 - eps) - np.log(Dr0))
    Am0 = Ydata0/((alpha * (L0**theta * Da0 / Ar0)**((eps - 1)/eps) + (1 - alpha) * (H0**theta * Dm0)**((eps - 1)/eps))**(eps/(eps - 1)))
    Aa0 = Am0/Ar0
     
    Arx = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(trx) + ((eps - 1) * theta - eps) * np.log(hdatax)) /(1 - eps) - np.log(Dr0))
    Arg = np.exp((np.log(Arx/Ar0))/((Year_name[tx] - Tstart)/Tstep)) - 1
     
    Amgx = (1 + Amgrx)**Tstep - 1
    Aagx = (1 + Amgx)/(1 + Arg) - 1
     
    Ya0 = Aa0 * L0 * Da0
    Ym0 = Am0 * H0 * Dm0
    Yr0 = Ym0 / Ya0
     
    pr0 = (Yr0)**(-1/eps) * (1 - alpha) / alpha
     
    cmu0 = Ym0 / (H0 * tr0 + L0)
    cms0 = cmu0 * tr0
    cau0 = Ya0 / (H0 * tr0 + L0)
    cas0 = cau0 * tr0    
    cu0 = (alpha * cau0**((eps - 1)/eps) + (1 - alpha) * cmu0**((eps - 1)/eps))**(eps/(eps - 1))
    cs0 = (alpha * cas0**((eps - 1)/eps) + (1 - alpha) * cms0**((eps - 1)/eps))**(eps/(eps - 1))
    wu0 = cu0 / (1 - gamma0)
    ws0 = cs0 / (1 - gamma0)
    pa0 = wu0 / (theta * Da0 * Aa0 * L0**(theta - 1))
    pm0 = ws0 / (theta * Dm0 * Am0 * H0**(theta - 1))
    wr0 = ws0/wu0
     
    Outputx = [Da0, Dm0, Dr0, Ndata0, H0, L0, hdata0, Ydata0, Ya0, Ym0, Yr0, Aa0, Am0, Ar0, cmu0, cms0, cau0, cas0, cu0, cs0, wu0, ws0, wr0, pa0, pm0, pr0]
    Ratex = [Aagx, Amgx, tu0, ts0, tr0]
    return (Outputx, Ratex)
 
# ============================================== Transition Function ============================================== #
def State(num):   
    NM = [0] * nreg
    HM = [0] * nreg
    LM = [0] * nreg
    nuM = [0] * nreg
    nsM = [0] * nreg
    hM = [0] * nreg
    YaM = [0] * nreg
    YmM = [0] * nreg
    YM = [0] * nreg
    wrM = [0] * nreg
    cmuM = [0] * nreg
    cmsM = [0] * nreg
    cauM = [0] * nreg
    casM = [0] * nreg
    cuM = [0] * nreg
    csM = [0] * nreg
    wuM = [0] * nreg
    wsM = [0] * nreg
    paM = [0] * nreg
    pmM = [0] * nreg
    prM = [0] * nreg
    NG = [0] * nreg
    MigsM = np.zeros((nreg, nreg))
    MiguM = np.zeros((nreg, nreg))
    MigM = np.zeros((nreg, nreg))
    for kx in range(nreg):
        NG[kx] = (num[kx, 0] + num[kx, 1]) * PopM[kx]
        for ky in range(kx, nreg):
            MiguM[ky, kx] = num[ky, 0] * PopM[ky] * buM[ky, kx]
            MigsM[ky, kx] = num[ky, 1] * PopM[ky] * bsM[ky, kx]
            MigM[ky, kx] = MiguM[ky, kx] + MigsM[ky, kx] 
            MiguM[kx, ky] = num[kx, 0] * PopM[kx] * buM[kx, ky]
            MigsM[kx, ky] = num[kx, 1] * PopM[kx] * bsM[kx, ky] 
            MigM[kx, ky] = MiguM[kx, ky] + MigsM[kx, ky]                   
        HM[kx] = sum(MigsM[:, kx])
        LM[kx] = sum(MiguM[:, kx])
        nuM[kx] = LM[kx]/PopM[kx]
        nsM[kx] = HM[kx]/PopM[kx]
        NM[kx] = HM[kx] + LM[kx]
        hM[kx] = HM[kx]/LM[kx]
        YaM[kx] = AaM[kx] * LM[kx]**theta * DaM[kx]
        YmM[kx] = AmM[kx] * HM[kx]**theta * DmM[kx]
        wrM[kx] = np.exp(np.log((1 - alpha)/alpha) + ((eps - 1) * theta - eps)/eps * np.log(hM[kx]) - (1 - eps)/eps * np.log(DrM[kx] * ArM[kx]))
        cmuM[kx] = YmM[kx] / (HM[kx] * wrM[kx] + LM[kx])
        cmsM[kx] = cmuM[kx] * wrM[kx]
        cauM[kx] = YaM[kx] / (HM[kx] * wrM[kx] + LM[kx])
        casM[kx] = cauM[kx] * wrM[kx]
        cuM[kx] = (alpha * cauM[kx]**((eps - 1)/eps) + (1 - alpha) * cmuM[kx]**((eps - 1)/eps))**(eps/(eps - 1))
        csM[kx] = (alpha * casM[kx]**((eps - 1)/eps) + (1 - alpha) * cmsM[kx]**((eps - 1)/eps))**(eps/(eps - 1))
        wuM[kx] = cuM[kx] / (1 - gammaN[kx])
        wsM[kx] = csM[kx] / (1 - gammaN[kx])
        paM[kx] = wuM[kx] / (theta * DaM[kx] * AaM[kx] * LM[kx]**(theta - 1))
        pmM[kx] = wsM[kx] / (theta * DmM[kx] * AmM[kx] * HM[kx]**(theta - 1))
        prM[kx] = pmM[kx]/paM[kx]   
        YM[kx] = YaM[kx] * paM[kx] + YmM[kx] * pmM[kx]
        OutputM = [nuM, nsM, NG, NM, HM,LM, hM, paM, pmM, prM, YM, YaM, YmM, wrM, cmuM, cmsM, cauM, casM, cuM, csM, wuM, wsM, MigsM, MiguM, MigM]
    return(OutputM)
 
# ============================================== Model Dynamics ============================================== #
for i in range(nrcp):
    for j in range(nssp):
        for k in range(nreg):
            [Output, Rate] = Calib(hdata[k, 0, i, j], Ndata[k, 0, i, j], Ydata[k, 0, i, j], Amgr[k], tug[k], tsg[k], Iudata[k], Isdata[k], Tempdata[k, 0, i, j], N0data[k], k, 2, i, j)
            [Aag[k, i, j], Amg[k, i, j], tu[k, 0, i, j], ts[k, 0, i, j], tr[k, 0, i, j]] = Rate
            [Da[k, 0, i, j], Dm[k, 0, i, j], Dr[k, 0, i, j], N[k, 0, i, j], H[k, 0, i, j], L[k, 0, i, j], h[k, 0, i, j], Y[k, 0, i, j], Ya[k, 0, i, j], Ym[k, 0, i, j], Yr[k, 0, i, j], Aa[k, 0, i, j], Am[k, 0, i, j], Ar[k, 0, i, j], cmu[k, 0, i, j], cms[k, 0, i, j], cau[k, 0, i, j], cas[k, 0, i, j], cu[k, 0, i, j], cs[k, 0, i, j], wu[k, 0, i, j], ws[k, 0, i, j], wr[k, 0, i, j], pa[k, 0, i, j], pm[k, 0, i, j], pr[k, 0, i, j]] = Output
         
        for t in range(T - 1):
            for k in range(nreg):
             
                tu[k, t + 1, i, j] = tu[k, t, i, j] * (1 + tug[k])
                ts[k, t + 1, i, j] = ts[k, t, i, j] * (1 + tsg[k])
                tr[k, t + 1, i, j] = ts[k, t, i, j]/tu[k, t, i, j]
                 
                Da[k, t + 1, i, j] = max(0.1, g0a + g1a * Tempdata[k, t + 1, i, j] + g2a * Tempdata[k, t + 1, i, j]**2)
                Dm[k, t + 1, i, j] = max(0.1, g0m + g1m * Tempdata[k, t + 1, i, j] + g2m * Tempdata[k, t + 1, i, j]**2)
                Dr[k, t + 1, i, j] = Dm[k, t + 1, i, j]/Da[k, t + 1, i, j]
                #######################################################                 
                # Fixed Climate Change ################################
                #######################################################  
#                Da[k, t + 1, i, j] = Da[k, 0, i, j]
#                Dm[k, t + 1, i, j] = Dm[k, 0, i, j]
#                Dr[k, t + 1, i, j] = Dr[k, 0, i, j]
                #######################################################   
                Aa[k, t + 1, i, j] = Aa[k, t, i, j] * (1 + Aag[k, i, j])
                Am[k, t + 1, i, j] = Am[k, t, i, j] * (1 + Amg[k, i, j])
                Ar[k, t + 1, i, j] = Am[k, t + 1, i, j]/Aa[k, t + 1, i, j]
             
                for kk in range(nreg):
                    if kk == k:
                        PMs[k, kk, t, i, j] = 0
                        PMu[k, kk, t, i, j] = 0
                    else:
                        if t == 0 and i == 0 and j == 0:
    
                            # fixed effect for estimating probability of migration
                            betas[k, kk] = (1000 * prob0[k, kk]) - (Coef_s[0] * np.log(H0data[k]*1e6) + Coef_s[1] * np.log(H0data[kk]*1e6) + Coef_s[2] * np.log(Is0data[kk]/Is0data[k]) + Coef_s[3] * np.log(Dist[k, kk]) + Fsodata[k] + Fsddata[kk])
                            betau[k, kk] = (1000 * prob0[k, kk]) - (Coef_u[0] * np.log(L0data[k]*1e6) + Coef_u[1] * np.log(L0data[kk]*1e6) + Coef_u[2] * np.log(Iu0data[kk]/Iu0data[k]) + Coef_u[3] * np.log(Dist[k, kk]) + Fuodata[k] + Fuddata[kk])
    
                        # probability of migration
                        PMs[k, kk, t, i, j] = max(0, min(1000, (betas[k, kk] + (Coef_s[0] * np.log(H[k, t, i, j]*1e6) + Coef_s[1] * np.log(H[kk, t, i, j]*1e6) + Coef_s[2] * np.log(ws[kk, t, i, j]/ws[k, t, i, j]) + Coef_s[3] * np.log(Dist[k, kk]) + Fsodata[k] + Fsddata[kk])))) * (1 - MCs[k, kk, t, i, j])/1000
                        PMu[k, kk, t, i, j] = max(0, min(1000, (betau[k, kk] + (Coef_u[0] * np.log(L[k, t, i, j]*1e6) + Coef_u[1] * np.log(L[kk, t, i, j]*1e6) + Coef_u[2] * np.log(wu[kk, t, i, j]/wu[k, t, i, j]) + Coef_u[3] * np.log(Dist[k, kk]) + Fuodata[k] + Fuddata[kk])))) * (1 - MCu[k, kk, t, i, j])/1000
                        #######################################################                 
                        # No Migration ########################################
                        #######################################################  
#                        PMs[k, kk, t, i, j] = 0
#                        PMu[k, kk, t, i, j] = 0
                        #######################################################     
                        # Fixed Migration #####################################
                        #######################################################  
#                        PMs[k, kk, t, i, j] = prob0[k, kk]
#                        PMu[k, kk, t, i, j] = prob0[k, kk]
                        #######################################################
                PMs[k, k, t, i, j] = 1 - sum(PMs[k, :, t, i, j])
                PMu[k, k, t, i, j] = 1 - sum(PMu[k, :, t, i, j])
         
            PopM = [0] * nreg
            DaM = [0] * nreg
            DmM = [0] * nreg
            DrM = [0] * nreg
            AaM = [0] * nreg
            AmM = [0] * nreg
            ArM = [0] * nreg
            trM = [0] * nreg
            bsM = np.zeros((nreg, nreg))
            buM = np.zeros((nreg, nreg))
            mcsM = np.zeros((nreg, nreg))
            mcuM = np.zeros((nreg, nreg))
            nsolx = np.zeros((nreg, 2))
            for k in range(nreg):
                trs = 1
                tru = 1
                PopM[k] = N[k, t, i, j]
                DaM[k] = Da[k, t + 1, i, j]
                DmM[k] = Dm[k, t + 1, i, j]
                DrM[k] = Dr[k, t + 1, i, j]
                AaM[k] = Aa[k, t + 1, i, j]
                AmM[k] = Am[k, t + 1, i, j]
                ArM[k] = Ar[k, t + 1, i, j]
                for kk in range(nreg):
                    buM[k, kk] = PMu[k, kk, t, i, j]
                    bsM[k, kk] = PMs[k, kk, t, i, j]
                    if kk != k:
                        mcuM[k, kk] = MCu[k, kk, t, i, j]
                        mcsM[k, kk] = MCs[k, kk, t, i, j]
                    trs = trs + bsM[k, kk] * mcsM[k, kk]
                    tru = tru + buM[k, kk] * mcuM[k, kk]
                tu[k, t, i, j] = tu[k, t, i, j] * tru
                ts[k, t, i, j] = ts[k, t, i, j] * trs
                tr[k, t, i, j] = ts[k, t, i, j]/tu[k, t, i, j]
                trM[k] = tr[k, t, i, j]
#                hn[k, t + 1, i, j] = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(wr[k, t, i, j]) - (1 - eps) * np.log(ArM[k]) - (1 - eps) * np.log(DrM[k]))/(eps - (eps - 1) * theta))
                hn[k, t + 1, i, j] = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(trM[k]) - (1 - eps) * np.log(ArM[k]) - (1 - eps) * np.log(DrM[k]))/(eps - (eps - 1) * theta))
                hN = hn[k, t + 1, i, j]
                nsolx[k, 0] = gamma0/(tu[k, t, i, j] + hN * ts[k, t, i, j])
                nsolx[k, 1] = nsolx[k, 0] * hN
                #######################################################
                ## Following SSP Population projections ###############
                #######################################################
#                NN = Ndata[k, t + 1, i, j]/Ndata[k, t, i, j]
#                nsolx[k, 0] = NN/(1 + hN)
#                nsolx[k, 1] = nsolx[k, 0] * hN
                #######################################################
                gamma[k, t, i, j] = tu[k, t, i, j] * nsolx[k, 0] + ts[k, t, i, j] * nsolx[k, 1]
                gammaN[k] = gamma[k, t, i, j]
            Outputsol = State(nsolx)
            [nu[:, t, i, j], ns[:, t, i, j], Ng[:, t + 1, i, j], N[:, t + 1, i, j], H[:, t + 1, i, j],L[:, t + 1, i, j], h[:, t + 1, i, j], pa[:, t + 1, i, j], pm[:, t + 1, i, j], pr[:, t + 1, i, j], Y[:, t + 1, i, j], Ya[:, t + 1, i, j], Ym[:, t + 1, i, j], wr[:, t + 1, i, j], cmu[:, t + 1, i, j], cms[:, t + 1, i, j], cau[:, t + 1, i, j], cas[:, t + 1, i, j], cu[:, t + 1, i, j], cs[:, t + 1, i, j], wu[:, t + 1, i, j], ws[:, t + 1, i, j], Ms[:, :, t + 1, i, j], Mu[:, :, t + 1, i, j], Mt[:, :, t + 1, i, j]] = Outputsol
            Pop[:, t, i, j] = (N[:, t, i, j] + N[:, t + 1, i, j]) + Xdata[:, t, i, j]
            Yp[:, t, i, j] = Y[:, t, i, j]/N[:, t, i, j]
 
            for k in range(nreg):
                for kk in range(nreg):
                    if kk != k:
                        Mout_s[k, t + 1, i, j] = Mout_s[k, t + 1, i, j] + Ms[k, kk, t + 1, i, j]
                        Min_s[kk, t + 1, i, j] = Min_s[kk, t + 1, i, j] + Ms[k, kk, t + 1, i, j]
                        Mout_u[k, t + 1, i, j] = Mout_u[k, t + 1, i, j] + Mu[k, kk, t + 1, i, j]
                        Min_u[kk, t + 1, i, j] = Min_u[kk, t + 1, i, j] + Mu[k, kk, t + 1, i, j]
                        Min[kk, t + 1, i, j] = Min_u[kk, t + 1, i, j] + Min_s[kk, t + 1, i, j]
                        Mout[k, t + 1, i, j] = Mout_u[k, t + 1, i, j] + Mout_s[k, t + 1, i, j]
                Mtotal_s[t + 1, i, j] = Mtotal_s[t + 1, i, j] + Mout_s[k, t + 1, i, j]
                Mtotal_u[t + 1, i, j] = Mtotal_u[t + 1, i, j] + Mout_u[k, t + 1, i, j]
            Mtotal[t + 1, i, j] = Mtotal_s[t + 1, i, j] + Mtotal_u[t + 1, i, j]
            Htotal[t + 1, i, j] = sum(H[:, t + 1, i, j])
            Ltotal[t + 1, i, j] = sum(L[:, t + 1, i, j])
            htotal[t + 1, i, j] = Htotal[t + 1, i, j]/Ltotal[t + 1, i, j]
# ===================================================== Output ===================================================== #    
 
#x = Year_name
#x1 = Year_name[:T-1]
#x2 = Year_name[1:]
# 
#for k in range(nreg):
#    plt.plot(x, hdata[k,:,0], 'b--', label = "Data - SSP 1")
#    plt.plot(x, h[k, :, 0], 'b', label = "SSP 1")
#    plt.plot(x, hdata[k,:,1], 'g--', label = "Data - SSP 2")
#    plt.plot(x, h[k, :, 1], 'g', label = "SSP 2")
#    plt.plot(x, hdata[k,:,2], 'k--', label = "Data - SSP 3")
#    plt.plot(x, h[k, :, 2], 'k', label = "SSP 3")
#    plt.plot(x, hdata[k,:,3], 'y--', label = "Data - SSP 4")
#    plt.plot(x, h[k, :, 3], 'y', label = "SSP 4")
#    plt.plot(x, hdata[k,:,4], 'r--', label = "Data - SSP 5")
#    plt.plot(x, h[k, :, 4], 'r', label = "SSP 5")
#    plt.xlabel('Time')
#    plt.ylabel('ratio')
#    plt.title('Ratio of high-skilled to low-skilled labor in ' + Regname[k] + ' region')
#    axes = plt.gca()
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(loc=2, prop={'size':8})
#    plt.show()
#     
#    plt.plot(x, Ndata[k,:,0], 'b--', label = "Data - SSP 1")
#    plt.plot(x, N[k, :, 0], 'b', label = "SSP 1")
#    plt.plot(x, Ndata[k,:,1], 'g--', label = "Data - SSP 2")
#    plt.plot(x, N[k, :, 1], 'g', label = "SSP 2")
#    plt.plot(x, Ndata[k,:,2], 'k--', label = "Data - SSP 3")
#    plt.plot(x, N[k, :, 2], 'k', label = "SSP 3")
#    plt.plot(x, Ndata[k,:,3], 'y--', label = "Data - SSP 4")
#    plt.plot(x, N[k, :, 3], 'y', label = "SSP 4")
#    plt.plot(x, Ndata[k,:,4], 'r--', label = "Data - SSP 5")
#    plt.plot(x, N[k, :, 4], 'r', label = "SSP 5")
#    plt.xlabel('Time')
#    plt.ylabel('million')
#    plt.title('Adult population in ' + Regname[k] + ' region')
#    axes = plt.gca()
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(loc=2, prop={'size':8})
#    plt.show()    
#     
##    plt.plot(x1, Ypdata[k,:T-1, 0] * 1e3, 'b--', label = "Data - SSP 1")
##    plt.plot(x1, Yp[k, :T-1, 0] * 1e3, 'b', label = "SSP 1")
##    plt.plot(x1, Ypdata[k,:T-1, 1] * 1e3, 'g--', label = "Data - SSP 2")
##    plt.plot(x1, Yp[k, :T-1, 1] * 1e3, 'g', label = "SSP 2")
##    plt.plot(x1, Ypdata[k,:T-1, 2] * 1e3, 'k--', label = "Data - SSP 3")
##    plt.plot(x1, Yp[k, :T-1, 2] * 1e3, 'k', label = "SSP 3")
##    plt.plot(x1, Ypdata[k,:T-1, 3] * 1e3, 'y--', label = "Data - SSP 4")
##    plt.plot(x1, Yp[k, :T-1, 3] * 1e3, 'y', label = "SSP 4")
##    plt.plot(x1, Ypdata[k,:T-1, 4] * 1e3, 'r--', label = "Data - SSP 5")
##    plt.plot(x1, Yp[k, :T-1, 4] * 1e3, 'r', label = "SSP 5")
##    plt.xlabel('Time')
##    plt.ylabel('USD/capita')
##    plt.title('GDP per capita in ' + Regname[k] + ' region')
##    axes = plt.gca()
##    plt.xticks(np.arange(min(x1), max(x1) + 1, 20))
##    plt.legend(loc=2, prop={'size':8})
##    plt.show() 
#     
#for j in range(nssp):
## =============================================== low-skilled =============================================== #
#
#    plt.stackplot(x, [(L[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of people')
#    plt.title('low-skilled labor in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
#    
#    plt.stackplot(x, [(Mout_u[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of migrants')
#    plt.title('Number of low-skilled migrants by origin in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()   
#    
#    plt.stackplot(x, [(Min_u[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of migrants')
#    plt.title('Number of low-skilled migrants by destination in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
## =============================================== High-skilled =============================================== #
#
#    plt.stackplot(x, [(H[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of people')
#    plt.title('high-skilled labor in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()  
#    
#    plt.stackplot(x, [(Mout_s[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of migrants')
#    plt.title('Number of high-skilled migrants by origin in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
#
#    plt.stackplot(x, [(Min_s[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of migrants')
#    plt.title('Number of high-skilled migrants by destination in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
## =============================================== Total =============================================== #
#
#    plt.stackplot(x1, [(Pop[k, :T-1, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of people')
#    plt.title('Population in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x1), max(x1) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
#
#    plt.stackplot(x, [(Mout[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of migrants')
#    plt.title('Number of migrants by origin in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
# 
#    plt.stackplot(x, [(Min[k, :, j])*1e6 for k in range(nreg)],
#                      labels=[RegName[k] for k in range(nreg)], colors = [pal[k] for k in range(nreg)])
#    plt.xlabel('Time')
#    plt.ylabel('Number of migrants')
#    plt.title('Number of migrants by destination in SSP' + str(j + 1))
#    axes = plt.gca()
#    axes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
#
## =============================================== Wages =============================================== #
# 
#for j in range(nssp):
#    for k in range(nreg):
#        plt.plot(x, wr[k,:,j], label=RegName[k], color = pal[k])
#    plt.xlabel('Time')
#    plt.ylabel('ratio of wages')
#    plt.title('Ratio of high-skilled to low-skilled wages in SSP' + str(j + 1))
#    axes = plt.gca()
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
#    
#    for k in range(nreg):
#        ws1 = [ws[k,i,j]/np.average(ws[:,i,j]) for i in range(T)]
#        plt.plot(x, ws1, label=RegName[k], color = pal[k])
#    plt.xlabel('Time')
#    plt.ylabel('ratio of wages to global average')
#    plt.title('Ratio of high-skilled wages to global average in SSP' + str(j + 1))
#    axes = plt.gca()
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()
#
#    for k in range(nreg):
#        wu1 = [wu[k,i,j]/np.average(wu[:,i,j]) for i in range(T)]
#        plt.plot(x, wu1, label=RegName[k], color = pal[k])
#    plt.xlabel('Time')
#    plt.ylabel('ratio of wages to global average')
#    plt.title('Ratio of low-skilled wages to global average in SSP' + str(j + 1))
#    axes = plt.gca()
#    plt.xticks(np.arange(min(x), max(x) + 1, 20))
#    plt.legend(bbox_to_anchor=(0., 1.1, 1., .11), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
#    plt.show()


# =============================================== Export into Excel =============================================== #
 
#def output(filename, sheet1, out1, out2):
#    book = xlwt.Workbook(filename)
#    sht = book.add_worksheet(sheet1)    
#    var_name = ['tu', 'ts', 'nu', 'ns', 'GrossN', 'N', 'L', 'H', 'Aa', 'Am', 'Da', 'Dm', 'Y', 'Ya', 'Ym', 'cu', 'cs', 'wu', 'ws', 'Min', 'Mout', 'MCu', 'MCs', 'Mu', 'Ms', 'Mt', 'betas', 'betau']
#    sht.write(0, 0, 'variable')
#    sht.write(0, 1, 'year')
#    sht.write(0, 2, 'RCP')
#    sht.write(0, 3, 'SSP')
#    sht.write(0, 4, 'origin')
#    sht.write(0, 5, 'destination')
#    sht.write(0, 6, 'value')
#     
#    outpt = [out1, out2]
#    sz1 = len(out1)
#    sz2 = len(out2)
#    outsz = [sz1, sz2]
#    for i in range(2):
#        for varx in range(outsz[i]):
#            for rcpx in range(nrcp):
#                for sspx in range(nssp):
#                    for tx in range(T):
#                        for orgx in range(nreg):
#                            for destx in range(nreg):
#                                if i == 0:
#                                    destname = ' '
#                                    valx = outpt[i][varx][orgx][tx][rcpx][sspx]
#                                else:
#                                    destname = Country_name[destx]
#                                    if varx > 4:
#                                        valx = outpt[i][varx][orgx][destx]
#                                    else:
#                                        valx = outpt[i][varx][orgx][destx][tx][rcpx][sspx]
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 0, var_name[i * outsz[0] + varx])
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 1, Year_name[tx])
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 2, RCP_name[rcpx])
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 3, SSP_name[sspx])
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 4, Country_name[orgx])
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 5, destname)
#                                sht.write(1 + i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx, 6, valx)
#                                if i == 0:
#                                    destx = nreg - 1
#    book.close()
# 
#output1 = [tu, ts, nu, ns, Pop, L, H, Aa, Am, Da, Dm, Y, Ya, Ym, cu, cs, wu, ws, Min, Mout]
#output2 = [MCu, MCs, Mu, Ms, Mt, betas, betau]
#output('IMT_Total.xlsx', 'Sheet1', output1, output2)

# =============================================== Export into CSV =============================================== #

def output(out1, out2, var_name):
    sz1 = len(out1)
    sz2 = len(out2)
    outpt = [out1, out2]
    outsz = [sz1, sz2]
    
    lngt = (sz1 * nreg * T * nrcp * nssp + sz2 * nreg * nreg * T * nrcp * nssp)
    var_col = [0] * lngt
    year_col = [0] * lngt
    rcp_col = [0] * lngt
    ssp_col = [0] * lngt
    org_col = [0] * lngt
    dst_col = [0] * lngt
    val_col = [0] * lngt
    
    for i in range(2):
        for varx in range(outsz[i]):
            for rcpx in range(nrcp):
                for sspx in range(nssp):
                    for tx in range(T):
                        for orgx in range(nreg):
                            for destx in range(nreg):
                                if i == 0:
                                    destname = ' '
                                    valx = outpt[i][varx][orgx][tx][rcpx][sspx]
                                else:
                                    destname = Country_name[destx]
                                    if varx > 4:
                                        valx = outpt[i][varx][orgx][destx]
                                    else:
                                        valx = outpt[i][varx][orgx][destx][tx][rcpx][sspx]
                                var_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = var_name[i * outsz[0] + varx]
                                year_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = Year_name[tx]
                                rcp_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = RCP_name[rcpx]
                                ssp_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = SSP_name[sspx]
                                org_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = Country_name[orgx]
                                dst_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = destname
                                val_col[i * outsz[0] * nrcp * nssp * T * nreg + varx * nrcp * nssp * T * nreg * (1 + (nreg - 1) * i) + rcpx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = valx
                                if i == 0:
                                    destx = nreg - 1
    data = {'variable':var_col, 'year':year_col, 'RCP':rcp_col, 'SSP':ssp_col, 'origin':org_col, 'destination':dst_col, 'value':val_col}
    df1 = pd.DataFrame(data)
    return(df1)
 
#output1 = [tu, ts, nu, ns, Pop, N, L, H, Aa, Am, Da, Dm, Y, Ya, Ym, cu, cs, wu, ws, Min, Mout]
#output2 = [MCu, MCs, Mu, Ms, Mt, betas, betau]
#varb = ['tu', 'ts', 'nu', 'ns', 'Pop', 'N', 'L', 'H', 'Aa', 'Am', 'Da', 'Dm', 'Y', 'Ya', 'Ym', 'cu', 'cs', 'wu', 'ws', 'Min', 'Mout', 'MCu', 'MCs', 'Mu', 'Ms', 'Mt', 'betas', 'betau']
#resultdf = output(output1, output2, varb)
#resultdf.to_csv ('MIT_result_NEW.csv', index = None, header=True)

# =============================================== Export into CSV only one RCP =============================================== #

def outputrcp(out1, out2, var_name, rcp_n):
    sz1 = len(out1)
    sz2 = len(out2)
    outpt = [out1, out2]
    outsz = [sz1, sz2]
    
    lngt = (sz1 * nreg * T * nssp + sz2 * nreg * nreg * T * nssp)
    var_col = [0] * lngt
    year_col = [0] * lngt
    ssp_col = [0] * lngt
    org_col = [0] * lngt
    dst_col = [0] * lngt
    val_col = [0] * lngt
    
    for i in range(2):
        for varx in range(outsz[i]):
            for sspx in range(nssp):
                for tx in range(T):
                    for orgx in range(nreg):
                        for destx in range(nreg):
                            if i == 0:
                                destname = ' '
                                valx = outpt[i][varx][orgx][tx][rcp_n][sspx]
                            else:
                                destname = Country_name[destx]
                                if varx > 4:
                                    valx = outpt[i][varx][orgx][destx]
                                else:
                                    valx = outpt[i][varx][orgx][destx][tx][rcp_n][sspx]
                            var_col[i * sz1 * nssp * T * nreg + varx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = var_name[i * sz1 + varx]
                            year_col[i * sz1 * nssp * T * nreg + varx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = Year_name[tx]
                            ssp_col[i * sz1 * nssp * T * nreg + varx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = SSP_name[sspx]
                            org_col[i * sz1 * nssp * T * nreg + varx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = Country_name[orgx]
                            dst_col[i * sz1 * nssp * T * nreg + varx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = destname
                            val_col[i * sz1 * nssp * T * nreg + varx * nssp * T * nreg * (1 + (nreg - 1) * i) + sspx * T * nreg * (1 + (nreg - 1) * i) + tx * nreg * (1 + (nreg - 1) * i) + orgx * (1 + (nreg - 1) * i) + destx] = valx
                            if i == 0:
                                destx = nreg - 1
    data = {'variable':var_col, 'year':year_col, 'SSP':ssp_col, 'origin':org_col, 'destination':dst_col, 'value':val_col}
    df1 = pd.DataFrame(data)
    return(df1)
 
output1 = [tu, ts, nu, ns, Pop, N, L, H, Aa, Am, Da, Dm, Y, Ya, Ym, Yp, cu, cs, wu, ws, Min, Mout]
output2 = [MCu, MCs, Mu, Ms, Mt, betas, betau]
varb = ['tu', 'ts', 'nu', 'ns', 'Pop', 'N', 'L', 'H', 'Aa', 'Am', 'Da', 'Dm', 'Y', 'Ya', 'Ym', 'Yp', 'cu', 'cs', 'wu', 'ws', 'Min', 'Mout', 'MCu', 'MCs', 'Mu', 'Ms', 'Mt', 'betas', 'betau']
# resultdfrcp26 = outputrcp(output1, output2, varb, 0)
# resultdfrcp26.to_csv ('G:/My Drive/05-IMT_International Migration/IMR-2nd round/IMT_RCP26-NP.csv', index = None, header=True)
# resultdfrcp45 = outputrcp(output1, output2, varb, 1)
# resultdfrcp45.to_csv ('G:/My Drive/05-IMT_International Migration/IMR-2nd round/IMT_RCP45-NP.csv', index = None, header=True)
# resultdfrcp60 = outputrcp(output1, output2, varb, 2)
# resultdfrcp60.to_csv ('G:/My Drive/05-IMT_International Migration/IMR-2nd round/IMT_RCP60-NP.csv', index = None, header=True)
# resultdfrcp85 = outputrcp(output1, output2, varb, 3)
# resultdfrcp85.to_csv ('G:/My Drive/05-IMT_International Migration/IMR-2nd round/IMT_RCP85-NP.csv', index = None, header=True)

# =============================================== PLOTS =============================================== #

sms = np.zeros((nreg*nreg,T,nssp))
sps = np.zeros((nreg*nreg,T,nssp))
smu = np.zeros((nreg*nreg,T,nssp))
spu = np.zeros((nreg*nreg,T,nssp))

jrcp = 2

for t in range(T-1):
    for j in range(nssp):
        for k in range(nreg):
            for kk in range(nreg):
                if kk != k and Ms[k,kk,t+1,jrcp,j]<2:
                    sms[k*nreg+kk,t+1,j]=Ms[k,kk,t+1,jrcp,j]
                    sps[k*nreg+kk,t,j]=PMs[k,kk,t,jrcp,j]
                    smu[k*nreg+kk,t+1,j]=Mu[k,kk,t+1,jrcp,j]
                    spu[k*nreg+kk,t,j]=PMu[k,kk,t,jrcp,j]
                    
fig = plt.figure(figsize=(20,20))
for t in range(T-1):
    for j in range(nssp):                    
        # Create plot
#        fig = plt.figure(figsize=(5,5))j*nssp+t+1
        plt.subplot(5, 5, t*(T-1)+j+1)
        plt.scatter(sms[:,t+1,j], sps[:,t,j], marker='*', c='r', label = 'High-skilled')
        plt.scatter(smu[:,t+1,j], spu[:,t,j], marker='*', c='b', label = 'Low-skilled')
#        plt.xlabel('migration flow (milion)', fontsize=16)
#        plt.ylabel('migration probability', fontsize=16)
        plt.xlim(right=np.max(sms)+0.5)
        plt.ylim(top=np.max(sps)+0.01)
        if t == 0:
            plt.title(f'{SSP_name[j]}', fontsize=30)
        if j == 0:
             plt.ylabel(f'{Year_name[t]} - {Year_name[t+1]}', fontsize=30)
#        plt.legend()
#        fig.savefig(f'{SSP_name[j]}_{Year_name[t]}.png', bbox_inches='tight', dpi=150)
# fig.savefig('migprob.png', bbox_inches='tight', dpi=150)
plt.show()

# =============================================== examples =============================================== #

cind=[19, 27, 43, 38, 67, 66, 94, 105, 112, 80, 123, 157, 150]
nind = np.size(cind)
Nind = np.zeros((nind, nssp))
Ndataind = np.zeros((nind, nssp))
Mind = np.zeros((nind, nssp))
tind = 4
for i in range(nind):
    for j in range(nssp):
        Nind[i, j] = N[cind[i], tind, 2, j]
        Ndataind[i, j] = Ndata[cind[i], tind, 2, j]
        Mind[i, j] = Min[cind[i], tind, 2, j] - Mout[cind[i], tind, 2, j]