#!/usr/bin/env python


# Toumazou and Cretaux (2000):
#
# A code that performs EOF analysis should be composed of 3 main steps
#
# Step 1: Preprocessing the data
#  - Reading of the field of m points at n timesteps
#  - Computation of the time average and centering of the data
#  - If necessary, subtraction of the long periodic terms
#  - Representation of the data as a matrix DcR(mxn)

# Step 2: Decomposition of the matrix of data D
#  - Resolution  of the linear algebra problem: compu-
# tation of U(k),S(k),V(T,k) [T : Transpose]

# Step 3: Postprocessing of the data
#  - Computation of the components m(j) and e(j) for each mode for
# j=1,...,k from the solution of the linear algebra problem
#

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import random
from scipy import interpolate


def pcomp(pcomponents, s):
    for t in range(1,N-M+2): #from soi_norm[0] to soi_norm[N-1]
        for j in range(0,M):
            s += soi_norm[t+j-1]*eigenVec[j]
        pcomponents.append(s)
        s = 0
    return pcomponents 

## Variables ##

# Data matrix
M = 15
N = 10
VAL_MIN = 1
VAL_MAX = 10
X = np.random.randint(VAL_MIN, VAL_MAX, size=(N,M))
TOL = None

## SVD ##

# X shape is NxM, U is NxR, S is RxM, S is diagonal RxR 
# U, s, V = np.linalg.svd(X, full_matrices=False)
# V_LINE, V_COL = V.shape[0], V.shape[1]
# U_LINE, U_COL = U.shape[0], U.shape[1]
# s_SHAPE = s.shape[0] # 1D array

# R = np.linalg.matrix_rank(X,tol=TOL)
# #lambdas, Vs = np.linalg.eig(X.T)
# S = np.diag(s)


## Plot Vs, Us and singular value in decreasing order ##
# plt.figure(1)
# plt.plot([i for i in range(V_LINE*V_COL)], np.reshape(V,V_LINE*V_COL),'b-')
# plt.xlabel('rank')
# plt.ylabel('EOFs')

# plt.figure(2)
# plt.plot([i for i in range(U_LINE*U_COL)], np.reshape(U,U_LINE*U_COL), 'r-')
# plt.xlabel('rank')
# plt.ylabel('PCs')

# plt.figure(3)
# plt.plot([i for i in range(s_SHAPE)], np.reshape(s,s_SHAPE), 'ko')
# plt.xlabel('rank')
# plt.ylabel('Singular value')


## SOI example (Ghil et al. 2001) ##

#
# extract data and put it in form
#
N = 690
YEAR_START = 9 # year 1: 1942 ... year 58: 1999
YEAR_END = 39
MONTH_PER_YEAR =12
data_list = []
float_list = []
soi = open('SOI_DATA.txt','r')
data = soi.readlines()
for line in data:
    words=line.split()
    data_list.append(words[1:13]) #n0 is date, 14 is average, don't need it
    
for i in range(len(data)):
    for j in range(len(data_list[0][0:12])):
        float_list.append(float(data_list[i][j]))

#
# eigenvalues & EOFs calculations
#
M = 60 # lag window length
C = np.ndarray((M,M)) # init covariance matrix with zeros

# time series normalization 
soi_std = np.std(float_list[MONTH_PER_YEAR*YEAR_START:MONTH_PER_YEAR*(YEAR_END+1)])
soi_norm = (float_list[0:N]-np.mean(float_list))/soi_std

# lag covariance matrix
summ = 0
for i in range(0,M):
    for j in range(0,M):
        for time in range(0,N-abs(i-j)):
            summ += soi_norm[time]*soi_norm[time+abs(i-j)]
        C[i][j] = (1./(N-abs(i-j)))*float(summ)
        summ = 0

# compute eigenvalues, eigenvectors using Singular Value Decomposition
eigenVec, eigenVal, eigenVec_T = np.linalg.svd(C, full_matrices=False)
d_SHAPE = eigenVal.shape[0] # 1D array

#
# principal component (PCs) calculation
#
interp_factor = 10
smooth_factor = 0.2
somme = 0
PComponents = []
for t in range(1,N-M+2): #from soi_norm[0] to soi_norm[N-1]
    for j in range(0,M):
        somme += soi_norm[t+j-1]*eigenVec[j]
    PComponents.append(somme)
    somme = 0

# get two PCs
NUM1 = 0
NUM2 = 1
PC1 = []
PC2 = []
for i in range(N-M):
    PC1.append(PComponents[i][NUM1])
    PC2.append(PComponents[i][NUM2])

# # first PC
# # extrapolate
# x = np.arange(len(PC1))
# tck = interpolate.splrep(x, PC1, s=0)
# x_new = np.arange(0, len(PC1), interp_factor) 
# PC1_interp = interpolate.splev(x_new, tck, der=0)

# # interpolate the extrapolated function to get a smooth curve
# x_fine = np.arange(0, len(PC1), smooth_factor)
# first_interp = interpolate.splrep(x_new, PC1_interp, s=0)
# PC1_smoothed = interpolate.splev(x_fine, first_interp, der=0)

# # second PC
# tck2 = interpolate.splrep(x, PC2, s=0)
# PC2_interp = interpolate.splev(x_new, tck2, der=0)

# first_interp2 = interpolate.splrep(x_new, PC2_interp, s=0)
# PC2_smoothed = interpolate.splev(x_fine, first_interp2, der=0)

#
# reconstructed components (RCs) calculation
#
RCsum = 0
Rcomponent = []
kset = 10
for k in range(0, kset):
    for t in range(0, N):
        if (t <= M-2):
            for j in range(0, t):
                RCsum += (1./t)*PComponents[t-j][k]*eigenVec[j]
        if (t >= M-1 and t <= N-M):
            for j in range(0, M):
                RCsum += (1./M)*PComponents[t-j][k]*eigenVec[j]
        if (t >= N-M+1 and t <= N-1):
            for j in range(t-N+M, M):
                RCsum += (1./(N-t+1))*PComponents[t-j][k]*eigenVec[j]
        Rcomponent.append(RCsum)
        RCsum = 0

RC1 = []
RC2 = []
RC3 = []
RC4 = []
RC5 = []
RC6 = []
RC7 = []
RC8 = []
Rall = []
RC = []
RCC = []
RCCC = []
Rsum = 0
for i in range(1, N):
    RC1.append(Rcomponent[i][0])
    RC2.append(Rcomponent[i][1])
    RC3.append(Rcomponent[i][2])
    RC4.append(Rcomponent[i][3])
    RC5.append(Rcomponent[i][4])
    RC6.append(Rcomponent[i][5])
    RC7.append(Rcomponent[i][6])
    RC8.append(Rcomponent[i][7])
    for j in range(0, M):
        Rsum += Rcomponent[i][j]
    Rall.append(Rsum)
    Rsum = 0
        
for j in range(0, N-1):
    RC.append(RC1[j])
    RCC.append(RC1[j] + RC2[j])
    RCCC.append(RC1[j] + RC2[j] + RC3[j] + RC4[j])
    #RCCC.append(RC1[j] + RC2[j] + RC3[j] + RC4[j] + RC5[j] + RC6[j] + RC7[j] + RC8[j])

#
# Graphics
#
plt.figure(1)
plt.semilogy([i for i in range(d_SHAPE)], eigenVal, 'ko')
plt.xlabel('rank')
plt.ylabel('eigenvalues')

#plt.plot([i for i in range()], U

plt.figure(2)
#plt.plot([i for i in range(N)], float_list[0:N], 'k-',linewidth=0.8)
plt.plot([i for i in range(N)], soi_norm, 'k-',linewidth=0.3)
plt.plot([i for i in range(1,N)], RC, 'k-',linewidth=1.2)
plt.plot([i for i in range(1,N)], RCC, 'b-',linewidth=1.2)
plt.plot([i for i in range(1,N)], RCCC, 'r-',linewidth=1.2)
plt.plot([i for i in range(1,N)], Rall, 'g-',linewidth=1.2)
plt.xlabel('Time (months)')
plt.ylabel('SOI')

plt.figure(3)
EOF1 = 0 # choose which EOF pair to display
EOF2 = 1
plt.plot(range(M), eigenVec_T[EOF1], 'k-', linewidth=2)
plt.plot(range(M), eigenVec_T[EOF2], 'k-', linewidth=1)
plt.xlabel('rank')
plt.ylabel('EOF %d-%d' %(EOF1+1, EOF2+1))

plt.figure(4)
plt.plot([i for i in range(len(PC1))], PC1, 'g-',linewidth=1.2)
plt.plot([i for i in range(len(PC1))], PC2, 'r-',linewidth=1.2)
plt.ylabel('PC %d-%d' %(NUM1+1, NUM2+1))
plt.ylim(-10, 10)


# plt.figure(5)
# plt.plot([i for i in range(N)], PC1, 'g-',linewidth=1.2)
# plt.plot([i for i in range(N)], PC2, 'r-',linewidth=1.2)
# plt.ylabel('PC %d-%d' %(NUM1+1, NUM2+1))
# plt.ylim(-10, 10)


# plt.figure(5)
# #plt.plot(x_new, PC1_interp, 'k-',linewidth=0.8)
# plt.plot(x_fine, PC1_smoothed, 'g-',linewidth=1.2)
# plt.plot(x_fine, PC2_smoothed, 'r-',linewidth=1.2)
# plt.xlabel('Time (months)')
# plt.ylabel('PCs')
# plt.ylim(-10, 10)
# plt.xlim(-0.5, 631)

plt.show()


