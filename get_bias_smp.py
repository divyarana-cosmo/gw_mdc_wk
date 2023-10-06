import pandas as pd
from scipy.spatial import KDTree
import numpy as np

def bias_smp(bb, lbox = 2500.0):
    '''get a million ready'''
    ncts = 1000000
    data = pd.read_csv('2020-08-14-11-24-59-2138.csv')
    rws  = np.arange(len(data['x'].values))
    np.random.seed(123)
    ipik = np.random.choice(rws, size=int(bb*ncts))
    xx   = data['x'].values[ipik]%lbox
    yy   = data['y'].values[ipik]%lbox
    zz   = data['z'].values[ipik]%lbox
    flg  = 0.0*xx
    print(xx.max(), yy.max(), zz.max())
    ptree = KDTree(np.transpose([xx,yy,zz]), boxsize=lbox)

    unipos = np.random.uniform(size=3*int(abs(bb - 1)*ncts))*lbox
    unipos = unipos.reshape((-1,3))
    if bb>=1:
        dis, ids =  ptree.query(unipos, k=1)
        flg[ids] = 1.0
        idx = (flg!=1.0)
        xx = xx[idx]; yy=yy[idx]; zz=zz[idx];
    else:
        xx = np.append(xx,unipos[:,0])
        yy = np.append(yy,unipos[:,1])
        zz = np.append(zz,unipos[:,2])

    return np.transpose([xx, yy, zz])


def get_autocorr(data,rbin, lbox = 2500.0):
    ptree = KDTree(data, boxsize=lbox)
    pairs = ptree.count_neighbors(ptree, rbin, cumulative=True)
    dd = pairs[1:] - pairs[:-1]

    volfrac = rbin[1:]**3 - rbin[:-1]**3
    rr = len(data[:,0])**2 * 4*np.pi/3.0 *volfrac * 1.0/lbox**3
    ans = dd*1.0/rr - 1

    return ans


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.subplot(3,3,1)
rbin = np.logspace(1,2,21)
sq_bias = 0
for bb in [1.0,0.5]:
    try:
        rbins,corr = np.loadtxt('./bias_%s.dat'%bb, unpack=1)
        print('file exists')
    except:
        data = bias_smp(bb)
        corr = get_autocorr(data,rbin)
        rbins = (rbin[1:] + rbin[:-1])/2.0
        np.savetxt('./bias_%s.dat'%bb, np.transpose([rbins, corr]))

    #if bb==1.0:
    #    corr = corr*0.5**2
    idx = rbins<70
    #corr = corr*1.0/bb**2
    plt.plot(rbins[idx], corr[idx], label = 'b=%s'%bb)
    if bb==1.0:
        corr = corr*0.5**2
        plt.plot(rbins[idx], corr[idx], ls='--', color='C0')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r[{\rm h^{-1}Mpc}]$')
plt.ylabel(r'$\xi(r)$')


plt.legend()
plt.savefig('test.png', dpi=300)



    
    

