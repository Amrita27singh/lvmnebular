#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:33:11 2022
​@author: amrita
"""

import numpy as np
from astropy.io import fits
import pyneb as pn
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys


def new_measurements(value, errs):
    delta = np.random.randn(len(errs)) * errs
    new_values = np.max([value + delta, np.zeros(len(errs))], axis=0)
    return new_values

    

atoms=pn.getAtomDict()
O3=pn.Atom('O',3)
S2=pn.Atom('S',2)
N2=pn.Atom('N',2)
O2=pn.Atom('O',2)
S3=pn.Atom('S',3)

if __name__ == '__main__':

    simname='Bubble_v2_1e-11'
    linefitfile='./'+simname+'/'+simname+'_linefits.fits'

###################spliting text#########################

    split=simname.split('_')
    s=split[0:4]  
    print(s)  

    table=Table.read(linefitfile)
    nfib=len(table)
    print(table.colnames)
    
    ratio=table['4363_flux']/table['5007_flux']
    T=O3.getTemDen(ratio, den=1.e2, wave1=4363, wave2=5007)
    print(T)
    
    ratio_S2=table['6717_flux']/table['6731_flux']
    N=S2.getTemDen(ratio_S2, tem=T, wave1=6717, wave2=6731)
    print(N)
    
    ratio_O3=table['4363_flux']/(table['4959_flux']+table['5007_flux'])
    ratio_N2=table['5755_flux']/(table['6548_flux']+table['6584_flux'])
    ratio_O2=table['3726_flux']/(table['3729_flux'])
    
    diags=pn.Diagnostics()
    TO3, NS2=diags.getCrossTemDen('[OIII] 4363/5007+', '[SII] 6731/6716', ratio_O3, 1/ratio_S2)
    #TN2, NO2=diags.getCrossTemDen('[NII] 5755/6584+', '[OII] 3726/3729', ratio_N2, ratio_O2)
    TN2=N2.getTemDen(ratio_N2, den=1e2, wave1=5755, wave2=6584)
    
    ############################# MC getCrossTemDen ##############################################
    
    niter = 10 # number of iterations
    T_out = np.zeros((nfib,niter))  # output list of T
    N_out = np.zeros((nfib,niter))  # output list of N
    T_out2 = np.zeros((nfib,niter))
    N_out2 = np.zeros((nfib,niter))
    
    
    for i in range(niter):
        print("Doing MC Iteration:", i, niter)

        ###########################  O III ###########################
        tmp_4363 = new_measurements(table['4363_flux'], table['4363_flux_err'])
        tmp_5007 = new_measurements(table['5007_flux'], table['5007_flux_err'])

        ########################### S II ###########################
        tmp_6731 = new_measurements(table['6731_flux'], table['6731_flux_err'])
        tmp_6717 = new_measurements(table['6717_flux'], table['6717_flux_err'])
    
        tmp_S2 = tmp_6731/tmp_6717
        #tmp_T, tmp_N =diags.getCrossTemDen('[OIII] 4363/5007+', '[SII] 6731/6716', tmp_O3, tmp_S2)
    
        tmp_T2 = O3.getTemDen(tmp_4363/tmp_5007, den=NS2, wave1=4363, wave2=5007)
        tmp_N2 = S2.getTemDen(tmp_S2, tem=TO3, wave1=6731, wave2=6717)
        T_out2[:,i]=tmp_T2
        N_out2[:,i]=tmp_N2
    
    
    T2_mean = np.nanmean(T_out2, axis=1)
    T2_std = np.nanstd(T_out2, axis=1)
    
    N2_mean = np.nanmean(N_out2, axis=1)
    N2_std = np.nanstd(N_out2, axis=1)
    
    table['Temp_mean_O3']=T2_mean
    table['Temp_std_O3']=T2_std
    table['Den_mean_S2']=N2_mean
    table['Den_std_S2']=N2_std
    
    
    ################################  O2 & N2 & S3 ################################ 
    T_out_N2 = np.zeros((nfib,niter))
    N_out_O2 = np.zeros((nfib,niter))
    T_out_S3 = np.zeros((nfib,niter))

    for i in range(niter):
        tmp_5755 = new_measurements(table['5755_flux'], table['5755_flux_err'])
        tmp_6584 = new_measurements(table['6584_flux'], table['6584_flux_err'])
        tmp_3726 = new_measurements(table['3726_flux'], table['3726_flux_err'])
        tmp_3729 = new_measurements(table['3729_flux'], table['3729_flux_err'])

    ################################## S III ################################

        tmp_6312 = new_measurements(table['6312_flux'], table['6312_flux_err'])
        tmp_9069 = new_measurements(table['9069_flux'], table['9069_flux_err'])    
        
        tmp_O2=tmp_3726/tmp_3729
        temp_TN2=N2.getTemDen(tmp_5755/tmp_6584, den=1e2, wave1=5755, wave2=6584)
        T_out_N2[:,i]=temp_TN2
        Den_O2=O2.getTemDen(tmp_3726/tmp_3729, tem=TN2, wave1=3726, wave2=3729)
        N_out_O2[:,i]=Den_O2

        temp_TS3=S3.getTemDen(tmp_6312/tmp_9069, den=1e2, wave1=6312, wave2=9069)
        T_out_S3[:,i]=temp_TS3
    
    TS3_mean=np.nanmean(T_out_S3, axis=1)
    TS3_std=np.nanstd(T_out_S3, axis=1)

    TN2_mean=np.nanmean(T_out_N2, axis=1)
    TN2_std=np.nanstd(T_out_N2, axis=1)
    
    NO2_mean=np.nanmean(N_out_O2, axis=1)
    NO2_std=np.nanstd(N_out_O2, axis=1)
    
    table['Temp_mean_N2']=TN2_mean
    table['Temp_std_N2']=TN2_std
    table['Den_mean_O2']=NO2_mean
    table['Den_std_O2']=NO2_std
    table['T[NII]_snr']=TN2_mean/TN2_std

    table['Temp_mean_S3']=TS3_mean
    table['Temp_std_S3']=TS3_std
    table['T[SIII]_snr']=TS3_mean/TS3_std
    
    table.write('newdata_Temp_Den.fits', overwrite=True)
    
    x= table['delta_ra']
    y= table['delta_dec']

    def plotmap(z1, min, max, nlevels=40, title='line_map', output='line_map'):

        sel=np.isfinite(z1)
        
        '''
        newtable=Table.read('Reference.fits')
        fig, ax = plt.subplots(figsize=(8,5))
        triang = tri.Triangulation(x[sel], y[sel]) 
        c = ax.tricontourf(triang, z1[sel], levels=np.linspace(min, max, nlevels))    
        plt.colorbar(c) 
        ax.set_title(title)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.axis('equal')
        plt.savefig(output+'.png')
        '''

        r=np.sqrt(x**2+y**2)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(r[sel], z1[sel], '.')
        ax.set_ylim(min, max)
        ax.set_xlim(0, 260)
        ax.set_ylabel(title)
        ax.set_xlabel('Radius')
        ax.legend()
        #plt.savefig(output+'_rad.png')

       
        
      
    '''
    # Mean NII Temperature
    z=TN2_mean
    plotmap(z, 0, 48000, title=r'T$_{NII}$ (mean)binned', output='./'+simname+'/TN2_mean_1e-16_binned')
    '''
   
    # Mean SIII Temperature
    z1=TS3_mean
    print (z1)
    plotmap(z1, 1000, 9000, title=r'T$_{NII}$ (mean) '+s[2], output='./'+simname+'/TS3_mean_'+s[1]+'_'+s[2])
    plt.show()
    
