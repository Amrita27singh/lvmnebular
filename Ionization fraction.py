#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:01:51 2023

@author: amrita
"""

'''
Ionization fraction of O, N and S 

n(sI)=n(sI)/(n(s0)+n(SI)+n(SII))

'''
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


#Radius’, 'Te', 'ne', 'H+', 'O0', 'O+', 'O++', 'N0', 'N+', 'N++', 'S0', 'S+', 'S++
def Ionization(self):


    hdu=fits.open('True_vals.fits')
    data=hdu[1].data
    #print(data)
    
    nS0=data[10]/data[3]
    nSI=data[11]/data[3]
    nSII=data[12]/data[3]
    
    nO0=data[4]/data[3]
    nOI=data[5]/data[3]
    nOII=data[6]/data[3]
    
    nN0=data[7]/data[3]
    nNI=data[8]/data[3]
    nNII=data[9]/data[3]
    
    fig, ax=plt.subplots()
    
    ax.plot(data[0], nS0, label='nS0', color='black', linestyle='--')
    ax.plot(data[0], nSI, linestyle='-', label='nSI', color='red')
    ax.plot(data[0], nSII, linestyle=':', label='nSII')
    ax.legend()
    ax.set_title('Ionization fraction of Sulphur ions')
    ax.set_xlabel('Radius(pc)')
    ax.set_ylabel('Ionization fraction')
    plt.savefig(self.datadir+self.simname+'/'+self.simname+'_plotprofile/'+'Ionization fraction of Sulphur vs R.png', dpi=300)
    plt.show()
