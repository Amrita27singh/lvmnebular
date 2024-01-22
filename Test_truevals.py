#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:36:41 2023

@author: amrita
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
import os

#Radius’, 'Te', 'ne', 'H+', 'O0', 'O+', 'O++', 'N0', 'N+', 'N++', 'S0', 'S+', 'S++

simname = 'Bubble_v2_5e-14_v1'

plt.rcParams.update({'axes.titlesize': 'x-large',
                 'axes.labelsize':'X-large',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'X-large',
                 'xtick.labelsize': 'X-large',
                 'font.size': '12.0',
                 'legend.fontsize':'small'}) 
        

hdu=fits.open('/home/amrita/LVM/lvmnebular/'+simname+'/testneb_tutorial3_ex1.fits')
vals=hdu['Comp_0_PhysParams'].data
e_5007 = hdu['COMP_0_FLUX_5006.84'].data
print(np.mean(e_5007))

#print("Radius:",vals[0], vals[0].shape)
#print("True Te:",vals[1],vals[1].shape)
#print("True Ne:",vals[2],vals[2].shape)

datadir = '/home/amrita/LVM/lvmnebular/'+simname+ '/'
plotdir=datadir+'plot_true_ne_profile/'
if not (os.path.isdir(plotdir)):
    os.mkdir(plotdir)

### Te vs R(pc)
fig, ax3=plt.subplots(figsize=(8,8))
ax3.plot(vals[0], vals[1], label='True Te')
ax3.set_xlabel('Radius(pc)')
ax3.set_ylabel('Gas Te (K)')
ax3.set_title('True Te structure')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/Te vs R.png', dpi=300, bbox_inches = 'tight')
plt.show()


### ne vs R(pc) 
fig, ax4=plt.subplots(figsize=(8,8))
ax4.plot(vals[0], vals[2], label='True ne')
ax4.set_xlabel('Radius(pc)')
ax4.set_ylabel('Gas ne (cm^-3)')
ax4.set_title('True ne radial variation')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/ne vs R.png', dpi=300, bbox_inches = 'tight')
plt.show()


### Te vs ne
fig, ax5=plt.subplots(figsize=(8,8))
ax5.plot(vals[2], vals[1], label='Te_vs_ne')
ax5.set_xlabel('ne (cm^-3)')
ax5.set_ylabel('Te (K)')
ax5.set_title('True Te vs ne variation')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/Te vs ne.png', dpi=300, bbox_inches = 'tight')
plt.show()


### Te(O) vs R(pc)
fig, ax=plt.subplots(figsize=(8,8))
ax.plot(vals[0], vals[3], label='H+', color='black')
ax.plot(vals[0], vals[4], label='O0', color='black', linestyle='--')
ax.plot(vals[0], vals[5], linestyle='-', label='O+', color='red')
ax.plot(vals[0], vals[6], linestyle=':', label='O++')
ax.plot(vals[0], vals[4]+vals[5]+vals[6], label='O')
ax.legend()

ax.set_title('Ionic abundances radial variation of Oxygen ions')
ax.set_xlabel('Radius(pc)')
ax.set_ylabel('Ionic abundance')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/nO vs R.png', dpi=300, bbox_inches = 'tight')
plt.show()


### Te(N) vs R(pc)
fig, ax1=plt.subplots(figsize=(8,8))
ax1.plot(vals[0], vals[3], label='H+', color='black')
ax1.plot(vals[0], vals[7], label='N0', color='black', linestyle='--')
ax1.plot(vals[0], vals[8], linestyle='-', label='N+', color='red')
ax1.plot(vals[0], vals[9], linestyle=':', label='N++')
ax1.plot(vals[0], vals[7]+vals[8]+vals[9], label='N')
ax1.legend()

ax1.set_title('Ionic abundances radial variation of Nitrogen ions')
ax1.set_xlabel('Radius(pc)')
ax1.set_ylabel('Ionic abundance')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/nN vs R.png', dpi=300, bbox_inches = 'tight')
plt.show()

### Te(S) vs R(pc)
fig, ax2=plt.subplots(figsize=(8,8))
ax2.plot(vals[0], vals[3], label='H+', color='black')
ax2.plot(vals[0], vals[10], label='S0', color='black', linestyle='--')
ax2.plot(vals[0], vals[11], linestyle='-', label='S+', color='red')
ax2.plot(vals[0], vals[12], linestyle=':', label='S++')
ax2.plot(vals[0], vals[10]+vals[11]+vals[12], label='S')
ax2.legend()
ax2.set_title('Ionic abundances radial variation of Sulphur ions')
ax2.set_xlabel('Radius(pc)')
ax2.set_ylabel('Ionic abundance')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/nS vs R.png', dpi=300, bbox_inches = 'tight')
plt.show()


fig, ax6=plt.subplots(figsize=(8,8))
ax6.plot(vals[0], vals[6], '.', color='red'   , label = 'O++')
ax6.plot(vals[0], vals[5], '.', color='orange', label = 'O+')
ax6.plot(vals[0], vals[8], '.', color='navy'  , label = 'N+')
ax6.plot(vals[0], vals[12], '.', color='green', label = 'S++')
ax6.plot(vals[0], vals[11], '.', color='cyan', label = 'S+')

ax6.set_xlabel('Radius (pc)')
ax6.set_ylabel('Relative ionic abundance')
ax6.legend()
ax6.set_title('Ionization structure radial variation')
plt.savefig('/home/amrita/LVM/lvmnebular/'+simname+'/plot_true_ne_profile/relative ionic ne vs R.png', dpi=300, bbox_inches = 'tight')
plt.show()

