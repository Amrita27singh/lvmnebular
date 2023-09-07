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

#Radius’, 'Te', 'ne', 'H+', 'O0', 'O+', 'O++', 'N0', 'N+', 'N++', 'S0', 'S+', 'S++


hdu=fits.open('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/testneb_tutorial3_ex1.fits')
vals1=hdu['Comp_0_PhysParams'].data

print("Radius:",vals1[0], vals1[0].shape)
print("True Te:",vals1[1],vals1[1].shape)
print("True Ne:",vals1[2],vals1[2].shape)

### Te vs R(pc)
fig, ax3=plt.subplots()
ax3.plot(vals1[0], vals1[1], label='True Te')
ax3.set_xlabel('Radius(pc)')
ax3.set_ylabel('Gas Te (K)')
ax3.set_title('True Te radial variation')
plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/plotprofile/Te vs R.png', dpi=300)
plt.show()


### ne vs R(pc) 
fig, ax4=plt.subplots()
ax4.plot(vals1[0], vals1[2], label='True ne')
ax4.set_xlabel('Radius(pc)')
ax4.set_ylabel('Gas ne (cm^-3)')
ax4.set_title('True ne radial variation')
plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/plotprofile/ne vs R.png', dpi=300)
plt.show()


### Te vs ne
fig, ax5=plt.subplots()
ax5.plot(vals1[2], vals1[1], label='Te_vs_ne')
ax5.set_xlabel('ne (cm^-3)')
ax5.set_ylabel('Te (K)')
ax5.set_title('True Te vs ne variation')
plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/plotprofile/Te vs ne.png', dpi=300)
plt.show()


### Te(O) vs R(pc)
fig, ax=plt.subplots()
ax.plot(vals1[0], vals1[3], label='H+', color='black')
ax.plot(vals1[0], vals1[4], label='O0', color='black', linestyle='--')
ax.plot(vals1[0], vals1[5], linestyle='-', label='O+', color='red')
ax.plot(vals1[0], vals1[6], linestyle=':', label='O++')
ax.plot(vals1[0], vals1[4]+vals1[5]+vals1[6], label='O')
ax.legend()

ax.set_title('Ionic abundances radial variation of Oxygen ions')
ax.set_xlabel('Radius(pc)')
ax.set_ylabel('Ionic abundance')
plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/plotprofile/nO vs R.png', dpi=300)
plt.show()


### Te(N) vs R(pc)
fig, ax1=plt.subplots()
ax1.plot(vals1[0], vals1[3], label='H+', color='black')
ax1.plot(vals1[0], vals1[7], label='N0', color='black', linestyle='--')
ax1.plot(vals1[0], vals1[8], linestyle='-', label='N+', color='red')
ax1.plot(vals1[0], vals1[9], linestyle=':', label='N++')
ax1.plot(vals1[0], vals1[7]+vals1[8]+vals1[9], label='N')
ax1.legend()

ax1.set_title('Ionic abundances radial variation of Nitrogen ions')
ax1.set_xlabel('Radius(pc)')
ax1.set_ylabel('Ionic abundance')
plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/plotprofile/nN vs R.png', dpi=300)
plt.show()

### Te(S) vs R(pc)
fig, ax2=plt.subplots()
ax2.plot(vals1[0], vals1[3], label='H+', color='black')
ax2.plot(vals1[0], vals1[10], label='S0', color='black', linestyle='--')
ax2.plot(vals1[0], vals1[11], linestyle='-', label='S+', color='red')
ax2.plot(vals1[0], vals1[12], linestyle=':', label='S++')
ax2.plot(vals1[0], vals1[10]+vals1[11]+vals1[12], label='S')
ax2.legend()

ax2.set_title('Ionic abundances radial variation of Sulphur ions')
ax2.set_xlabel('Radius(pc)')
ax2.set_ylabel('Ionic abundance')
plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14_z_0.5/plotprofile/nS vs R.png', dpi=300)
plt.show()
