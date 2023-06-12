#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:44:08 2023

@author: amrita
"""

from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

'''
LVM_cloudy_models_phys.fits file contains all approx 3500 cloudy models, this code extracts Te,
ne, ionic abundances, lines and their respective emissivities variation radially.
'''

hdulist = fits.open('/home/amrita/lvmdatasimulator/data/LVM_cloudy_models_phys.fits')

#print("List of extensions:")
#hdulist.info()

# Comp_0_PhysParams gives a subset of values from the above file: 13 quantities.
hdu=fits.open('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14/testneb_tutorial3_ex1.fits')
vals=hdu['Comp_0_PhysParams'].data

#print(vals, vals.shape)

extension_number = 258
selected_extension = hdulist[extension_number]

data = selected_extension.data
header = selected_extension.header
#print(type(data))

#table=Table(rows=data)
#table.write('shell_108.fits', format='fits', overwrite=True)

#print(data[0, 2:], data[0, 2:]*vals[0])
print(data[66,0], data[93,0], data[73,0], data[71,0], vals[8]*6.76e-5*0.013*1e2, vals[8].shape)
plt.plot(vals[0], data[93, 2:], label='H_alpha_vs_R')
plt.plot(vals[0], data[66, 2:], label='H_beta_vs_R')
plt.plot(vals[0], data[73, 2:], label='[OIII_5007]_vs_R')
plt.plot(vals[0], data[71, 2:], label='[OIII_4959]_vs_R')
plt.xlabel('R(pc)')
plt.ylabel('Emissivity')
plt.legend()

fig, ax=plt.subplots()
ax.plot(vals[0], vals[8]*6.67e-5*1e2, label='n[NII]_vs_R')
#ax.plot(vals[0], vals[8], label='ionic_abundance_[NII]')
ax.legend()
plt.show()

#print("Data values:", data, data.shape)
#print("Header:", header)

hdulist.close()
