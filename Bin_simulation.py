import numpy as np
from astropy.io import fits
from astropy.table import Table
import os.path 
import matplotlib.pyplot as plt
import sys

def bin_spectra(rmin, rmax, radius, spectra, errors):

    selected = np.where((radius >= rmin) & (radius < rmax))[0]

    newflux = np.zeros((len(selected), spectra.shape[1]))
    newerr = np.zeros((len(selected), spectra.shape[1]))

    for i, id in enumerate(selected):
        newflux[i] = spectra[id]
        newerr[i] = errors[id]

    newflux = newflux.sum(axis=0)
    newerr = np.sqrt(np.sum(newerr**2, axis=0))

    return newflux, newerr, len(selected)

if __name__ == '__main__':
    simname='Bubble_v2_e-15'

    name = './'+simname+'/outputs/'+simname+'_linear_full_900_flux.fits'

    with fits.open(name) as hdu:

        wave = hdu['WAVE'].data
        flux = hdu['TARGET'].data
        err = hdu['ERR'].data
        fibers = Table.read(hdu['FIBERID'])
        header = hdu[0].header

    radius = np.sqrt(fibers['x']**2 + fibers['y']**2)

    rbinmax=240
    drbin=20
    bins = np.arange(drbin/2, rbinmax-drbin/2, drbin)
    nspax=np.zeros(len(bins))

    binned_fluxes = np.zeros((len(bins)-1, len(wave)))
    binned_err = np.zeros((len(bins)-1, len(wave)))

    newx = []

    for i in range(len(bins)-1):
        nflux, nerr, nsel = bin_spectra(bins[i]-drbin, bins[i]+drbin, radius, flux, err)
        nspax[i]=nsel
        binned_fluxes[i] = nflux
        binned_err[i] = nerr
        newx.append(bins[i])

    hdu_primary = fits.PrimaryHDU(header=header)
    hdu_target = fits.ImageHDU(data=binned_fluxes, name='TARGET')
    hdu_errors = fits.ImageHDU(data=binned_err, name='ERR')
    hdu_wave = fits.ImageHDU(data=wave, name='WAVE')

    newtable = {'id': range(0, len(bins)-1),
                'y': np.zeros(len(bins)-1),
                'x': newx}

    newtable = Table(newtable)
    hdu_table = fits.BinTableHDU(newtable, name='FIBERID')

    hdul = fits.HDUList([hdu_primary, hdu_target, hdu_errors, hdu_wave, hdu_table])

    #hdul.writeto('binned_900_flux.fits', overwrite=True)

    filename =f'{simname}_binned_900_flux.fits'
    directory=f'./{simname}_binned/'
    if ( not os.path.isdir(directory)):
        os.mkdir(directory)
    plotdir=directory+'/linefitplots/'
    if ( not os.path.isdir(plotdir)):
       os.mkdir(plotdir)

    hdul.writeto(directory+filename, overwrite=True)
    plt.plot(bins, nspax)
    plt.show()