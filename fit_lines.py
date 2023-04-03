#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:06:43 2022

@author: amrita
"""

from statistics import mean
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.io import fits 
from astropy.table import Table, QTable
import matplotlib.tri as tri
import sys
import os.path


def gaussian(wave,flux,mean,sd):
    return flux/(np.sqrt(2*np.pi)*sd)*np.exp((wave-mean)**2/(-2*sd**2))


def error_func(wave, gaussian, popt, pcov, e=1e-7):

    f0=gaussian(wave, popt[0], popt[1], popt[2])
    
    dfdx=(gaussian(wave, popt[0]*(1+e), popt[1], popt[2])-f0)/(popt[0]*e)
    dfdy=(gaussian(wave, popt[0], popt[1]*(1+e), popt[2])-f0)/(popt[1]*e)
    dfdz=(gaussian(wave, popt[0], popt[1],popt[2]*(1+e))-f0)/(popt[2]*e)
    
    df=(dfdx, dfdy, dfdz)
    

    dx=df[0]**2*pcov[0][0]
    dy=df[1]**2*pcov[1][1]
    dz=df[2]**2*pcov[2][2]
    dxdy=df[0]*df[1]*pcov[1][0]
    dxdz=df[0]*df[2]*pcov[2][0]
    dydz=df[1]*df[2]*pcov[2][1]
    
    var=dx+ dy + dz + 2*(dxdy + dxdz + dydz)

    sigma=np.sqrt(var)
    return sigma

def fit_gauss(wave, spectrum, error,  lwave, dwave=4, plot=False, plotout='linefit'):
 
    sel=(wave>lwave-dwave/2)*(wave<lwave+dwave/2)

    p0=(np.abs(spectrum[sel].max()), lwave, 0.7)

    print("Initial Guess:", p0)

    try: 
        popt, pcov=curve_fit(gaussian, wave[sel], spectrum[sel], sigma=error[sel], p0=p0, absolute_sigma=True, bounds=((0,lwave-1.0,0.3),(np.inf,lwave+1.0,2.0)))
    except RuntimeError: 
        popt=np.array([-99, p0[1], p0[2]])
        pcov=np.zeros((3,3))
        pcov[0,0]=np.sum(error[sel]**2)
    #   print("Best Fit Parameters:", popt)
        
    #if np.sqrt(pcov[0,0])/popt[0]>4:
     #   popt[0]=np.sum(spectrum[sel])
      #  pcov[0,0]=np.sum(error[sel]**2)

    #print(popt)
    #print(pcov)

    if plot==True:
        xm=np.arange(wave[sel][0], wave[sel][-1], 0.01)

        sigma = error_func(xm, gaussian, popt, pcov)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(wave, spectrum, error, c='k', label='data')
        ax.scatter(wave[sel], spectrum[sel], c='b', label='data')
        ax.plot(wave[sel], spectrum[sel], c='b', label='masked data')

        ax.set_xlim(lwave-dwave, lwave+dwave)
        
        ym = gaussian(xm, popt[0], popt[1], popt[2])
        ax.plot(xm, ym, c='r', label='model')
        ax.plot(xm, gaussian(xm, p0[0], p0[1], p0[2]), c='g', label='fit')
        ax.fill_between(xm, ym+3*sigma, np.max([ym-ym,ym-3*sigma], axis=0), alpha=0.7, 
                    linewidth=3, label='error limits')
        
        ax.set_ylim(-0.4*spectrum[sel].max(),1.7*np.max([spectrum[sel].max(), (ym+3*sigma).max()]))

        
        ax.legend()
        plt.savefig(plotout+'.png')
        #plt.show()

    return popt, pcov
    

if __name__ == '__main__':
     
    simname='Bubble_v2_5e-14'
    if ( not os.path.isdir(simname)):
        os.mkdir(simname)

    filename ='./'+simname+'/outputs/'+simname+'_linear_full_900_flux.fits'
    outfilename='./'+simname+'/'+simname+'_linefits.fits'
    plotdir='./'+simname+'/linefitplots/'
    if ( not os.path.isdir(plotdir)):
        os.mkdir(plotdir)


    with fits.open(filename) as hdu:
        header = hdu[0].header
        print(repr(header))

        wave = hdu['WAVE'].data
        fiberid = Table.read(hdu['FIBERID'])
        print(fiberid)
        flux = hdu['TARGET'].data
        err = hdu['ERR'].data
        
    sys_vel=20
    c=299792.458 
    lines0= np.array([9069, 6731, 6717, 6584, 6563, 6548, 6312, 5755, 5007, 4959, 4861, 4363, 4069, 3970, 3729, 3726])
    lines=lines0*(1+sys_vel /c)

    output_dict= {'fiber_id': [], 
                'delta_ra':[], 
                'delta_dec':[], 
                '9069_flux':[], '9069_flux_err':[], '9069_lambda':[], '9069_lambda_err':[], '9069_sigma':[], '9069_sigma_err':[],
                '6731_flux':[], '6731_flux_err':[], '6731_lambda':[], '6731_lambda_err':[], '6731_sigma':[], '6731_sigma_err':[],
                '6717_flux':[], '6717_flux_err':[], '6717_lambda':[], '6717_lambda_err':[], '6717_sigma':[], '6717_sigma_err':[],
                '6584_flux':[], '6584_flux_err':[], '6584_lambda':[], '6584_lambda_err':[], '6584_sigma':[], '6584_sigma_err':[],
                '6563_flux':[], '6563_flux_err':[], '6563_lambda':[], '6563_lambda_err':[], '6563_sigma':[], '6563_sigma_err':[],
                '6548_flux':[], '6548_flux_err':[], '6548_lambda':[], '6548_lambda_err':[], '6548_sigma':[], '6548_sigma_err':[],
                '6312_flux':[], '6312_flux_err':[], '6312_lambda':[], '6312_lambda_err':[], '6312_sigma':[], '6312_sigma_err':[],
                '5755_flux':[], '5755_flux_err':[], '5755_lambda':[], '5755_lambda_err':[], '5755_sigma':[], '5755_sigma_err':[],
                '5007_flux':[], '5007_flux_err':[], '5007_lambda':[], '5007_lambda_err':[], '5007_sigma':[], '5007_sigma_err':[],
                '4959_flux':[], '4959_flux_err':[], '4959_lambda':[], '4959_lambda_err':[], '4959_sigma':[], '4959_sigma_err':[],
                '4861_flux':[], '4861_flux_err':[], '4861_lambda':[], '4861_lambda_err':[], '4861_sigma':[], '4861_sigma_err':[],
                '4363_flux':[], '4363_flux_err':[], '4363_lambda':[], '4363_lambda_err':[], '4363_sigma':[], '4363_sigma_err':[],
                '4069_flux':[], '4069_flux_err':[], '4069_lambda':[], '4069_lambda_err':[], '4069_sigma':[], '4069_sigma_err':[],
                '3970_flux':[], '3970_flux_err':[], '3970_lambda':[], '3970_lambda_err':[], '3970_sigma':[], '3970_sigma_err':[],
                '3729_flux':[], '3729_flux_err':[], '3729_lambda':[], '3729_lambda_err':[], '3729_sigma':[], '3729_sigma_err':[],
                '3726_flux':[], '3726_flux_err':[], '3726_lambda':[], '3726_lambda_err':[], '3726_sigma':[], '3726_sigma_err':[]
                }  

    for i in range(len(fiberid)):
    #for i in [105, 106, 107]:
        mask = fiberid['id'] == i
        spectrum = flux[mask][0]  # select only fiber n. 200
        error = err[mask][0]
        output_dict['fiber_id'].append(fiberid[mask]['id'])
        output_dict['delta_ra'].append(fiberid[mask]['x'][0])
        output_dict['delta_dec'].append(fiberid[mask]['y'][0])       

        for j,line in enumerate(lines):
            print("Fitting Line:", line)
            plot=False
            plotout='junk'
            list=[6312, 9069]
            #list=[5755, 6584, 6563, 6312, 9069]
            if lines0[j] in list:
                plot=True               
                plotout=plotdir+str(fiberid['id'][i])+'_'+str(lines0[j])
            popt, pcov = fit_gauss(wave, spectrum, error, line, plot=plot, plotout=plotout)
            #popt, pcov = fit_gauss(wave, spectrum, error, line, plot=plot)
           
               
            output_dict[str(lines0[j])+'_flux'].append(popt[0])
            output_dict[str(lines0[j])+'_flux_err'].append(np.sqrt(pcov[0, 0]))
            output_dict[str(lines0[j])+'_lambda'].append(popt[1])
            output_dict[str(lines0[j])+'_lambda_err'].append(np.sqrt(pcov[1, 1]))
            output_dict[str(lines0[j])+'_sigma'].append(popt[2])
            output_dict[str(lines0[j])+'_sigma_err'].append(np.sqrt(pcov[2, 2]))
            
            print("fiber: ", i)
            print(popt)
            print(np.sqrt(pcov[0, 0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2, 2]), "\n")
            
            #for key, value in output_dict.items():
               # print(key, ' \n ', value)
      

    t=Table(output_dict, names=output_dict.keys())
    t.write(outfilename, overwrite=True)    




