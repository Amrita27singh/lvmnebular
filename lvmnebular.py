import numpy as np
import matplotlib.pyplot as plt
import os.path
from astropy.table import Table, QTable
import astropy.io.fits as fits
import pyneb as pn
from scipy.optimize import curve_fit
import itertools
from scipy.fft import fftn, ifftn
import imageio
import matplotlib.tri as tri


class simulation:
    '''
    Object simulation contains all the relevant/needed output from an LVM simulator run
    '''
    def __init__(self):
        '''
        Input:
        simname: id of lvmsimulator output simulation (str)
        exptime: exposure time of simulation (float)
        
        Attributes: 

        datadir: directory containing output from lvmsimulator (str)
        simname: id of lvmsimulator output simulation (str)
        exptime: exposure time of simulation (float)
        simfile: filename of simulation output
        wave: wavelength array of spectra in simulations (1D numpy array)
        flux: flux array spectum for each spaxel (2D numpy array (Nspax, Nwave))
        err: flux error array spectum for each spaxel (2D numpy array (Nspax, Nwave))
        fiberdata: spaxel information table (Table)

        ra : RA of each spaxel (1D numpy array)
        dec: DEC of each spaxel (1D numpy array)
        lineid: list of line IDs (list)
        lineflux: line fluxes of each line in each spaxel (2D numpy array)

        '''

        self.datadir=None
        self.simname=None
        self.exptime=None
        self.simfile=None
        self.wave =None
        self.flux = None
        self.err = None
        self.fiberdata = None
        self.nfib=None
        self.nfibbin=None
        self.linefitdict=None

        self.ra=None
        self.dec=None
        self.lineid=None
        

        self.simfile = None
        self.linefitfile= None
        self.dim=None

        self.plot=None

        self.TeO2=None
        self.TeO2err=None
        self.TeO3=None
        self.TeO3err=None
        self.TeN2=None
        self.TeN2err=None
        self.TeS2=None
        self.TeS2err=None
        self.TeS3=None
        self.TeS3err=None
        self.neO2=None
        self.neO2err=None
        self.neS2=None
        self.neS2err=None


    def loadsim(self, simname, exptime, datadir='/home/amrita/LVM/lvmnebular/'):

        self.datadir=datadir
        self.simname=simname
        self.exptime=exptime
        self.simfile = self.datadir+self.simname+'/outputs/'+self.simname+'_linear_full_'+str(int(self.exptime))+'_flux.fits'        
        #read simulator output in the right path correpsonding to "name"
        print("Loading simulation: "+self.datadir+self.simname+'\n')

        if ( not os.path.isdir(self.datadir+self.simname)):
            raise Exception("Simulation "+self.simname+" does not exist. Run run_simulation.py first")
            
        with fits.open(self.simfile) as hdu:
            self.header = hdu[0].header
            self.wave = hdu['WAVE'].data ##1D array
            self.flux = hdu['TARGET'].data ##2D array(no. of fibers, wave)
            self.err = hdu['ERR'].data
            self.fiberdata = Table.read(hdu['FIBERID'])
            #print(self.fiberdata)

        self.nfib=len(self.fiberdata)

    
    def fitlines(self, sys_vel=0, lines0= np.array([6563, 6583]) , bin=False, pertsim=False, loadfile=True, plot=False):  #try plotting for a subset of lines along with fitting all lines in lines0
        '''
        This function fits each line in self.lineid in the spectrum of each spaxel and measures fluxes, linewidthsm and line centers

        Input:

        sys_vel (optional): first guess of systemic velocity of emission lines, default is sys_vel=0 (float)
        lines (optional): rest-frame wavelength of emission lines to be fitted. Default is just Ha and NII6583 (numpy array of floats)
        bin: if True work on binned simulation spectra, if False (default) fit native simulation spectra (boolean)
        pertsim: if True work on perturbed simulation spectra, if False (default) fit native simulation spectra (boolean)
        
        

        Output:
        A table named self.linefitdict which contains ra, dec and fiber Id for each fiber; flux, flux_err, wavelength, wavelength_err, sigma and sigma_err for each line in lines0.
        
        '''
        self.plot=plot
        self.lineid=lines0.astype(str)
        print('Fitting Emmission Lines:', self.lineid)
        if (self.nfib == None):
            raise Exception('Simulation has not been loaded. Run loadsim first.')
        
        self.linefitfile=self.datadir+self.simname+'/'+self.simname+'_linefits.fits'
        
        plotdir=self.datadir+self.simname+'/linefitplots/'
        if (not os.path.isdir(plotdir)):
            os.mkdir(plotdir)


        # pending to write logic to deal with bin
        wave = self.wave
        fiberid = self.fiberdata
        flux = self.flux
        err = self.err
            
        c=299792.458 # speed of light in km/s
        lines=lines0*(1+sys_vel /c)

        if loadfile:
            
            
            t=Table.read(self.linefitfile)
            self.linefitdict=t
            #print(self.linefitdict)

        else:
            
            self.linefitdict= {'fiber_id': [], 
                    'delta_ra':[], 
                    'delta_dec':[]}
        
            for i in range(len(self.lineid)):
                self.linefitdict[self.lineid[i]+'_flux']=[]
                self.linefitdict[self.lineid[i]+'_flux_err']=[]
                self.linefitdict[self.lineid[i]+'_lambda']=[]
                self.linefitdict[self.lineid[i]+'_lambda_err']=[]
                self.linefitdict[self.lineid[i]+'_sigma']=[]
                self.linefitdict[self.lineid[i]+'_sigma_err']=[]
            

            auxnfib=self.nfib
            if bin:

                auxnfib=self.nfibbin

            for i in range(auxnfib):                           # put a limit on auxnfib to examine fittings (ex. auxnfib<5)
                mask = self.fiberdata['id'] == i
                self.linefitdict['fiber_id'].append(fiberid[mask]['id'])
                self.linefitdict['delta_ra'].append(fiberid[mask]['x'].flatten())
                self.linefitdict['delta_dec'].append(fiberid[mask]['y'].flatten())       
    
                for j,line in enumerate(lines):
                    print("Fitting Line:", line)

                    if lines0[j]:         
                        plotout=plotdir+str(fiberid['id'][i])+'_'+str(lines0[j])
                    popt, pcov = fit_gauss(wave, flux[i,:], err[i,:], line, plot=self.plot, plotout=plotout)      

                   
                    self.linefitdict[str(lines0[j])+'_flux'].append(popt[0])
                    self.linefitdict[str(lines0[j])+'_flux_err'].append(np.sqrt(pcov[0, 0]))
                    self.linefitdict[str(lines0[j])+'_lambda'].append(popt[1])
                    self.linefitdict[str(lines0[j])+'_lambda_err'].append(np.sqrt(pcov[1, 1]))
                    self.linefitdict[str(lines0[j])+'_sigma'].append(popt[2])
                    self.linefitdict[str(lines0[j])+'_sigma_err'].append(np.sqrt(pcov[2, 2]))

            self.linefitdict=Table(self.linefitdict)
            self.linefitdict.write(self.linefitfile, overwrite=True)
                            

    def runpyneb(self, niter=4, bin=False, pertsim=False):

        '''
        This function will use the line fluxes to calculate the Te, ne and errors in Te nad ne running a MonteCarlo.
        The function uses the following diagnostics:

        
        TeO2: Te from "[OII] 3727+/7325+"  ; ne=100 cm-3
        TeO3: Te from "[OIII] 4363/5007"   ; ne=100 cm-3
        TeN2: Te from "[NII] 5755/6584"    ; ne=100 cm-3
        TeS2: Te from "[SII] 4072+/6720+"  ; ne=100 cm-3
        TeS3: Te from "[SIII] 6312/9069"   ; ne=100 cm-3

        neO2: ne from 3726/3729 ; Te=TeN2
        neS2: ne from 6717/6731 ; Te=TeN2

        Input:

        niter: number of MC realizations used to get errors on temperature and density, default is niter=10 (int)
        bin: if True work on binned spaxels line fluxes, if False (default) work on native spaxels line fluxes (bool)
        pertsim: if True work on perturbed simulation spaxels line fluxes, if False (default) work on native spaxels line fluxes (bool)
        
        Output:

        self.TeO2: electron temperature from O2 diagnostic
        self.TeO2err=error on electron temperature from O2 diagnostic

        self.TeO3=electron temperature from O3 diagnostic
        self.TeO3err=error on electron temperature from O3 diagnostic 

        self.TeN2=electron temperature from N2 diagnostic
        self.TeN2err=error on electron temperature from N2 diagnostic

        self.TeS2=electron temperature from S2 diagnostic
        self.TeS2err=error on electron temperature from S2 diagnostic

        self.TeS3=electron temperature from S3 diagnostic
        self.TeS3err=error on electron temperature from S3 diagnostic

        self.neO2=electron density from O2 diagnostic 
        self.neO2err=error on electron density from O2 diagnostic

        self.neS2=electron density from S2 diagnostic 
        self.neS2err=error on electron density from OS2 diagnostic

        '''

        #self.lineid = lines0.astype(str)

        if (self.nfib is None):
            RuntimeWarning('Undefined number of fibers. Probably you have not run fitlines yet', RuntimeWarning)

        # Load PyNeb packages
        atoms=pn.getAtomDict()
        O3=pn.Atom('O',3)
        S2=pn.Atom('S',2)
        N2=pn.Atom('N',2)
        O2=pn.Atom('O',2)
        S3=pn.Atom('S',3)
        #diags=pn.Diagnostics()   

        ############################################################## Electron Temperature diagnostics ##############################################################


        # TO2 temperature diagnostic #################(eduardo suggested trying blending 7319, 7320, 7330 and 7331 lines)#####################
        ne=100
        TO2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f3726=self.linefitdict['3726_flux']+np.random.randn(self.nfib)*self.linefitdict['3726_flux_err']
            f3729=self.linefitdict['3729_flux']+np.random.randn(self.nfib)*self.linefitdict['3729_flux_err']
            f7319=self.linefitdict['7319_flux']+np.random.randn(self.nfib)*self.linefitdict['7319_flux_err']
            f7330=self.linefitdict['7330_flux']+np.random.randn(self.nfib)*self.linefitdict['7330_flux_err']
            TO2[:,i]=O2.getTemDen((f3726+f3729)/(f7319+f7330), den=ne, wave1=3727, wave2=7325)

        self.TeO2 = np.nanmean(TO2, axis=1)
        self.TeO2err = np.nanstd(TO2, axis=1)
        print(self.TeO2)

        
        self.linefitdict['Temp_mean_O2']=self.TeO2
        self.linefitdict['Temp_std_O2']=self.TeO2err
        
        

        # TO3 temperature diagnostic
        ne=100
        TO3=np.zeros((self.nfib, niter))
        for i in range (niter):

            f4363=self.linefitdict['4363_flux']+np.random.randn(self.nfib)*self.linefitdict['4363_flux_err']
            f5007=self.linefitdict['5007_flux']+np.random.randn(self.nfib)*self.linefitdict['5007_flux_err']
            TO3[:,i]=O3.getTemDen((f4363)/(f5007), den=ne, wave1=4363, wave2=5007)

        self.TeO3 = np.nanmean(TO3, axis=1)
        self.TeO3err = np.nanstd(TO3, axis=1)

      
        self.linefitdict['Temp_mean_O3']=self.TeO3
        self.linefitdict['Temp_std_O3']=self.TeO3err
        

        # TN2 temperature diagnostic
        ne=100
        TN2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f5755=self.linefitdict['5755_flux']+np.random.randn(self.nfib)*self.linefitdict['5755_flux_err']
            f6584=self.linefitdict['6584_flux']+np.random.randn(self.nfib)*self.linefitdict['6584_flux_err']
            TN2[:,i]=N2.getTemDen(f5755/f6584, den=ne, wave1=5755, wave2=6584)

        self.TeN2 = np.nanmean(TN2, axis=1)
        self.TeN2err = np.nanstd(TN2, axis=1)

        
        self.linefitdict['Temp_mean_N2']=self.TeN2
        self.linefitdict['Temp_std_N2']=self.TeN2err
        

        # TS2 temperature diagnostic
        ne=100
        TS2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f4069=self.linefitdict['4069_flux']+np.random.randn(self.nfib)*self.linefitdict['4069_flux_err']
            f4076=self.linefitdict['4076_flux']+np.random.randn(self.nfib)*self.linefitdict['4076_flux_err']
            f6717=self.linefitdict['6717_flux']+np.random.randn(self.nfib)*self.linefitdict['6717_flux_err']
            f6731=self.linefitdict['6731_flux']+np.random.randn(self.nfib)*self.linefitdict['6731_flux_err']
            TS2[:,i]=S2.getTemDen((f4069+f4076)/(f6717+f6731), den=ne, wave1=4072, wave2=6720)

        self.TeS2 = np.nanmean(TS2, axis=1)
        self.TeS2err = np.nanstd(TS2, axis=1)

        
        self.linefitdict['Temp_mean_S2']=self.TeS2
        self.linefitdict['Temp_std_S2']=self.TeS2err
        

        # TS3 temperature diagnostic
        ne=100
        TS3=np.zeros((self.nfib, niter))
        for i in range (niter):

            f6312=self.linefitdict['6312_flux']+np.random.randn(self.nfib)*self.linefitdict['6312_flux_err']
            f9069=self.linefitdict['9069_flux']+np.random.randn(self.nfib)*self.linefitdict['9069_flux_err']
            TN2[:,i]=S3.getTemDen(f6312/f9069, den=ne, wave1=6312, wave2=9069)

        #self.TeS3 = np.nanmean(TS3, axis=1)
        #self.TeS3err = np.nanstd(TS3, axis=1)
             
        
        #self.linefitdict['Temp_mean_S3']=self.TeS3
        #self.linefitdict['Temp_std_S3']=self.TeS3err   
        self.linefitdict['delta_ra']=self.linefitdict['delta_ra']
        self.linefitdict['delta_dec']=self.linefitdict['delta_dec']

        self.linefitdict.write('diag_Temp_Den.fits', overwrite=True)
        print(Table.read('diag_Temp_Den.fits'))

    ############################################################## Electron density diagnostics ##############################################################

        # NO2 electron density diagnostic
        NO2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f3726=self.linefitdict['3726_flux']+np.random.randn(self.nfib)*self.linefitdict['3726_flux_err']
            f3729=self.linefitdict['3729_flux']+np.random.randn(self.nfib)*self.linefitdict['3729_flux_err']
            NO2[:,i]=O2.getTemDen(f3726/f3729, tem=TN2[:,i], wave1=3726, wave2=3729)

        self.neO2 = np.nanmean(NO2, axis=1)
        self.neO2err = np.nanstd(NO2, axis=1)
        #print(self.neO2)

        # NS2 electron density diagnostic
        NS2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f6717=self.linefitdict['6717_flux']+np.random.randn(self.nfib)*self.linefitdict['6717_flux_err']
            f6731=self.linefitdict['6731_flux']+np.random.randn(self.nfib)*self.linefitdict['6731_flux_err']
            NS2[:,i]=S2.getTemDen(f6717/f6731, tem=TN2[:,i], wave1=6731, wave2=6717)

        self.neS2 = np.nanmean(NS2, axis=1)
        self.neS2err = np.nanstd(NS2, axis=1)
        #print(self.neS2)

    def plot(self, z1, min, max, nlevels=40, title='line_map', output='line_map', bin=False, pertsim=False):

        sel=np.isfinite(z1)
        
        newtable=Table.read('diag_Temp_Den.fits')
        fig, ax = plt.subplots(figsize=(8,5))
        triang = tri.Triangulation(self.linefitdict['delta_ra'][sel], self.linefitdict['delta_dec'][sel]) 
        c = ax.tricontourf(triang, z1[sel], levels=np.linspace(min, max, nlevels))    
        plt.colorbar(c) 
        ax.set_title(title)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.axis('equal')
        plt.savefig(output+'.png')
        

        r=np.sqrt(self.linefitdict['delta_ra']**2+self.linefitdict['delta_dec']**2)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(r[sel], z1[sel], '.')
        ax.set_ylim(min, max)
        ax.set_xlim(0, 260)
        ax.set_ylabel(title)
        ax.set_xlabel('Radius')
        ax.legend()
        plt.savefig(output+'_rad.png')
        
        '''
        ########################################## Example #########################################
        # Mean SIII Temperature
        z1=TS3_mean
        print (z1)
        plotmap(z1, 1000, 9000, title=r'T$_{NII}$ (mean) '+s[2], output='./'+simname+'/TS3_mean_'+s[1]+'_'+s[2])
        plt.show()
        '''


    def bin(self, rbinmax, drbin, pertsim=False):
        '''
        This function will a 3d array which will provide us the binned line flux for each self.lineid in each spaxel.
        
        Input:
        rbinmax:max radius for binning(float)
        drbin: no. of bins (int)
        
        Output:
        Binned spectra for lineflux in each spaxel
        
        '''

        radius = np.sqrt(self.fiberdata['x']**2 + self.fiberdata['y']**2)
        bins = np.arange(drbin/2, rbinmax-drbin/2, drbin)
        nspax=np.zeros(len(bins))
        header = self.header
        binned_fluxes = np.zeros((len(bins)-1, len(self.wave)))
        binned_err = np.zeros((len(bins)-1, len(self.wave)))
        newx = []
        for i in range(len(bins)-1):
            nflux, nerr, nsel = bin_spectra(bins[i]-drbin, bins[i]+drbin, radius, self.flux, self.err)
            nspax[i]=nsel
            binned_fluxes[i] = nflux
            binned_err[i] = nerr
            newx.append(bins[i])
        hdu_primary = fits.PrimaryHDU(header=header)
        hdu_target = fits.ImageHDU(data=binned_fluxes, name='TARGET')
        hdu_errors = fits.ImageHDU(data=binned_err, name='ERR')
        hdu_wave = fits.ImageHDU(data=self.wave, name='WAVE')
        newtable = {'id': range(0, len(bins)-1),
                    'y': np.zeros(len(bins)-1),
                    'x': newx}
        newtable = Table(newtable)
        hdu_table = fits.BinTableHDU(newtable, name='FIBERID')
        hdul = fits.HDUList([hdu_primary, hdu_target, hdu_errors, hdu_wave, hdu_table])
        
        filename =self.datadir+self.simname+'_binned_'+str(int(self.exptime))+'_flux.fits'
        directory=self.datadir+self.simname+'_binned/'
        if ( not os.path.isdir(directory)):
            os.mkdir(directory)
        plotdir=directory+'/linefitplots/'
        if ( not os.path.isdir(plotdir)):
           os.mkdir(plotdir)
        hdul.writeto(filename, overwrite=True)
        plt.plot(bins, nspax)
        plt.show()


    def pertsim(self, npoints=30, dim=3, n=3, k0=0.5, dk0=0.05 ):
       
        '''
        This function will contain the perturbed line emmissivities for each line Id, 
        it is a 4D array containing the line emissivities, Te, Ne and line ids.

        Input:



        Output:

        '''
        

        #new_field=field_powerlaw(n, npoints)
        new_field=field_delta(k0, dk0, dim, npoints)

        A=0.1 # fluctuation amplitude standard deviation
        norm_field=new_field/np.std(new_field)*A

        print(np.mean(norm_field), np.std(norm_field))
        plt.hist(norm_field.flatten())
        plt.show()

        xx, yy, zz = np.mgrid[0:npoints, 0:npoints, 0:npoints]
        r = np.sqrt((xx-npoints/2)**2 + (yy-npoints/2)**2 + (zz-npoints/2)**2)

        with imageio.get_writer('./new_field_test_'+str(float(k0))+'.gif', mode='I') as writer:
            for slice in new_field:
                writer.append_data(slice)



#################################################################################### Functions used in above methods #################################################################################################

################################################ Functions used in fitlines method ####################################################

def gaussian(wave,flux,mean,sd):
    '''
    This function evaluates a 1D Gaussian
    
    Input:
    wave=wavelength array(1D numpy array)
    flux=line flux (float)
    mean=line centroid (float)
    sd=1 sigma line width (float) 

    Output:
    gaussian profile (1D numpy array)
    '''
    return flux/(np.sqrt(2*np.pi)*sd)*np.exp((wave-mean)**2/(-2*sd**2))

def error_func(wave, gaussian, popt, pcov, e=1e-7):
    
    '''
    This function produce error to all emission line in lines.

    Input:
    wave: Wavelength array of spectra in simulations (1D numpy array)
    gaussian: To fit gaussian profile to errors on emission lines flux
    popt:Optimal values for the parameters (1D numpy array)
    pcov: Estimated covariance of popt (3D matrix)

    Outut:
    standard deviation on each parameter    
    '''
    
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

def fit_gauss(wave, spectrum, error, lwave, dwave=4, plot=True, plotout='linefit'):
    '''
    This function produce gaussion fit to all emission line in lines.

    Input:
    wave: wavelength array of spectra in simulations (1D numpy array)
    spectrum: flux array(1D numpy array)
    error: flux error array(1D numpy array)
    lwave: initial approximation (1D numpy array)
    
    Outut: 
    
    ''' 
    sel=(wave>lwave-dwave/2)*(wave<lwave+dwave/2)

    p0=(np.abs(spectrum[sel].max()), lwave, 0.7)

    try: 
        popt, pcov=curve_fit(gaussian, wave[sel], spectrum[sel], sigma=error[sel], p0=p0, absolute_sigma=True, bounds=((0,lwave-1.0,0.3),(np.inf,lwave+1.0,2.0)))

    except RuntimeError: 
        popt=np.array([-99, p0[1], p0[2]])
        pcov=np.zeros((3,3))
        pcov[0,0]=np.sum(error[sel]**2)

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
        ax.fill_between(xm, ym+3*sigma, np.max([ym-ym,ym-3*sigma], axis=0), alpha=0.7, 
                    linewidth=3, label='error limits')
        
        ax.set_ylim(-0.4*spectrum[sel].max(),1.7*np.max([spectrum[sel].max(), (ym+3*sigma).max()]))
        plt.ylabel("Flux, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$")
        plt.xlabel("wavelength, $\AA$")

        
        ax.legend()
        plt.savefig(plotout+'.png')
    return popt, pcov
##################################################################################################################################################


################################################### Function used to bin spectra in simulations #######################################################

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

#################################################### Functions used in perturbing the simulation ######################################################

def k_vector(npoints):
    k1 = np.arange(npoints/2+1)
    k2 = np.arange(-npoints/2+1, 0)
    
    kvector = 2*np.pi/ npoints* np.concatenate([k1, k2])
    return kvector


def pk_vector_delta(kvector, dim, k0, dk0):
    
    npoints = len(kvector)
    shape = [npoints] * dim
    kk = np.zeros(shape)
    
      
    for i, j, k in itertools.product(range(npoints), range(npoints), range(npoints)):
        kk[i, j, k] = np.sqrt(kvector[i]**2 + kvector[j]**2 + kvector[k]**2)
               
    print(np.min(kk),np.max(kk))
    
    pk=np.zeros_like(kk)
    sel=(kk > k0-dk0/2)*(kk < k0+dk0/2)
    pk[sel]=1
    
    xx, yy, zz = np.mgrid[0:npoints, 0:npoints, 0:npoints]
    r = np.sqrt((xx-npoints/2)**2 + (yy-npoints/2)**2 + (zz-npoints/2)**2)

    mask = r > npoints/2
    mask2 = r < 0.8 * npoints/2
    pk[mask*mask2]=0
    #pk[mask]=0

    pk[0,0,0] = 0
    
    return pk

def field_delta(k0, dk0, dim, npoints):
    
    k = k_vector(npoints)
    pk = pk_vector_delta(k, dim, k0, dk0)
    Pk1 = np.zeros_like(pk)
    #Pk1 /= Pk1.sum()
    Pk1 = pk

    field=np.random.randn(npoints, npoints, npoints)
    fft_field=fftn(field)
    
    pspect_field = np.sqrt(Pk1) * fft_field
    new_field = np.real(ifftn(pspect_field))
    
    return new_field

##################################################################################################################################################

################################################### Function used to run MC simulations ##########################################################
'''
def new_measurements(value, errs):
    
    This function will be used to run monte carlo simulation on line fluxes.
    Input:
    value:line_flux
    errs: line_fluxerr
    
    Output:
    random values (1D numpy array)
    
    
    delta = np.random.randn(len(errs)) * errs
    new_values = np.max([value + delta, np.zeros(len(errs))], axis=0)
    return new_values
'''

















