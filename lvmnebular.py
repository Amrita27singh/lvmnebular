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
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import astropy.units as unit

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
        self.linefitdict=None

        self.nfibbin = None
        self.binflux = None
        self.binerr = None

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
        
        self.radbins=None

        # voronoi binning attributes
        self.newtable=None
        self.nbins=None
        self.vorbinflux=None
        self.vorbinerr=None
        self.binid=None
        self.target_sn=None

        #sn_radial binning attributes
        self.vals=None
        self.rbin=None
        self.snbin=None
        self.snbinned_flux=None
        self.snbinned_err=None
        self.rbinright=None
        self.rbinleft=None

        #projectedTe attributes
        self.R=None #projected R(in pc)
        self.Teproj=None
        self.aproj=None
        self.distance=None


        #Chemical abundance from pyneb attributes
        self.OppH=None
        self.OpH = None
        self.NpH = None
        self.SppH = None
        self.SpH = None

        self.int_OppH=None
        self.int_TO3=None
        self.int_OpH=None
        self.int_TO2=None
        self.int_TN2=None
        self.int_NpH=None 
        self.int_TS3 = None
        self.int_SppH = None
        self.int_TS2 = None
        self.int_SpH = None

        self.avgTe = None
        self.avg_abund = None

        #Chemical abundance using emperical formulae from Perez montero 2017 
        self.Abund_O2 = None
        self.Abund_O3 = None
        self.Abund_N2 = None
        self.Abund_S2 = None
        self.Abund_S3 = None

        #chi computation for varying amp and scales of perturbation: gives out: Tp(ion)-T0/T0
        self.Chi_TeN = None
        self.Chi_TeO = None
        self.Chi_TeS = None

    def loadsim(self, simname, exptime, datadir='/home/amrita/LVM/lvmnebular/', vorbin=False, snbin=False):

        self.datadir=datadir
        self.simname=simname
        self.exptime=exptime

        
        if vorbin:
            self.simfile=self.datadir+self.simname+'/'+self.simname+'_vorbinned'+'/'+self.simname+'_vorbinned_linear_full_'+str(int(self.exptime))+'_flux.fits'

        elif snbin:
            self.simfile=self.datadir+self.simname+'/'+self.simname+'_snbinned'+'/'+self.simname+'_snbinned_linear_full_'+str(int(self.exptime))+'_flux.fits'

        else:
            self.simfile = self.datadir+self.simname+'/outputs/'+self.simname+'_linear_full_'+str(int(self.exptime))+'_flux.fits'    

        print("Loading simulation: "+self.datadir+self.simname+'\n')
        print("Loading simfile: "+self.simfile)

        if ( not os.path.isdir(self.datadir+self.simname)):
            raise Exception("Simulation "+self.simname+" does not exist. Run run_simulation.py first")
            
        with fits.open(self.simfile) as hdu:
            self.header = hdu[0].header
            self.wave = hdu['WAVE'].data ##1D array
            self.flux = hdu['TARGET'].data ##2D array(no. of fibers, wave)
            self.err = hdu['ERR'].data
            self.fiberdata = Table.read(hdu['FIBERID'])

        self.nfib=len(self.fiberdata)
        print("no.of bins:", self.nfib)

        #reading True Te and ne values to overplot with Te and ne profiles in background.
        hdu=fits.open(self.datadir+self.simname+'/'+'testneb_tutorial3_ex1.fits')
        vals=hdu['Comp_0_PhysParams'].data
        self.vals=vals

    def fitlines(self, sys_vel=0, lines0= np.array([6563, 6584]) , radbin=False, vorbin=False, snbin=False, pertsim=False, rbinmax=250, drbin=20, loadfile=True, plot=False):  
        '''
        This function fits each line in self.lineid in the spectrum of each spaxel and measures fluxes, linewidthsm and line centers

        Input:

        sys_vel (optional): first guess of systemic velocity of emission lines, default is sys_vel=0 (float)
        lines (optional): rest-frame wavelength of emission lines to be fitted. Default is just Ha and NII6583 (numpy array of floats)
        radbin: if True work on radially binned simulation spectra, if False (default) fit native simulation spectra (boolean)
        vorbin: if True work on voronoi binned simulation spectra, if False (default) fit native simulation spectra (boolean)
        snbin: if True work on binned simulation spectra, if False (default) fit native simulation spectra (boolean)
        pertsim: if True work on perturbed simulation spectra, if False (default) fit native simulation spectra (boolean)
        rbinmax:max radius for binning(float, default=250)
        drbin: no. of bins (int, default=20)
        

       returns:
        A table named self.linefitdict which contains ra, dec and fiber Id for each fiber; flux, flux_err, wavelength, wavelength_err, sigma and sigma_err for each line in lines0.
        
        '''
        self.plot = plot
        self.lineid = lines0.astype(str)
        print('Fitting Emmission Lines:', self.lineid)

        if (self.nfib == None):
            raise Exception('Simulation has not been loaded. Run loadsim first.')
        

        if vorbin:
            plotdir = self.datadir+self.simname+'/'+self.simname+'_vorbinned/'+'linefitplots/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)
            outfilename = self.datadir+self.simname+'/'+self.simname+'_vorbinned/'+self.simname+'_vorbinned_linefits.fits'

        elif snbin:
            plotdir = self.datadir+self.simname+'/'+self.simname+'_vorbinned/'+'linefitplots/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)
            outfilename = self.datadir+self.simname+'/'+self.simname+'_snbinned/'+self.simname+'_snbinned_linefits.fits'


        elif radbin:
            self.rbinmax = rbinmax
            self.drbin = drbin
            self.radialbin(rbinmax, drbin, pertsim=False)

            self.simfile = self.datadir+self.simname+'/'+self.simname+'_radbinned'+'/'+self.simname+'_radbinned_linear_full_'+str(int(self.exptime))+'_flux.fits'

            with fits.open(self.simfile) as hdu:
                self.header = hdu[0].header
                self.wave = hdu['WAVE'].data ##1D array
                self.flux = hdu['TARGET'].data ##2D array(no. of fibers, wave)
                self.err = hdu['ERR'].data
                self.fiberdata = Table.read(hdu['FIBERID'])

            self.nfib = len(self.fiberdata)
            print("no.of bins:", self.nfib)

            plotdir = self.datadir+self.simname+'/'+self.simname+'_radbinned/'+'linefitplots/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir) 
            outfilename = self.datadir+self.simname+'/'+self.simname+'_radbinned/'+self.simname+'_radbinned_linefits.fits'   

        else:
            plotdir = self.datadir+self.simname+'/linefitplots/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)
            outfilename = self.datadir+self.simname+'/'+self.simname+'_linefits.fits'

        self.linefitfile = outfilename 
        print("linefitfile:",self.linefitfile)

        wave = self.wave
        fiberid = self.fiberdata
        flux = self.flux
        err = self.err
            
        c = 299792.458   #speed of light in km/s
        lines = lines0*(1+sys_vel /c)

        if loadfile:
            
            t = Table.read(self.linefitfile)
            self.linefitdict = t

        else:
            
            self.linefitdict = {'fiber_id': [], 
                    'delta_ra':[], 
                    'delta_dec':[]}
        
            for i in range(len(self.lineid)):
                self.linefitdict[self.lineid[i]+'_flux']=[]
                self.linefitdict[self.lineid[i]+'_flux_err']=[]
                self.linefitdict[self.lineid[i]+'_lambda']=[]
                self.linefitdict[self.lineid[i]+'_lambda_err']=[]
                self.linefitdict[self.lineid[i]+'_sigma']=[]
                self.linefitdict[self.lineid[i]+'_sigma_err']=[]
            
            auxnfib = self.nfib
            print(self.nfib)

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
            self.linefitdict.write(outfilename, overwrite=True)

    def runpyneb(self, niter=4, pertsim=False):

        '''
        This function will use the line fluxes to calculate the Te, ne and errors in Te nad ne running a MonteCarlo.
        The function uses the following diagnostics:

        
        TeO2: Te from "[OII] 3727+/7325+"  ; ne = 120 cm-3
        TeO3: Te from "[OIII] 4363/5007"   ; ne = 120 cm-3
        TeN2: Te from "[NII] 5755/6584"    ; ne = 120 cm-3
        TeS2: Te from "[SII] 4072+/6720+"  ; ne = 120 cm-3
        TeS3: Te from "[SIII] 6312/9069"   ; ne = 120 cm-3

        neO2: ne from 3726/3729 ; Te=TeN2
        neS2: ne from 6731/6716 ; Te=TeN2

        returns:

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


        # TO2 temperature diagnostic
        ne = 120
        TO2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f3726= self.linefitdict['3726_flux']+np.random.randn(self.nfib)*self.linefitdict['3726_flux_err']
            f3729= self.linefitdict['3729_flux']+np.random.randn(self.nfib)*self.linefitdict['3729_flux_err']
            f7319= self.linefitdict['7319_flux']+np.random.randn(self.nfib)*self.linefitdict['7319_flux_err']
            f7320= np.zeros(self.nfib)
            f7330= np.zeros(self.nfib)
            f7331= self.linefitdict['7331_flux']+np.random.randn(self.nfib)*self.linefitdict['7331_flux_err']

            TO2[:,i]=O2.getTemDen((f7320+f7331+f7319+f7330)/(f3726+f3729), den=ne, wave1= 3727.5, wave2=7325)

        self.TeO2 = np.nanmean(TO2, axis=1)
        self.TeO2err = np.nanstd(TO2, axis=1)
        
        self.linefitdict['TeO2']=self.TeO2
        self.linefitdict['TeO2err']=self.TeO2err
        
        # TO3 temperature diagnostic
        TO3 = np.zeros((self.nfib, niter))
        
        for i in range (niter):

            f4363 = self.linefitdict['4363_flux']+np.random.randn(self.nfib)*self.linefitdict['4363_flux_err']
            f5007 = self.linefitdict['5007_flux']+np.random.randn(self.nfib)*self.linefitdict['5007_flux_err']
            TO3[:,i] = O3.getTemDen((f4363)/(f5007), den=ne, wave1=4363, wave2=5007)

    
        self.TeO3 = np.nanmean(TO3, axis=1)
        self.TeO3err = np.nanstd(TO3, axis=1)

        self.linefitdict['TeO3']=self.TeO3
        self.linefitdict['TeO3err']=self.TeO3err

        # TN2 temperature diagnostic
        TN2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f5755=self.linefitdict['5755_flux']+np.random.randn(self.nfib)*self.linefitdict['5755_flux_err']
            f6584=self.linefitdict['6584_flux']+np.random.randn(self.nfib)*self.linefitdict['6584_flux_err']
            TN2[:,i]=N2.getTemDen(f5755/f6584, den=ne, wave1=5755, wave2=6584)

        self.TeN2 = np.nanmean(TN2, axis=1)
        self.TeN2err = np.nanstd(TN2, axis=1)
        
        self.linefitdict['TeN2']=self.TeN2
        self.linefitdict['TeN2err']=self.TeN2err        

        # TS2 temperature diagnostic
        
        TS2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f4069=self.linefitdict['4069_flux']+np.random.randn(self.nfib)*self.linefitdict['4069_flux_err']
            f4076=self.linefitdict['4076_flux']+np.random.randn(self.nfib)*self.linefitdict['4076_flux_err']
            f6716=self.linefitdict['6716_flux']+np.random.randn(self.nfib)*self.linefitdict['6716_flux_err']
            f6731=self.linefitdict['6731_flux']+np.random.randn(self.nfib)*self.linefitdict['6731_flux_err']
            TS2[:,i]=S2.getTemDen((f4069+f4076)/(f6716+f6731), den=ne, wave1=4072.5, wave2=6723.5)
            
        self.TeS2 = np.nanmean(TS2, axis=1)
        self.TeS2err = np.nanstd(TS2, axis=1)
        
        self.linefitdict['TeS2']=self.TeS2
        self.linefitdict['TeS2err']=self.TeS2err       

        # TS3 temperature diagnostic
        TS3=np.zeros((self.nfib, niter))
        for i in range (niter):

            f6312=self.linefitdict['6312_flux']+np.random.randn(self.nfib)*self.linefitdict['6312_flux_err']
            f9069=self.linefitdict['9069_flux']+np.random.randn(self.nfib)*self.linefitdict['9069_flux_err']
            TS3[:,i]=S3.getTemDen(f6312/f9069, den=ne, wave1=6312, wave2=9069)

        self.TeS3 = np.nanmean(TS3, axis=1)
        self.TeS3err = np.nanstd(TS3, axis=1)
            
    
        self.linefitdict['TeS3']=self.TeS3
        self.linefitdict['TeS3err']=self.TeS3err   
        
        
                 ############################################################## Electron density diagnostics ##############################################################

        # NO2 electron density diagnostic
        NO2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f3726=self.linefitdict['3726_flux']+np.random.randn(self.nfib)*self.linefitdict['3726_flux_err']
            f3729=self.linefitdict['3729_flux']+np.random.randn(self.nfib)*self.linefitdict['3729_flux_err']
            NO2[:,i]=O2.getTemDen(f3726/f3729, tem=TN2[:,i], wave1=3726, wave2=3729)

        self.neO2 = np.nanmean(NO2, axis=1)
        self.neO2err = np.nanstd(NO2, axis=1)
        
        self.linefitdict['neO2']=self.neO2
        self.linefitdict['neO2err']=self.neO2err   

        # NS2 electron density diagnostic
        NS2=np.zeros((self.nfib, niter))
        for i in range (niter):

            f6716=self.linefitdict['6716_flux']+np.random.randn(self.nfib)*self.linefitdict['6716_flux_err']
            f6731=self.linefitdict['6731_flux']+np.random.randn(self.nfib)*self.linefitdict['6731_flux_err']
            NS2[:,i]=S2.getTemDen(f6731/f6716, tem=TN2[:,i], wave1=6731, wave2=6716) ############################### check ##########

        self.neS2 = np.nanmean(NS2, axis=1)
        self.neS2err = np.nanstd(NS2, axis=1)
        
        self.linefitdict['neS2']=self.neS2
        self.linefitdict['neS2err']=self.neS2err
        

        self.linefitdict['delta_ra']=self.linefitdict['delta_ra']
        self.linefitdict['delta_dec']=self.linefitdict['delta_dec']

        self.linefitdict.write(self.datadir+'/'+self.simname+'/'+self.simname+' diag_Temp_Den.fits', overwrite=True)

    def avg_Te(self, ion):    #reproducing avergae temperature from Eduardo's 2023 Nature paper
        
        avgTe = np.sum(self.vals[1]*self.vals[2]*ion)/np.sum(self.vals[2]*ion)                
        self.avgTe = avgTe

    def avg_abundance(self, ion):    #reproducing avergae abund
        
        avg_abund = np.sum(ion)/np.sum(self.vals[3])                
        self.avg_abund = avg_abund

    def radialbin(self, rbinmax, drbin, pertsim=False):

        '''
        This function will a 3d array which will provide us the binned line flux for each self.lineid in each spaxel.
        
        Input:
        rbinmax:max radius for binning(float)
        drbin: no. of bins (int)
        
        returns:
        Binned spectra for lineflux in each spaxel
        '''
        self.simfile = self.datadir+self.simname+'/outputs/'+self.simname+'_linear_full_'+str(int(self.exptime))+'_flux.fits'

        with fits.open(self.simfile) as hdu:
            self.header = hdu[0].header
            self.wave = hdu['WAVE'].data ##1D array
            self.flux = hdu['TARGET'].data ##2D array(no. of fibers, wave)
            self.err = hdu['ERR'].data
            self.fiberdata = Table.read(hdu['FIBERID'])

        radius = np.sqrt(self.fiberdata['x']**2 + self.fiberdata['y']**2)
        bins = np.arange(drbin/2, rbinmax-drbin/2, drbin)
        self.radbins=len(bins)
        nspax=np.zeros(len(bins))
        header = self.header
        radbinned_fluxes = np.zeros((len(bins)-1, len(self.wave)))
        radbinned_err = np.zeros((len(bins)-1, len(self.wave)))

        newx = []

        for i in range(len(bins)-1):
            nflux, nerr, nsel = binrad_spectra(bins[i]-drbin, bins[i]+drbin, radius, self.flux, self.err)
            nspax[i]=nsel
            radbinned_fluxes[i] = nflux
            radbinned_err[i] = nerr
            newx.append(bins[i])

        hdu_primary = fits.PrimaryHDU(header=header)
        hdu_target = fits.ImageHDU(data=radbinned_fluxes, name='TARGET')
        hdu_errors = fits.ImageHDU(data=radbinned_err, name='ERR')
        hdu_wave = fits.ImageHDU(data=self.wave, name='WAVE')
        newtable = {'id': range(0, len(bins)-1),
                    'x': np.zeros(len(bins)-1),
                    'y': newx}
        
        newtable = Table(newtable)
        self.newtable=newtable
        
        hdu_table = fits.BinTableHDU(newtable, name='FIBERID')
        hdul = fits.HDUList([hdu_primary, hdu_target, hdu_errors, hdu_wave, hdu_table])

        filename=self.simname+'_radbinned'+'_linear_full_'+str(int(self.exptime))+'_flux.fits'
        directory=self.simname+'/'+self.simname+'_radbinned/'
        
        if ( not os.path.isdir(directory)):
            os.mkdir(directory)
            
        plotdir=directory+'/linefitplots/'
        if ( not os.path.isdir(plotdir)):
           os.mkdir(plotdir)

        hdul.writeto(directory+filename, overwrite=True)
        plt.plot(bins, nspax)
        plt.xlabel('nbins')
        plt.ylabel('nspaxels')
        plt.show()

    def sn_radialbin(self,  target_sn=100,  lineid='6563', rmin=0, rmax=250, pertsim=False):

        '''
        This function will be a 3d array which will provide us the binned line flux for each self.lineid in each spaxel.
    
        Input:
        targetsnr: The desired minimum snr (int; default is 100)
        lineid: rest frame wavelength of emission lines

        Returns:
        Radially binned flux, error spectrum with constant snr
        '''
        if self.linefitdict is None:
            raise Exception('Emission lines not fit yet, run fitlines first.')

        rbinleft= np.array([0])
        rbinright= np.array([])
        snbin= np.array([])
        npix=np.array([])

        signal, noise=self.linefitdict[lineid+'_flux'], self.linefitdict[lineid+'_flux_err']
        radius = np.sqrt(self.fiberdata['x']**2 + self.fiberdata['y']**2)

        selected = (radius >= rmin)*(radius < rmax)
        radius_unique=np.unique(radius[selected])

        snbinned_flux = np.zeros((len(radius_unique), len(self.wave)))
        snbinned_err = np.zeros((len(radius_unique),  len(self.wave)))
        newx=[]
        
        cnt=0
        for i, rad in enumerate(radius_unique):
            indices=(radius > rbinleft[-1])*(radius <= rad)*selected  #checking all three conditions
            snr_rad = np.sum(signal[indices])/np.sqrt(np.sum(noise[indices]**2))

            if (snr_rad >= target_sn):
                rbinright = np.append(rbinright, rad)
                rbinleft = np.append(rbinleft, rad)
                snbin = np.append(snbin, snr_rad)
                npix=np.append(npix, indices.sum())
                snbinned_flux[cnt, :] = np.sum(self.flux[indices, :], axis=0)
                snbinned_err[cnt, :] = np.sum(self.err[indices, :], axis=0)
                newx.append(rad)
                cnt=cnt+1
                #print(cnt)


        #print("GB TEST")
        #print(np.shape(snbinned_flux))
        snbinned_flux=snbinned_flux[0:cnt,:]
        snbinned_err=snbinned_err[0:cnt,:]
        #print(np.shape(snbinned_flux))

        self.snbinned_flux=snbinned_flux
        self.snbinned_err=snbinned_err
        self.rbinright=rbinright
        self.rbinleft=rbinleft
        

        hdu_primary = fits.PrimaryHDU(header=self.header)
        hdu_target = fits.ImageHDU(data=snbinned_flux, name='TARGET')
        hdu_errors = fits.ImageHDU(data=snbinned_err, name='ERR')
        hdu_wave = fits.ImageHDU(data=self.wave, name='WAVE')
        newtable = {'id': range(len(snbinned_flux)),
                    'x': rbinright,
                    'y': np.zeros(len(snbinned_flux))}
        
        newtable = Table(newtable)
        self.newtable=newtable
        
        hdu_table = fits.BinTableHDU(newtable, name='FIBERID')
        hdul = fits.HDUList([hdu_primary, hdu_target, hdu_errors, hdu_wave, hdu_table])

        filename=self.simname+'_snbinned'+'_linear_full_'+str(int(self.exptime))+'_flux.fits'
        directory=self.simname+'/'+self.simname+'_snbinned/'
        
        if (not os.path.isdir(directory)):
            os.mkdir(directory)
            
        plotdir=directory+'/linefitplots/'
        if ( not os.path.isdir(plotdir)):
           os.mkdir(plotdir)

        # printing values
        #print(len(indices), rbinright, len(rbinright), len(rbinleft), len(snbin), npix, len(npix), indices)

        plt.plot(rbinright, snbin)
        plt.xlabel('rbinright')
        plt.ylabel('snbin')
        hdul.writeto(directory+filename, overwrite=True)
        
    def voronoibin(self, target_sn=500, lineid='6563', label='flux', plot=False, pertsim=False):
        '''
        Input:
        targetsnr: The desired minimum snr (int; default is 10)
        lineid: rest frame wavelength of emission lines
        label: used to put title of plot

        Returns:
        Binned flux, error spectrum in each spaxel.
        '''

        if self.linefitdict is None:
            raise Exception('Emission lines not fit yet, run fitlines first.')
        
        x, y=self.fiberdata['x'], self.fiberdata['y']
        signal, noise=self.linefitdict[lineid+'_flux'], self.linefitdict[lineid+'_flux_err']

        bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, noise, target_sn, cvt=True, pixelsize=None, plot=True, quiet=True, sn_func=None, wvt=False)
        
        print(len(nPixels))

        self.nbins=len(nPixels)
        vorbinflux=np.zeros((self.nbins, len(self.wave)))
        vorbinerr=np.zeros((self.nbins, len(self.wave)))
        binid=np.unique(bin_number)
        header = self.header
        
        for i in range(self.nbins):

            sel=bin_number==binid[i]
            vorbinflux[i,:]=np.sum(self.flux[sel,:], axis=0)
            vorbinerr[i,:]=np.sum(self.err[sel,:], axis=0)

        print(vorbinflux)
            
        hdu_primary = fits.PrimaryHDU(header=header)
        hdu_target = fits.ImageHDU(data=vorbinflux, name='TARGET')
        hdu_errors = fits.ImageHDU(data=vorbinerr, name='ERR')
        hdu_wave = fits.ImageHDU(data=self.wave, name='WAVE')
        newtable = {'id': range(0, self.nbins),
                    'x': x_gen,
                    'y': y_gen,
                    }

        newtable = Table(newtable)
        hdu_table = fits.BinTableHDU(newtable, name='FIBERID')
        hdul = fits.HDUList([hdu_primary, hdu_target, hdu_errors, hdu_wave, hdu_table])
        
        self.vorbinflux=vorbinflux
        self.vorbinerr=vorbinerr  

        filename=self.simname+'_vorbinned'+'_linear_full_'+str(int(self.exptime))+'_flux.fits'
        directory=self.simname+'/'+self.simname+'_vorbinned/'
        if ( not os.path.isdir(directory)):
            os.mkdir(directory)

        # plotting directory storing the spectra (I think these plots aren't important, remove??)
        plotdir=directory+'/vorbin_fluxplots/'
        if ( not os.path.isdir(plotdir)):
           os.mkdir(plotdir)

        hdul.writeto(directory+filename, overwrite=True)
        
        if plot:
            fig, ax=plt.subplots()
            ax.plot(self.wave, vorbinflux[i,:], label='flux')
            ax.set_xlabel('wavelength $A$')
            ax.set_ylabel('Flux erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
            plt.savefig(plotdir+'/'+lineid+'.png')
            plt.show() 

    def pertsim(self, npoints=90, dim=3, n=3, k0=0.5, dk0=0.05 ):
       
        '''
        This function will contain the perturbed line emmissivities for each line Id, 
        it is a 4D array containing the line emissivities at each value of x, y and z coordinates, and line ids.

        Input:



        Returns:

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

    def projectedTe(self, a0, n=1000):

        #loading true 3D Radius (108 values)
        r0=self.vals[0]
        #loading true Temperature 
        T0=self.vals[1] 
        #loading ionic abundance of NII
        a=a0
        #loading true electron density
        ne = self.vals[0]

        r0=r0[1:]
        T0=T0[1:]
        a=a[1:]
        ne=ne[1:]

        cubic_interp_T0 = interp1d(r0, T0, kind='cubic', axis=-1, bounds_error=False)
        cubic_interp_a  = interp1d(r0, a, kind='cubic', axis=-1, bounds_error=False)
        cubic_interp_ne  = interp1d(r0, ne, kind='cubic', axis=-1, bounds_error=False)

        R=np.linspace(0, np.max(r0), n) # on-sky projected radius
        Teproj=np.zeros_like(R) # on-sky projected temperature
        aproj=np.zeros_like(R)

        for i,Ri in enumerate(R):

            if np.any(Ri > np.min(r0)):
                theta_max=np.arccos(Ri/np.max(R))*0.9999999
                theta=np.linspace(-theta_max, theta_max, n)             
                            
                r0aux=Ri/np.cos(theta)
                T0aux=cubic_interp_T0(r0aux)
                aaux=cubic_interp_a(r0aux)
                neaux=cubic_interp_ne(r0aux)

                T0aux[~np.isfinite(T0aux)]=0
                aaux[~np.isfinite(aaux)]=0
                neaux[~np.isfinite(neaux)]=0

                Teproj[i]=trapezoid(T0aux*aaux*neaux*np.cos(theta)**(-2), x=theta)/trapezoid(aaux*neaux*np.cos(theta)**(-2), x=theta)
                aproj[i]=trapezoid(aaux*neaux*np.cos(theta)**(-2), x=theta)/trapezoid(neaux*np.cos(theta)**(-2), x=theta)


            elif np.any(Ri>0) and np.any(Ri<= np.min(r0)):
                theta_max=np.arccos(Ri/np.max(R))*0.9999999
                theta_min=np.arccos(Ri/np.min(r0))*0.9999999 
                theta=np.linspace(theta_min, theta_max, n)             
                            
                r0aux=Ri/np.cos(theta)
                T0aux=cubic_interp_T0(r0aux)
                aaux=cubic_interp_a(r0aux)
                neaux=cubic_interp_ne(r0aux)

                T0aux[~np.isfinite(T0aux)]=0
                aaux[~np.isfinite(aaux)]=0
                neaux[~np.isfinite(neaux)]=0

                Teproj[i]=trapezoid(T0aux*aaux*neaux*np.cos(theta)**(-2), x=theta)/trapezoid(aaux*neaux*np.cos(theta)**(-2), x=theta)
                aproj[i]=trapezoid(aaux*neaux*np.cos(theta)**(-2), x=theta)/trapezoid(neaux*np.cos(theta)**(-2), x=theta)
        
            else:
                
                Teproj[i]=np.sum(T0*a, axis=-1)/np.sum(a, axis=-1)
                aproj[i]=np.average(a, axis=-1)   

        self.R=R
        self.Teproj=Teproj
        self.aproj=aproj

    def chem_abund(self, vals):

        '''
        This function computes ionic abundances of species using PyNeb.

        Input:
        vals: wavelength (generally the auroral line Id) to determine ionic abundance of a particular specie
        
        '''

        f4861=self.linefitdict['4861_flux']  

        # intensity ratio of strong lines with H-beta

        if vals==4363:
            f5007=self.linefitdict['5007_flux']
            
            O3=pn.Atom('O',3)
            self.OppH=O3.getIonAbundance(int_ratio=100*(f5007)/f4861, tem=self.linefitdict['TeO3'], den=self.linefitdict['neO2'], wave=5007, Hbeta=100)
            
        elif vals==3726:
            f3726=self.linefitdict['3726_flux']
            f3729=self.linefitdict['3729_flux']

            O2=pn.Atom('O',2)
            self.OpH=O2.getIonAbundance(int_ratio=100*(f3726+f3729)/f4861, tem=self.linefitdict['TeN2'], den=self.linefitdict['neO2'], wave=3726, Hbeta=100)

        elif vals==5755:
            f6584=self.linefitdict['6584_flux']

            N2=pn.Atom('N',2)
            self.NpH=N2.getIonAbundance(int_ratio=100*(f6584)/f4861, tem=self.linefitdict['TeN2'], den=self.linefitdict['neO2'], wave=6584, Hbeta=100)

        elif vals==6312:
            f9532=self.linefitdict['9532_flux']

            S3 = pn.Atom('S',3)
            self.SppH = S3.getIonAbundance(int_ratio=100*(f9532)/f4861, tem=self.linefitdict['TeS3'], den=self.linefitdict['neO2'], wave=9532, Hbeta=100)  

        elif vals==6716:
            f6731=self.linefitdict['6731_flux']

            S2 = pn.Atom('S',2)
            self.SpH = S2.getIonAbundance(int_ratio=100*(f6731)/f4861, tem=self.linefitdict['TeN2'], den=self.linefitdict['neO2'], wave=6731, Hbeta=100)  

    def chem_abund_emperical(self, line):

        '''
        This function computes ionic abundances of species using emperical formulae from P. Montero 2017.

        Input:
        line: wavelength (generally the auroral line Id) to determine ionic abundance of a particular specie
        
        '''

        #Emperical formula to compute chemical abund  of [OII]
        if line== 3726:

            I_3726=self.linefitdict['3726_flux'] 
            I_3729=self.linefitdict['3729_flux'] 
            I_4861=self.linefitdict['4861_flux'] 
            Tl=self.linefitdict['TeO2']/1e4 

            self.Abund_O2=np.log10(np.divide((I_3726+I_3729),I_4861))+5.887+np.divide(1.641,Tl)-0.543*np.log10(Tl)+0.000114*self.linefitdict['neO2']
            
        elif line == 4363:
            I_4959=self.linefitdict['4959_flux']
            I_5007=self.linefitdict['5007_flux']
            I_4861=self.linefitdict['4861_flux']
            Th=self.linefitdict['TeO3']/1e4 

            #Emperical formula to compute chemical abund of [OIII]
            self.Abund_O3=np.log10(np.divide((I_4959+I_5007),I_4861))+6.1868+np.divide(1.2491,Th)-0.5816*np.log10(Th) # from Perez-Montero 2017

        elif line == 5755:
            I_5755=self.linefitdict['5755_flux'] 
            I_6584=self.linefitdict['6584_flux'] 
            I_6548=self.linefitdict['6548_flux'] 
            I_4861=self.linefitdict['4861_flux'] 
            Tl=self.linefitdict['TeN2']/1e4 
            
            #Emperical formula to compute chemical abund of [NII]
            self.Abund_N2=np.log10(np.divide((I_6548 + I_6584),I_4861))+6.291+np.divide(0.90221,Tl)-0.5511*np.log10(Tl)

        elif line == 6312:
            I_9069=self.linefitdict['9069_flux']
            I_9532=self.linefitdict['9532_flux']
            I_4861=self.linefitdict['4861_flux']
            Tm=self.linefitdict['TeS3']/1e4 

            #Emperical formula to compute chemical abund of [SIII]
            self.Abund_S3=np.log10(np.divide((I_9069+I_9532),I_4861))+5.983+np.divide(0.661,Tm)-0.527*np.log10(Tm) # from Perez-Montero 2017

        else:
            I_6716=self.linefitdict['6716_flux'] 
            I_6731=self.linefitdict['6731_flux'] 
            I_4861=self.linefitdict['4861_flux'] 
            Tl=self.linefitdict['TeS2']/1e4 

            #Emperical formula to compute chemical abund of [SII]
            self.Abund_S2=np.log10(np.divide((I_6716+I_6731),I_4861))+5.463+np.divide(0.941,Tl)-0.37*np.log10(Tl)

    def Integrated_meas(self):
        
        '''
        This function computes integrated ionic abundances of species.
        '''


        O3=pn.Atom('O',3)
        O2=pn.Atom('O',2)
        N2=pn.Atom('N',2)
        S3=pn.Atom('S',3)
        S2=pn.Atom('S',2)
        ne = 120

        int_f4363 = 0
        int_f5007 = 0

        int_f4861 = 0

        int_f3726 = 0
        int_f3729 = 0
        int_f7319 = 0
        int_f7320 = 0
        int_f7330 = 0
        int_f7331 = 0

        int_f5755 = 0
        int_f6584 = 0

        int_f6312 = 0
        int_f9069 = 0
        int_f9532 = 0

        int_f4069 = 0
        int_f4076 = 0
        int_f6716 = 0
        int_f6731 = 0

        

        for i in range(len(self.fiberdata)):

            #[OIII]
            int_f4363 += self.linefitdict['4363_flux'][i]
            int_f5007 += self.linefitdict['5007_flux'][i]
            int_f4861 += self.linefitdict['4861_flux'][i]
        
            self.int_TO3 = O3.getTemDen(int_f4363/int_f5007, den=ne, wave1=4363, wave2=5007)
            self.int_OppH=O3.getIonAbundance(int_ratio=100*(int_f5007)/int_f4861, tem= self.int_TO3, den= ne, wave=5007, Hbeta=100)

            #[OII]
            int_f3726 += self.linefitdict['3726_flux'][i]
            int_f3729 += self.linefitdict['3729_flux'][i]
            int_f7319 += np.zeros(self.nfib)[i]
            int_f7320 += self.linefitdict['7320_flux'][i]
            int_f7330 += np.zeros(self.nfib)[i]
            int_f7331 += self.linefitdict['7331_flux'][i]
            self.int_TO2 = O2.getTemDen((int_f3726+int_f3729)*2/(int_f7320+int_f7331+int_f7319+int_f7330), den=ne, wave1=3727, wave2=7325)
            self.int_OpH=O2.getIonAbundance(int_ratio=100*(int_f3726)/int_f4861, tem=self.int_TO2, den=ne, wave=3726, Hbeta=100)

            #[NII]
            int_f5755 += self.linefitdict['5755_flux'][i]
            int_f6584 += self.linefitdict['6584_flux'][i]
            self.int_TN2 = N2.getTemDen(int_f5755/int_f6584, den=ne, wave1=5755, wave2=6584)
            self.int_NpH=N2.getIonAbundance(int_ratio=100*(int_f6584)/int_f4861, tem=self.int_TN2, den=ne, wave=6584, Hbeta=100)

            #[SIII]
            int_f6312 += self.linefitdict['6312_flux'][i]
            int_f9069 += self.linefitdict['9069_flux'][i]
            int_f9532 += self.linefitdict['9532_flux'][i]
            self.int_TS3 = S3.getTemDen(int_f6312/int_f9069, den=ne, wave1=6312, wave2=9069)
            self.int_SppH = S3.getIonAbundance(int_ratio=100*(int_f9532+int_f9069)/int_f4861, tem= self.int_TS3, den= ne, wave=9532, Hbeta=100)

            #[SII]
            int_f4069 += self.linefitdict['4069_flux'][i]
            int_f4076 += self.linefitdict['4076_flux'][i]
            int_f6716 += self.linefitdict['6716_flux'][i]
            int_f6731 += self.linefitdict['6731_flux'][i]
            self.int_TS2 = S2.getTemDen((int_f4069+int_f4076)/(int_f6716+int_f6731), den=ne, wave1=4072.5, wave2=6723.5)
            self.int_SpH = S2.getIonAbundance(int_ratio=100*(int_f6731)/int_f4861, tem= self.int_TN2, den= ne, wave=6731, Hbeta=100)

    def chi(self, simname):

        with fits.open('./'+simname+'/'+simname+' diag_Temp_Den.fits') as hdul:
            data = hdul[1].data
            header =hdul[0].header
            
            self.Chi_TeN = np.mean(data['TeN2']) - np.mean(self.vals[1])/ np.mean(self.vals[1])
            self.Chi_TeO = np.mean(data['TeO3']) - np.mean(self.vals[1])/ np.mean(self.vals[1])
            self.Chi_TeS = np.mean(data['TeS3']) - np.mean(self.vals[1])/ np.mean(self.vals[1])

##################################################################### Plotting methods ##############################################
    
    def Te_Abund_plot(self, Te, ion_vals, integrated_te, integrated_abund , chem_abund, testline = np.array(4076), z = 1, log_ion_sun = -3.31, rad1 = 11.8, rad2 = 17.8, label = '[OIII]', outfilename = 'chem_abundO3_vs_R_present.png'): 
        
        '''
        This function will solely created to plot Te and abundance radial variations.

        Input: 
        Te: Te calculated from PyNeb will be given.
        Ion_vals: relative ionic abundance of specie. 
        Integrated_te: Integrated Te computed in Integrated_meas method of specie of interest
        Integrated_abund: Integrated abundance computed in Integrated_meas method of specie of interest
        Chem_abund: Abundance computed from Pyneb
        testline: wavelength of auroral line, not really used in computation, using for if conditions in other methods called inside this method
        z: metallicity
        log_ion_sun: solar abundance for the specie
        rad1: radius at which 50% of the speice has taken place
        rad2: radius at which 50% of the H has taken place
        label: Specified to give label to plots
        outfilename: Defines output file name

        Output:
        Plots of Te and abundance radial variations as well as ADF (computed from equation 12 in Piembert 1967)        
        '''

        distance=16000 * unit.pc 
        r=np.sqrt(self.linefitdict['delta_ra']**2+self.linefitdict['delta_dec']**2) 
        rad=r*distance*np.pi/648000 # converting arcsecs to parsec

        lineid=testline.astype(str)

        self.projectedTe(ion_vals) 
        self.avg_Te(ion_vals)
        self.avg_abundance(ion_vals)        


        good=self.linefitdict[str(lineid)+'_flux']/self.linefitdict[str(lineid)+'_flux_err']>=3  
        bad= self.linefitdict[str(lineid)+'_flux']/self.linefitdict[str(lineid)+'_flux_err']<3    

        fig, (ax2, ax)=plt.subplots(2, 1, figsize=(15,20))

        plt.rcParams.update({'axes.titlesize': 'x-large',
                 'axes.labelsize':'X-large',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'X-large',
                 'xtick.labelsize': 'X-large',
                 'font.size': '12.0',
                 'legend.fontsize':'large'})

        # Electron temperature plots   
        ax2.plot(self.vals[0], self.vals[1], color='red', label='True Te')  
        ax2.plot(self.R, self.Teproj, color='orange', label='On-sky projected Te') 

        ax2.plot(rad[good], Te[good], 'o', color='brown', label='Te '+label+'(goodpix)')  
        ax2.plot(rad[bad], Te[bad], 'o', color='brown',   label='Te '+label+'(badpix)', alpha=0.2) 
        ax2.axhline(y=integrated_te, c='blue', linestyle='--', label='Integrated Te '+label+' measurement')   
        ax2.axhline(y=self.avgTe, c='black', linestyle='--', label='Average Te') 

        ax2.set_ylim(self.avgTe -3000, self.avgTe+4000)
        ax2.set_ylabel('Te ('+label+')(K)') 
        ax2.legend(loc='upper left')     

        Y=12+np.log10(integrated_abund)

        # O++ ionic abundance  
        Zmodel= z                                           # cloudy model abundance relative to solar  
        logOHsun= log_ion_sun                               # solar abundance patter from GASS (Grevesse et al 2010)  
        logOHmodel = logOHsun + np.log10(Zmodel)            # total Oxygen elemental abundance in the model  
        logOppHmodel = logOHmodel+np.log10(ion_vals)        # ionic abundance of the ion in the model  
        logOppHproj = logOHmodel+np.log10(self.aproj)       # projected ionic abundance of the ion in the model  


        ax.plot(self.vals[0], 12+logOppHmodel, color='red', label='Model ionic abund')  
        ax.plot(self.R, 12+logOppHproj, color='orange', label='On sky projected ionic abund')  
        ax.axhline(Y, c='blue', linestyle='--', label='Integrated '+label+' abundance measurement')    
        #ax.axhline(y=self.avg_abund, c='black', linestyle='--', label='Average abund')   

        ax.plot(rad[good], 12+np.log10(chem_abund)[good], 'o', color='brown', label='Ionic abundance '+label+'(goodpix)')  
        ax.plot(rad[bad],  12+np.log10(chem_abund)[bad], 'o',  color='brown', label='Ionic abundance '+label+'(badpix)', alpha=0.2)  

        ax.set_ylim(Y-3, Y+3) 
        ax2.set_title('Chemical abundance correlation with electron temperture for ' +label)
        ax.set_xlabel('Radius (pc)')  
        ax.set_ylabel(r'$12+ \log({'+label+'} / {HII})$') 
        ax.legend(loc='best')     
       

        ax2.set_position([0.125, 0.5, 0.775, 0.3])   
        ax.set_position([0.125, 0.2, 0.775, 0.3])  
        

        plotdir=self.datadir+self.simname+'/'+'Only_Te_Abundance_plots_'+self.simname +'/'
        if (not os.path.isdir(plotdir)):
            os.mkdir(plotdir) 

        te_adf_T0 = np.sum(self.vals[0]*(self.vals[1] - self.avgTe)**2 *self.vals[2]*ion_vals)/np.sum(self.vals[0]*self.avgTe**2 *self.vals[2]*ion_vals)
        print('ADF_T0 '+label+':',te_adf_T0)

        #te_adf_projT = np.sum(np.nanmean(self.Teproj - self.avgTe)**2 *120*ion_vals)/np.sum(np.nanmean(self.avgTe**2 *self.vals[2]*ion_vals))
        #print('ADF_projTe '+label+':',te_adf_projT)

        plt.savefig('/'+plotdir+'/'+outfilename, dpi=400, bbox_inches = 'tight')
        #plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14/Bubble_v2_5e-14_snbinned/Bubble_v2_5e-14_snbinned_plotprofile/snbin_TeO3_chem_abundO3_vs_R.png', dpi=300)  
        #plt.show() 


        '''
        #relative ionic abundance plots
        fig, (ax1, ax2, ax)=plt.subplots(3, 1, figsize=(15,15)) 

        plt.rcParams.update({'axes.titlesize': 'x-large',
                 'axes.labelsize':'X-large',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'X-large',
                 'xtick.labelsize': 'X-large',
                 'font.size': '12.0',
                 'legend.fontsize':'large'}) 

        ax1.plot(self.vals[0], ion_vals, color='red', label='relative ionic abund '+ label)  
        ax1.plot(self.R, self.aproj, color='orange', label='Projected ionic abund ' + label) 
        ax1.axvline(x= rad1, c='red', linestyle='--', label='50% ionization of O to '+ label)  
        ax1.axvline(x= rad2, c='black', label='50% ionization of H')
        ax1.axhline(y=np.average(ion_vals), c='cyan', linestyle='--', label='avg_rel_ion_abund_'+ label)
        ax1.legend(loc='upper left')  
        ax1.set_ylabel('relative ionic abund '+ label) 
        ax1.set_title('Chemical abundance correlation with electron temperture for ' +label)


        # Electron temperature plots   
        ax2.plot(self.vals[0], self.vals[1], color='red', label='True Te')  
        ax2.plot(self.R, self.Teproj, color='orange', label='On Sky_Projected_Te') 

        ax2.plot(rad[good], Te[good], 'o', color='green', label='Te '+label+'goodpix')  
        ax2.plot(rad[bad], Te[bad], 'o', color='green',   label='Te '+label+'badpix', alpha=0.2) 

        ax2.axhline(y=integrated_te, c='cyan', linestyle='--', label='Integrated Te '+label+'measurement')   
        ax2.axhline(y=self.avgTe, c='blue', linestyle='--', label='Average Te')   
        
        ax2.set_ylim(self.avgTe -2000, self.avgTe+3000)
        ax2.set_ylabel('Te '+label+'(K)')  
        ax2.legend(loc='upper left')     


        # O++ ionic abundance  
        Zmodel= z                                           # cloudy model abundance relative to solar  
        logOHsun= log_ion_sun                               # solar abundance patter from GASS (Grevesse et al 2010)  
        logOHmodel = logOHsun + np.log10(Zmodel)            # total Oxygen elemental abundance in the model  
        logOppHmodel = logOHmodel+np.log10(ion_vals)        # ionic abundance of the ion in the model  
        logOppHproj = logOHmodel+np.log10(self.aproj)       # projected ionic abundance of the ion in the model  

        ax.plot(self.vals[0], 12+logOppHmodel, color='red', label='model ionic abund')  
        ax.plot(self.R, 12+logOppHproj, color='orange', label='Projected model ionic abund')  

        ax.axhline(y=12+np.log10(integrated_abund), c='cyan', linestyle='--', label='Integrated '+label+'Abundance measurement')    

        ax.plot(rad[good], chem_abund_emp[good], 'o', color='blue', markersize='10', label='Perez-Montero 2017 (goodpix)')  
        ax.plot(rad[bad],  chem_abund_emp[bad], 'o', color='blue', markersize='10', label='Perez-Montero 2017 (badpix)', alpha=0.2)  
        ax.plot(rad[good], 12+np.log10(chem_abund)[good], 'o', color='red', label='This work goodpix')  
        ax.plot(rad[bad],  12+np.log10(chem_abund)[bad], 'o', color='red', label='This work badpix', alpha=0.2)  

        ax.set_ylim(4.5, 9) 

        ax.set_xlabel('Radius (pc)')  
        ax.set_ylabel(r'$12+\log({'+label+'}/{HII})$') 
        ax.legend(loc='lower left')     


        ax1.set_position([0.125, 0.8, 0.775, 0.26])  # [left, bottom, width, height]  
        ax2.set_position([0.125, 0.5, 0.775, 0.3])   
        ax.set_position([0.125, 0.2, 0.775, 0.3])  


        plotdir=self.datadir+self.simname+'/'+'Abundance_plots_'+self.simname +'/'
        if (not os.path.isdir(plotdir)):
            os.mkdir(plotdir) 

        plt.savefig('/'+plotdir+'/'+outfilename, dpi=300, bbox_inches = 'tight')
        #plt.savefig('/home/amrita/LVM/lvmnebular/Bubble_v2_5e-14/Bubble_v2_5e-14_snbinned/Bubble_v2_5e-14_snbinned_plotprofile/snbin_TeO3_chem_abundO3_vs_R.png', dpi=300)  
        #plt.show()  

        te_adf = np.sum(np.nanmean(rad*(Te - self.avgTe)**2 *120*self.linefitdict['neO2']))/np.sum(np.nanmean(rad*self.avgTe**2 *120*self.linefitdict['neO2']))
        print('ADF '+label+':',te_adf)
        '''


    def plotmap(self, z, min, max, table, nlevels=40, title='line_map', output='line_map', radbin=False, vorbin=False,  snbin=False, pertsim=False):

            '''
            This function will plot 1 D maps of Te, ne and error on Te and ne.

            Input:
            z: Te, ne and errors from linefitdict table (1 D array)
            min: minimum value of z (float)
            max: maximum value of z (float)
            nlevels: no. of levels in maps (int)
            title: Title of maps (str)
            output:Output names of the plot maps. (str)

            Output: 
            Plot out maps. 

            '''
            if vorbin:
                plotdir=self.simname+'/'+self.simname+'_vorbinned/'+self.simname+'_vorbinned_plotmap/'
                if (not os.path.isdir(plotdir)):
                    os.mkdir(plotdir)

            elif radbin:
                plotdir=self.simname+'/'+self.simname+'_radbinned/'+self.simname+'_radbinned_plotmap/'
                if (not os.path.isdir(plotdir)):
                    os.mkdir(plotdir) 

            elif snbin:
                plotdir=self.simname+'/'+self.simname+'_snbinned/'+self.simname+'_snbinned_plotmap/'
                if (not os.path.isdir(plotdir)):
                    os.mkdir(plotdir)    

            else:

                plotdir=self.datadir+self.simname+'/'+self.simname+'_plotpmap/'
                if not (os.path.isdir(plotdir)):
                    os.mkdir(plotdir)
        

            sel=np.isfinite(z)

            plt.rcParams.update({'axes.titlesize': 'x-large',
                 'axes.labelsize':'X-large',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'X-large',
                 'xtick.labelsize': 'X-large',
                 'font.size': '12.0',
                 'legend.fontsize':'large'})

            newtable=Table.read(table)
            fig, ax = plt.subplots(figsize=(8,5))
            triang = tri.Triangulation(self.linefitdict['delta_ra'][sel].flatten(), self.linefitdict['delta_dec'][sel].flatten()) 
            c = ax.tricontourf(triang, z[sel], levels=np.linspace(min, max, nlevels))    
            plt.colorbar(c) 
            ax.set_title(title)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.axis('equal')
            plt.savefig(plotdir+'/'+output+'.png', dpi=200)


    def plotprofile(self, z, min, max, title='line_map', output='line_map', radbin=False, vorbin=False, snbin=False, pertsim=False):

        '''
        This function will plot 2 D radial profiles of Te and ne.
        
        Input:
        z: Te, ne and errors from linefitdict table (1 D array)
        min: minimum value of z (float)
        max: maximum value of z (float)
        nlevels: no. of levels in maps (int)
        title: Title of maps (str)
        output:Output names of the plot maps. (str)

        Returns: 
        Plot out radial profiles of Te and ne.  
        
        '''
        
        if vorbin:
            plotdir=self.simname+'/'+self.simname+'_vorbinned/'+self.simname+'_vorbinned_plotprofile/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)

        elif radbin:
            plotdir=self.simname+'/'+self.simname+'_radbinned/'+self.simname+'_radbinned_plotprofile/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)

        elif snbin:
            plotdir=self.simname+'/'+self.simname+'_snbinned/'+self.simname+'_snbinned_plotprofile/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)    

        else:

            plotdir=self.datadir+self.simname+'/'+self.simname+'_plotprofile/'
            if not (os.path.isdir(plotdir)):
                os.mkdir(plotdir)
        
        sel=np.isfinite(z)  

        plt.rcParams.update({'axes.titlesize': 'x-large',
                 'axes.labelsize':'X-large',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'X-large',
                 'xtick.labelsize': 'X-large',
                 'font.size': '12.0',
                 'legend.fontsize':'large'})
            
        distance=16000 #self.pc

        r=np.sqrt(self.linefitdict['delta_ra']**2+self.linefitdict['delta_dec']**2)
        rad=r[sel]*distance*np.pi/648000 # converting arcsecs to parsec
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(rad, z[sel], '.', label='data')

        ax.set_ylim(min, max)
        ax.set_ylabel(title)
        ax.set_xlabel('Radius (parsec)')
        ax.legend()
        plt.savefig(plotdir+'/'+output+'_rad.png', dpi=200)      


    def overplotprofile(self, z, val1, val2, min, max, x, title='line_map', output='line_map', radbin=False, vorbin=False, snbin=False, pertsim=False):

        '''
        This function will plot 2 D radial profiles of Te and ne.
        
        Input:
        z: Te, ne and errors from linefitdict table (1 D array)
        val1:True electron temp or density provided
        val2:Ionic abundance for each species 
        min: minimum value of z (float)
        max: maximum value of z (float)
        x=50% ionization position
        n: no. of steps in theta (int)
        title: Title of maps (str)
        output:Output names of the plot maps. (str)

        Output: 
        Plot out radial profiles of Te and ne.  
        
        '''
               
        self.projectedTe(val2)

        if vorbin:
            plotdir=self.simname+'/'+self.simname+'_vorbinned/'+self.simname+'_vorbinned_overplotprofile/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)

        elif radbin:
            plotdir=self.simname+'/'+self.simname+'_radbinned/'+self.simname+'_radbinned_overplotprofile/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir) 

        elif snbin:
            plotdir=self.simname+'/'+self.simname+'_snbinned/'+self.simname+'_snbinned_overplotprofile/'
            if (not os.path.isdir(plotdir)):
                os.mkdir(plotdir)    

        else:

            plotdir=self.datadir+self.simname+'/'+self.simname+'_overplotprofile/'
            if not (os.path.isdir(plotdir)):
                os.mkdir(plotdir)
        
        sel=np.isfinite(z) 

        plt.rcParams.update({'axes.titlesize': 'x-large',
                 'axes.labelsize':'X-large',
                 'axes.linewidth':     '1.8' ,
                 'ytick.labelsize': 'X-large',
                 'xtick.labelsize': 'X-large',
                 'font.size': '12.0',
                 'legend.fontsize':'large'})

        distance=16000 #self.pc
        r=np.sqrt(self.linefitdict['delta_ra']**2+self.linefitdict['delta_dec']**2)
        rad=r[sel]*distance*np.pi/648000 # converting arcsecs to parsec

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12,8))
        ax.plot(rad, z[sel], '.', label='data') #Te from Pyneb
        ax.plot(self.vals[0], val1, c='grey', label='True profile') #true Te from model
        ax.plot(self.R, self.Teproj, color='orange', label='Projected Te') #Projected Te 
        #ax1.plot(self.vals[0], val2, color='green', label='Ionic abundance') #Ionic abundance
        
        ax.axvline(x, c='red', linestyle='--', label='50% ionization')  # a constant vertical line with 50% or 90% of ionization; try to make it general

        ax.set_ylim(min, max)
        ax.set_ylabel(title)
        ax.set_xlabel('Radius (parsec)')
        ax.legend()
        plt.savefig(plotdir+'/'+output+'_rad.png', dpi=300)      
 

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

'''
def double_gaussian(wave, flux1, mean1, sd1, flux2, mean2, sd2):
        
    #This function evaluates a 1D Gaussian
    #
    #Input:
    #wave=wavelength array(1D numpy array)
    #flux=line flux (float) #a
    #mean=line centroid (float)
    #sd=1 sigma line width (float) 

    #returns:
    #gaussian profile (1D numpy array)
   
    gaussian1 = a1 / (np.sqrt(2 * np.pi) * sd1) * np.exp(-(wave - mean1) ** 2 / (2 * sd1 ** 2))
    gaussian2 = a2 / (np.sqrt(2 * np.pi) * sd2) * np.exp(-(wave - mean2) ** 2 / (2 * sd2 ** 2))
    return gaussian1 + gaussian2

    #return ((flux1/(np.sqrt(2*np.pi)*sd1)*np.exp((wave1-mean1)**2/(-2*sd1**2))+ (flux2/(np.sqrt(2*np.pi)*sd2)*np.exp((wave2-mean2)**2/(-2*sd2**2)))
'''

def error_func(wave, gaussian, popt, pcov, e=1e-7):
    
    '''
    This function produce error to all emission line in lines.

    Input:
    wave: Wavelength array of spectra in simulations (1D numpy array)
    gaussian: To fit gaussian profile to errors on emission lines flux
    popt:Optimal values for the parameters (1D numpy array)
    pcov: Estimated covariance of popt (3D matrix)

    Output:
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

def fit_gauss(wave, spectrum, error, lwave, dwave=5, plot=True, plotout='linefit'):
    '''
    This function produce gaussion fit to all emission line in lines.

    Input:
    wave: wavelength array of spectra in simulations (1D numpy array)
    spectrum: flux array(1D numpy array)
    error: flux error array(1D numpy array)
    lwave: initial approximation (1D numpy array)
    
    Returns: 
    optimal and covarience parameters, popt and pcov.
    
    ''' 
    sel=(wave>lwave-dwave/2)*(wave<lwave+dwave/2)

    p0=(np.abs(spectrum[sel].max()), lwave, 0.7)

    try: 
        popt, pcov=curve_fit(gaussian, wave[sel], spectrum[sel], sigma=error[sel], p0=p0, absolute_sigma=True, bounds=((0,lwave-1.0,0.3),(np.inf,lwave+1.0,2.0)))
    except RuntimeError: 
        popt=np.array([-99, p0[1], p0[2]])
        pcov=np.zeros((3,3))
        pcov[0,0]=np.sum(error[sel]**2)

    if plot:
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
########################################################################################################################################

################################################### Function used to bin spectra in simulations ########################################

def binrad_spectra(rmin, rmax, radius, spectra, errors):

    sel = np.where((radius >= rmin) & (radius < rmax))[0]

    newflux = np.zeros((len(sel), spectra.shape[1]))
    newerr = np.zeros((len(sel), spectra.shape[1]))

    for i, id in enumerate(sel):
        newflux[i] = spectra[id]
        newerr[i] = errors[id]

    newflux = newflux.sum(axis=0)
    newerr = np.sqrt(np.sum(newerr**2, axis=0))

    return newflux, newerr, len(sel)

################################################# Functions used in perturbing the simulation ###########################################

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

















