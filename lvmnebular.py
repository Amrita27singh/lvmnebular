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

        
        self.ra=None
        self.dec=None
        self.lineid=None
        self.lineflux=None
        self.linecenter=None
        self.linewidth=None

        self.simfile = None
        self.linefitfile= None
        self.dim=None

        self.newvalue=None

    
    def loadsim(self, simname, exptime, datadir='/home/amrita/LVM/'):

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

        self.nfib=len(self.fiberdata)

    
    def fitlines(self, sys_vel=0, lines0= np.array([6563, 6583]) , bin=False, pertsim=False):
        '''
        This function fits each line in self.lineid in the spectrum of each spaxel and measures fluxes, linewidthsm and line centers

        Input:

        sys_vel (optional): first guess of systemic velocity of emission lines, default is sys_vel=0 (float)
        lines (optional): rest-frame wavelength of emission lines to be fitted. Default is just Ha and NII6583 (numpy array of floats)
        bin: if True work on binned simulation spectra, if False (default) fit native simulation spectra (boolean)
        pertsim: if True work on perturbed simulation spectra, if False (default) fit native simulation spectra (boolean)
        
        

        Output:
        
        '''
        self.lineid=lines0.astype(str)
        print('Fitting Emmission Lines:', self.lineid)
        if (self.nfib == None):
            raise Exception('Simulation has not been loaded. Run loadsim first.')
        
        self.linefitfile=self.datadir+self.simname+'/'+self.simname+'_linefits.fits'
        
        plotdir=self.datadir+self.simname+'/linefitplots/'
        if (not os.path.isdir(plotdir)):
            os.mkdir(plotdir)

        wave = self.wave
        fiberid = self.fiberdata
        flux = self.flux
        err = self.err
            
        c=299792.458 # speed of light in km/s
        lines=lines0*(1+sys_vel /c)

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
            

        for i in range(self.nfib):
            mask = self.fiberdata['id'] == i
            spectrum = flux[mask].flatten()  # select only fiber n. 200
            error = err[mask].flatten()
            self.linefitdict['fiber_id'].append(fiberid[mask]['id'])
            self.linefitdict['delta_ra'].append(fiberid[mask]['x'].flatten())
            self.linefitdict['delta_dec'].append(fiberid[mask]['y'].flatten())       
    
            for j,line in enumerate(lines):
                print("Fitting Line:", line)
                plot=False
                plotout='junk'
                list=lines0
            
                if lines0[j] in list:
                    plot=False               
                    plotout=plotdir+str(fiberid['id'][i])+'_'+str(lines0[j])
                popt, pcov = fit_gauss(wave, flux[i,:], err[i,:], line, plot=plot, plotout=plotout)
               
                   
                self.linefitdict[str(lines0[j])+'_flux'].append(popt[0])
                self.linefitdict[str(lines0[j])+'_flux_err'].append(np.sqrt(pcov[0, 0]))
                self.linefitdict[str(lines0[j])+'_lambda'].append(popt[1])
                self.linefitdict[str(lines0[j])+'_lambda_err'].append(np.sqrt(pcov[1, 1]))
                self.linefitdict[str(lines0[j])+'_sigma'].append(popt[2])
                self.linefitdict[str(lines0[j])+'_sigma_err'].append(np.sqrt(pcov[2, 2]))
            
                #print("fiber: ", i)
                #print(popt)
                #print(np.sqrt(pcov[0, 0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2, 2]), "\n")
                #
                #for key, value in output_dict.items():
                   # print(key, ' \n ', value)
          
    
        t=Table(self.linefitdict, names=self.linefitdict.keys())
        t.write(self.linefitfile, overwrite=True)        

    def pyneb(self, niter=10, lines0=np.array([4363, 5007, 5755, 6583]), bin=False, pertsim=False):

        '''
        This function will use the line fluxes to calculate the Te, Ne radial temp variation, etc. for each line in self.lineid 
        and then calculate and store error on the Te nad Ne running MC.
        
        Input:

        bin: if True work on binned simulation spectra, if False (default) run pyneb and MC on native simulation spectra (boolean)
        pertsim: if True work on perturbed simulation spectra, if False (default) run pyneb and MC on native simulation spectra (boolean)
        niter: number of iterations to run MC in order to get errors on temperature, density, etc., default is niter=10 (integer)
        lines0: rest-frame wavelength of emission lines required to compute lineflux. 
        Default is just NII and OIII auroral and strong lines(numpy array of floats)

        Output:
        Run pyneb package on emission line ratios to measure temperature of relevant ionic species.
        '''

        self.lineid = lines0.astype(str)

        if (self.nfib is None):
            RuntimeWarning('Undefined number of fibers. Probably you have not run fitlines yet', RuntimeWarning)

        table = Table.read(self.linefitfile)
        #print(table)

        T_out = np.zeros((self.nfib, niter))  # output list of Temperature
        N_out = np.zeros((self.nfib, niter))  # output list of Density

        for j in lines0:
            for i in range(niter):
                newvalue_name = f'newvalue_{self.lineid[j]}'  # creating a variable name for the ratio
                newvalues = new_measurements(table[str(self.lineid[j]) + '_flux']), table[str(self.lineid[j]) + '_flux_err']

                if newvalue_name in self.ratios:
                    self.newvalue[newvalue_name][j].append(newvalues) # appending new measurements to existing list for this line ID

                else:
                    self.newvalue[newvalue_name] = {j: [newvalues]} # creating a new dictionary entry for this ratio name and line ID

        print(self.newvalue[newvalue_name][j])   
        
    
    '''
    def pyneb(self, niter = 10, lines0= np.array([4363, 5007, 5755, 6583]), bin=False, pertsim=False):
    
        This function will use the line fluxes to calculate the Te, Ne radial temp variation, etc. for each line in self.lineid 
        and then calculate and store error on the Te nad Ne running MC.
        
        Input:

        bin: if True work on binned simulation spectra, if False (default) run pyneb and MC on native simulation spectra (boolean)
        pertsim: if True work on perturbed simulation spectra, if False (default) run pyneb and MC on native simulation spectra (boolean)
        niter: number of iterations to run MC in order to get errors on temperature, density, etc., default is niter=10 (integer)
        lines0: rest-frame wavelength of emission lines required to compute lineflux. 
        Default is just NII and OIII auroral and strong lines(numpy array of floats)

        Output:
        Run pyneb package on emission line ratios to measure temperature of relevant ionic species.

        
        self.lineid=lines0.astype(str)
        if (self.nfib==None):
            RuntimeWarning('Undefined number of fibers. Probably you have not run fitlines yet')
    
        table=Table.read(self.linefitfile)
        T_out = np.zeros((self.nfib,niter))  # output list of Temperature
        N_out = np.zeros((self.nfib,niter))  # output list of Density

############################ method 1 ######################################################
          for j in lines0:
            for i in range(niter):
                ratio_name = f'ratio_{lines0[j]}'  # a variable name for the ratio
                ratio_measurements = new_measurements(table[str(self.lineid[j]) + '_flux']), table[str(self.lineid[j]) + '_flux_err']
            if ratio_name in self.ratios:
                self.ratios[ratio_name].append(ratio_measurements) # appending new measurements to existing list

            else:
                self.ratios[ratio_name] = [ratio_measurements] # create a new list for this ratio name

        print(self.ratios[ratio_name])

############################ method 2 ######################################################        
        for j in lines0:
            for i in range(niter):
                self.ratio_ + str(j) = new_measurements(table[str(self.lineid[j])+'_flux']), table[str(self.lineid[j])+'_flux_err']
        print(self.ratio[j])


            ############################### Example #########################################
                tmp_6584 = new_measurements(table['6584_flux'], table['6584_flux_err']   
        
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


    
        for i in range(niter):
            tmp_linecenter = new_measurements(table[self.linecenter_flux], table[self.linecenter_flux_err])
        
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



################################################ Functions used in above methods ###############################################################

################################################ Functions used in fitlines method #############################################################

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

def fit_gauss(wave, spectrum, error, lwave, dwave=4, plot=False, plotout='linefit'):
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
        plt.ylabel(r'Flux (erg/cm^2 s \AA)')
        plt.xlabel(r'wavelength(\AA)')

        
        ax.legend()
        plt.savefig(plotout+'.png')
    return popt, pcov
##################################################################################################################################################


################################################### Function used to run MC simulations ##########################################################

def new_measurements(value, errs):
    '''
    This function will be used to run monte carlo simulation on line fluxes.
    Input:
    value:line_flux
    errs: line_fluxerr
    
    Output:
    random values (1D numpy array)
    
    '''
    delta = np.random.randn(len(errs)) * errs
    new_values = np.max([value + delta, np.zeros(len(errs))], axis=0)
    return new_values

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



















