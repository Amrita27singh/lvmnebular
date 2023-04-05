
"""
Created on Thu Aug  4 19:08:35 2022

@author: amrita
"""
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import LVM160
from lvmdatasimulator.instrument import LinearSpectrograph
from lvmdatasimulator.simulator import Simulator
from lvmdatasimulator.fibers import FiberBundle
import astropy.units as u
from lvmdatasimulator.run import run_simulator_1d
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch, LinearStretch
from matplotlib import pyplot as plt
import sys
from astropy.table import Table

tel = LVM160()
spec = LinearSpectrograph()
bundle = FiberBundle(bundle_name='full', nrings= 8, angle=0, custom_fibers=None)


#specifing parameters
ra = 90.5625 # u.degree
dec = 4.998333 # u.degree
fov_size = 10 # u.arcmin
fov_pixel = 10 # u.arcsec
distance = 16.0 * u.kpc
sys_vel = 20 #* u.km / u.s
turbulent_sigma = 15 #* u.km / u.s
preserve_kinematics = False

name='Bubble_v2_1e-11'
unit_ra = u.degree
unit_dec = u.degree
unit_size = u.arcmin
unit_pixel = u.arcsec

#defining FOV 
my_lvmfield = LVMField(ra=ra, dec=dec, size=fov_size, pxsize=fov_pixel, 
                       unit_ra=unit_ra, unit_dec=unit_dec, unit_size=unit_size, unit_pxsize=unit_pixel, name=name)



bubble = [{'type': 'Bubble', 'max_brightness':1e-11, 'thickness': 0.8, 'radius': 18, 'expansion_velocity': 10, 'sys_velocity': sys_vel, 
          'distance': distance,
          'model_params': {'Z': 1., 'Teff': 40000, 'nH': 100, 'qH': 50.0, 'Geometry': 'Shell'},
          'model_type': 'cloudy', 'offset_RA':0, 'offset_DEC':0}]
Circle = [{'type': 'Circle', 'max_brightness': 1e-17, 'radius': 18, 'sys_velocity': sys_vel, 
          'distance': distance,
          'model_params': {'Z': 1., 'Teff': 40000, 'nH': 100, 'qH': 50.0, 'Geometry': 'Shell'},
          'model_type': 'cloudy', 'offset_RA':0, 'offset_DEC':0}]

my_lvmfield.add_nebulae(bubble, save_nebulae='testneb_tutorial3_ex1.fits')
my_lvmfield.show(percentile=98, fibers=bundle.fibers)

#Observation
exptimes=[900, 3600, 10800]
days_moon=7
obs = Observation(ra=ra, dec=dec, unit_ra=u.deg, unit_dec=u.deg, exptimes=exptimes,
                  airmass=1.2, days_moon=days_moon)
print(obs.time)
print(obs.airmass)
print(obs.days_moon)

#starting simulator
sim = Simulator(my_lvmfield, obs, spec, bundle, tel)
sim.simulate_observations()
sim.save_outputs()
sim.save_output_maps(wavelength_ranges=[6560, 6567])


fig = plt.figure(figsize=(18,6))
fiber_id = 0
with fits.open(f"{name}/outputs/{name}_linear_full_{exptimes[0]}_flux.fits") as hdu:
    # hdu.info()
    wave = hdu['Wave'].data
    flux_900_exp = hdu['TOTAL'].data[fiber_id]
    flux_900_exp_nosky = hdu['TARGET'].data[fiber_id]
#    snr= hdu['SNR'].data[fiber_id]
with fits.open(f"{name}/outputs/{name}_linear_full_{exptimes[1]}_flux.fits") as hdu:
    flux_3600_exp = hdu['TOTAL'].data[fiber_id]
    flux_3600_exp_nosky = hdu['TARGET'].data[fiber_id]
 
with fits.open(f"{name}/outputs/{name}_linear_full_{exptimes[2]}_flux.fits") as hdu:
    flux_10800_exp = hdu['TOTAL'].data[fiber_id]
    flux_10800_exp_nosky = hdu['TARGET'].data[fiber_id]
   
    
ax = plt.subplot(121)
#plt.plot(wave, snr, linewidth=1, label=f"Texp = {exptimes[0]}s")
plt.plot(wave, flux_900_exp, linewidth=1, label=f"Texp = {exptimes[0]}s")
plt.plot(wave, flux_3600_exp, linewidth=1, label=f'Texp = {exptimes[1]}s')
plt.plot(wave, flux_10800_exp, linewidth=1, label=f'Texp = {exptimes[2]}s')
plt.legend()
plt.xlabel("Wavelength, $\AA$",fontsize=14)
plt.ylabel("Intensity, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$",fontsize=14)
#plt.ylim(0,1e-15)
plt.xlim(3000,9000)
plt.title(f"Fiber_id={fiber_id} (with sky)", fontsize=16)

ax = plt.subplot(122)
plt.plot(wave, flux_900_exp_nosky, linewidth=1, label=f"Texp = {exptimes[0]}s")
plt.plot(wave, flux_3600_exp_nosky, linewidth=1, label=f'Texp = {exptimes[1]}s')
plt.plot(wave, flux_10800_exp_nosky, linewidth=1, label=f'Texp = {exptimes[2]}s')
plt.legend()
plt.xlabel("Wavelength, $\AA$",fontsize=14)
plt.ylabel("Intensity, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$",fontsize=14)
# plt.ylim(0,1e-15)
plt.xlim(3000,9000)
plt.title(f"Fiber_id={fiber_id} (sky subtracted)", fontsize=16);
plt.show()


#filename = '{name}/outputs/{name}_linear_full_input.fits'
with fits.open(f"{name}/outputs/{name}_linear_full_input.fits") as hdu:
    # hdu.info()
    wave = hdu['WAVE'].data
    fiberid = Table.read(hdu['FIBERID'])
    flux = hdu['FLUX'].data
    
mask = fiberid['id'] == 0
spectrum = flux[mask][0]  # select only fiber n. 200
plt.plot(wave, spectrum)
plt.savefig('spectrum')
plt.show()

   
'''
fig = plt.figure(figsize=(18,6))
ax=plt.subplot(111)
filename = '{name}_linear_full_3600_flux.fits'
with fits.open(f"{name}/outputs/{filename}") as hdu:
    wave = hdu['WAVE'].data
    select_wl = (wave > 6000) & (wave < 7000)
    norm = ImageNormalize(hdu['TARGET'].data[:, select_wl], stretch=AsinhStretch(), interval=PercentileInterval(99.9))
    ax.imshow(hdu['TARGET'].data[:, select_wl], origin='lower', interpolation='nearest', cmap=plt.cm.Oranges,
              norm=norm, extent=(wave[select_wl][0], wave[select_wl][-1], 0, hdu['TARGET'].data.shape[0]), aspect=0.3)
    ax.set_xlabel('Wavelength, A')
    ax.set_ylabel('Fiber ID')
    ax.set_title(filename)

plt.show()

'''




