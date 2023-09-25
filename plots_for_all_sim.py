import lvmnebular
import numpy as np
import matplotlib.pyplot as plt
import os

u=lvmnebular.simulation()
simname = np.array(['Bubble_v2_1e-8', 'Bubble_v2_5e-14', 'Bubble_v2_1e-8_z_0.2', 'Bubble_v2_5e-14_z_0.2', 'Bubble_v2_1e-8_z_0.4', 'Bubble_v2_5e-14_z_0.4', 'Bubble_v2_1e-8_z_0.6', 
'Bubble_v2_5e-14_z_0.6', 'Bubble_v2_1e-8_z_0.8', 'Bubble_v2_5e-14_z_0.8', 'Perturbed_cube_Bubble_v2_1e-8', 'Perturbed_cube_Bubble_v2_5e-14'])

#simname = np.array(['Bubble_v2_1e-8', 'Bubble_v2_5e-14'])
#simname = np.array(['Bubble_v2_1e-8_z_0.2', 'Bubble_v2_5e-14_z_0.2'])
#simname = np.array(['Bubble_v2_1e-8_z_0.4',  'Bubble_v2_5e-14_z_0.4'])  
#simname = np.array(['Bubble_v2_1e-8_z_0.6', 'Bubble_v2_5e-14_z_0.6'])
#simname = np.array(['Bubble_v2_1e-8_z_0.8', 'Bubble_v2_5e-14_z_0.8'])


simname=simname.astype(str)

for i in simname:

    u.loadsim(i,900)          
    
    # fitting lines 
    u.fitlines(sys_vel=20, lines0= np.array([9532, 9069, 7319, 7320, 7330, 7331, 6731, 6716, 6584, 6563, 6548, 6312, 5755, 5007, 4959, 4861, 4363, 4069, 4076, 3970, 3729, 3726]), radbin=False, vorbin=False, rbinmax=260, drbin=15, loadfile=True, plot=False) #native sim   

    # running pyneb
    u.runpyneb(niter=6, pertsim=False)

    #def Te_Abund_plot(self, Te = self.linefitdict['TeO3], ion_vals = self.vals[6], integrated_te = self.int_TO3, integrated_abund = self.int_OppH,chem_abund, chem_abund_emp, testline = 4363, z = 1, log_ion_sun = -3.31, rad1 = 11.8, rad2 = 17.8, label = '[OIII]', outfilename = 'chem_abundO3_vs_R_present.png')
    #Radius:0, 'Te':1, 'ne':2, 'H+':3, 'O0':4, 'O+':5, 'O++':6, 'N0':7, 'N+':8, 'N++':9, 'S0':10, 'S+':11, 'S++:12
    
    #rad1 = ([OIII]:11.2, [OII]:11.2, [NII]:13.8, [SIII]:17.6, [SII]: 17.6) --------      z=0.5
    #rad1 = ([OIII]:9.78, [OII]:9.78, [NII]:13.36, [SIII]:17.35, [SII]: 17.35) --------   z=1
    #rad1 = ([OIII]:11.92, [OII]:11.92, [NII]:14.15, [SIII]:17.5, [SII]: 17.5) --------   z=0.2
    #rad1 = ([OIII]:11.17, [OII]:11.17, [NII]:13.83, [SIII]:17.5, [SII]: 17.5) --------   z=0.4
    #rad1 = ([OIII]:10.75, [OII]:10.75, [NII]:13.65, [SIII]:17.41, [SII]: 17.41) -------- z=0.6
    #rad1 = ([OIII]:10.26, [OII]:10.26, [NII]:13.44, [SIII]:17.33, [SII]: 17.33) -------- z=0.8

    #log_ion_sun = -3.31 ----[O]
    #log_ion_sun = -4.17 ----[N]
    #log_ion_sun = -4.88 ----[S]
    
    #s = i.split('_')
    #if len(s)==5:
    #    z1= float(s[4])
#
    #else:
    #     z1=1
#
    ##[OII]
    #line = 3726    
    #u.Integrated_meas()
    #u.chem_abund(line)
    #u.chem_abund_emperical(line)
    #u.Te_Abund_plot(u.linefitdict['TeO2'], u.vals[5], u.int_TO2, u.int_OpH, u.OpH, u.Abund_O2,  testline = np.array(line), z = z1, log_ion_sun = -3.31, rad1 = 11.92, rad2 = 18, label = '[OII]', outfilename = 'O2_Te_chem_abund_vs_R_present.png')
    #
    ##[OIII]
    #line = 4363  
    #u.Integrated_meas()
    #u.chem_abund(line)
    #u.chem_abund_emperical(line)
    #u.Te_Abund_plot(u.linefitdict['TeO3'], u.vals[6], u.int_TO3, u.int_OppH, u.OppH, u.Abund_O3,  testline = np.array(line), z = z1, log_ion_sun = -3.31, rad1 = 11.92, rad2 = 18, label = '[OIII]', outfilename = 'O3_Te_chem_abund_vs_R_present.png')
    #
    ##[NII]
    #line = 5755   
    #u.Integrated_meas()
    #u.chem_abund(line)
    #u.chem_abund_emperical(line)
    #u.Te_Abund_plot(u.linefitdict['TeN2'], u.vals[8], u.int_TN2, u.int_NpH, u.NpH, u.Abund_N2,  testline = np.array(line), z = z1, log_ion_sun = -4.17, rad1 = 14.15, rad2 = 18, label = '[NII]', outfilename = 'N2_Te_chem_abund_vs_R_present.png')
#
    ##[SII]
    #line = 6716   
    #u.Integrated_meas()
    #u.chem_abund(line)
    #u.chem_abund_emperical(line)
    #u.Te_Abund_plot(u.linefitdict['TeS2'], u.vals[11], u.int_TS2, u.int_SpH, u.SpH, u.Abund_S2,  testline = np.array(line), z = z1, log_ion_sun = -4.88, rad1 = 17.5, rad2 = 18, label = '[SII]', outfilename = 'S2_Te_chem_abund_vs_R_present.png')
    #
    ##[SIII]
    #line = 6312  
    #u.Integrated_meas()
    #u.chem_abund(line)
    #u.chem_abund_emperical(line)
    #u.Te_Abund_plot(u.linefitdict['TeS3'], u.vals[12], u.int_TS3, u.int_SppH, u.SppH, u.Abund_S3,  testline = np.array(line), z = z1, log_ion_sun = -4.88, rad1 = 17.5, rad2 = 18, label = '[SIII]', outfilename = 'S3_Te_chem_abund_vs_R_present.png')
    
