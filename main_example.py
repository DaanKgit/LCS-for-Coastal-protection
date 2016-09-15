# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 15:46:51 2016

@author: dkuitenbrouwer

This file aims to manage all Lagrangian simulation, hence based on D3D output it calculates coastal protection
inputs/specify:         surface flow fields (x,t)
                        forcing sets (names)
                        advection times
            
outputs:                advected paths (forward and backward)
                        location barrier (x,t)
                        location protection (x,t)          
                        plots (on/off) FTLE-, EV-, vorticity-, divergence- fields
"""

#Pre-ambule
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

#importing own modules
import prep_vel as pv
import RK4
import LCS
import DVD
import BAR_PROT as BP

[reload(mo) for mo in (np,pv,RK4,LCS,DVD,BP)]
    
#Set what actions you wish to perform and set parameters. Note that the code runs on a single core.

#Prepare velocity fields-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#LOG = True
LOG = True

PREP_VEL = True;    #Preparing the D3D (x,t) surface velocity fields
ad_t = 120                #Advection time in both backward and foreward time in minutes. Advection time can be different from time between to end/start moments of advection 
basefolder = 'D:\dkuitenbrouwer/waves/'
fld_folder = 'flds_realistic_wave/'
posa_or = np.load( basefolder + 'posa30000_init_cls.npy');posa_or = posa_or[:,0,:-1];#Setting the initial tracer setup

#Particle advection
RK4_BT = True
trac_grid = '30000shore'              #pick a specific setup. currently this is the only one available. See shore_tracers file to obtain info on where the tracers originate
trac_len = 300; trac_wdth = 100

#Calculate field based on advection.
CALC_LCS = True;                            #Calculating FTLE fields, Eigenvectors from CGT
num_squares = 3                             #LCS scheme makes use of np.linalg scheme that needs square matrices, standard setup is (3*100)*100 tracers, hence 3 squares

#Calculate Barriers and protection
PROT = True                                 #Calculate protection
sm = 60000                                  #Smoothing factor for the FTLE field
perc = 0.96                                 #*100% is lowest part that is not considered
lim_lab1 = -0.01                            #Maximum value for the smallest eigenvalue for a domain to be accepted as ridge
lim_GdotZ = 0.1                             #Max value for domain to be on ridge.
inc_num = 8                                 #number of gridpoints increase factor along each axis for increasing resolution in heigthfield
d_box = 50                                  #Box width within which no new peaks can be found to avoid many peaks very close to one another.
d_an_r_max = np.pi/4.                       #Max difference in orientation between smallest eigenvector and ridge
d_an_ev_max = np.pi/3.                      #Max difference in orientation between smallest eigenvectors of two consecutive points along a ridge
perc_num_disr = 0.15                        #percentage of ridge points that is disregarded for having to great height gradients
d_max = 2500                                #Max distance between two consecutive ridge points in m
orth_dist = 200                             #Orthogonal distance of line that shows barrier
min_oneside_stretch_rat = 7.                #ratio of convergence on both sides of ridge for which only one side is taken into account (one sided convergence)
num_stretch = 80                            #Number of stretches along the coast for which protection is considered
save_plot = True                            
fig_size = 18                               #inch width of figure, figures made such that 2 below one another fit on a4

#Set parameters simulation----------------------------------------------------------------------------------------------------------------------------------------------------------
wv_dirfrom_l = ['SW','SW','SW','S','S','S','SE','SE','SE','STORM']      #Refers to direction of wave sets
wv_height_l = [0.5,1,2,0.5,1,2,0.5,1,2,4]                               #Refers to wave height
wv_period_l = [4.3,5,6.5,5.9,6.4,7.8,5.4,5.8,7.2,9]                     #wave period. Untill here only neede for finding the right file
wind_dir_l = [247.5,247.5,247.5,180,157.5,157.5,135,135,112.5,135]      #wind direction, needed for file
wind_speed_l = [5,7.5,12,4,6,8.5,5,6.5,9,15]                            #wind speed, needed for file and windage
rotl = [np.pi/8.,np.pi/8.,np.pi/8.,np.pi/2.,(5*np.pi)/8.,(5*np.pi)/8.,(3*np.pi)/4.,(3*np.pi)/4.,(7*np.pi)/8.,(3*np.pi)/4.] #works for wind going to E is zero degrees, going to S --> -np.pi/2, needed for calculating windage
trl  = [0.3,0.6,1.2]                                                    #Tidal range list, needed for finding right file
trpercl = ['050','100','200']                                           #Tidal range list, needed for finding right file
wdgl = [0,0.02,0.035,0.05]                                              #Windage percentage
ndays = 6                                                               #length of simulation (must be same in D3D)
t_stp_D3D = 10                                                          #time step D3D

#Setting the timelists
minYD = 1./(60*24)                                                      #minutes to yearday
t_adv_end = np.arange(ad_t*minYD,ndays,ad_t*minYD)                      #timeseries with time at the end of the advection
dtRK4 = 20                                                              #RK4 timestep
adj_time = 338                                                          #D3D data starts with YD 338, does not make sense for the experiments.

def write_vars_to_file(_f, d):
    for (name, val) in d.items():
        _f.write("%s = %s\n" % (name, repr(val)))
    _f.close()
    return

#Starting simulation-------------------------------------------------------------------------------------------------------------------------------------------------------------------
wv_set = 5
for tr_ind,tr in enumerate(trl):
    for wdg in wdgl:
        trbs = int(tr);trdec = tr-trbs;trdec = format(trdec,'.1f');trdec=trdec[2]
        wdgbs = int(wdg);wdgdec = wdg-wdgbs;wdgdec = format(wdgdec,'.3f');wdgdec=wdgdec[2:5]
        wvhbs = int(wv_height_l[wv_set]);wvhdec = wv_height_l[wv_set]-wvhbs;wvhdec = format(wvhdec,'.1f');wvhdec=wvhdec[2]
        sim_name = 'wv_wnd_set_%str_%s_%s_wdg_%s_%s'%(wv_set,trbs,trdec,wdgbs,wdgdec)    #setting the simulation name based on above names
        runmap = sim_name +'/'
        print sim_name,'sim_name'
        if not os.path.exists(basefolder + sim_name):
            os.makedirs(basefolder + sim_name)
        basefilename = 'fldwv_%s%s%s_ampl_x%s'%(wv_dirfrom_l[wv_set],wvhbs,wvhdec,trpercl[tr_ind])
        print basefilename
        Uwnd = wdg*wind_speed_l[wv_set]*np.cos(rotl[wv_set])                           #Calculating extra velocity due to windage
        Vwnd = wdg*wind_speed_l[wv_set]*np.sin(rotl[wv_set])
        print Uwnd,'Uwnd',Vwnd,'Vwnd',wv_set
        desc_dict = {'LOG':LOG,'PREP_VEL':PREP_VEL,'ad_t':ad_t,'basefolder':basefolder,'RK4_BT':RK4_BT,'trac_grid':trac_grid,\
'trac_len':trac_len,'trac_wdth':trac_wdth,'CALC_LCS':CALC_LCS,'num_squares':num_squares,'PROT':PROT,\
'sm':sm,'perc':perc,'lim_lab1':lim_lab1,'lim_GdotZ':lim_GdotZ,'inc_num':inc_num,'d_box':d_box,'d_an_r_max':d_an_r_max, \
'd_an_ev_max':d_an_ev_max, 'perc_num_disr':perc_num_disr, 'd_max':d_max,'orth_dist':orth_dist,'min_oneside_stretch_rat':min_oneside_stretch_rat, \
'num_stretch':num_stretch,'save_plot':save_plot,'fig_size':fig_size,'ndays':ndays,'t_stp_D3D':t_stp_D3D,'minYD':minYD,'t_adv_end':t_adv_end,\
'dtRK4':dtRK4,'Windage':wdg,'Uwnd':Uwnd,'Vwnd':Vwnd,'adj_time':adj_time,'sim_name':sim_name,'basefilename':basefilename,'fld_folder':fld_folder}
        simulation_description = open(basefolder + runmap + 'simulation_description','w')
        write_vars_to_file(simulation_description,desc_dict)
        
        if LOG == True:                                                             #Switching on logger for info, warnings and errors
            logname = 'log_%s'%sim_name
            logger = logging.getLogger()
            fhandler = logging.FileHandler(filename= basefolder + runmap + logname, mode='a')#could add info on the D3D simulation to the filename
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fhandler.setFormatter(formatter)
            logger.addHandler(fhandler)
            logger.setLevel(logging.DEBUG)
            logging.info('This simulation:PREP_VEL = %s, ad_t = %s, RK4_BT = %s'%(PREP_VEL,ad_t,RK4_BT))
                            
        if PREP_VEL == True:
            nomatfile_error = pv.prep_vel(basefolder,runmap,fld_folder,basefilename,sim_name,t_stp_D3D,ndays,adj_time,Uwnd,Vwnd) #Preparing velocity fields, 
            if nomatfile_error == True:
                logging.error(basefolder+runmap+basefilename,'not found, continue with next forcing set')            
                print basefilename, 'not found, continue with next forcing set'
                continue
    
        if RK4_BT == True:
            time_dir = 'BT'
            RK4.main_RK4_unsteady(basefolder,runmap,sim_name,ndays,time_dir,t_adv_end,dtRK4,trac_grid,ad_t,posa_or)    
            
        if CALC_LCS == True:
            LCS.main_LCS(basefolder,runmap,sim_name,t_adv_end,ad_t/(24*60.),trac_len,trac_wdth,posa_or)
       
        if PROT == True:
            BP.main_BAR_PROT(basefolder,runmap,sim_name,t_adv_end,trac_len,trac_wdth,posa_or,sm,perc,lim_lab1,lim_GdotZ,inc_num, \
d_box,d_an_r_max,d_an_ev_max,perc_num_disr,d_max,orth_dist,min_oneside_stretch_rat,num_stretch,save_plot,fig_size)
        for day in range(ndays):
            try:
                os.remove(basefolder + runmap + 'u_int_%s_ti%s.npy'%(sim_name,day))
                os.remove(basefolder + runmap + 'v_int_%s_ti%s.npy'%(sim_name,day))
                os.remove(basefolder + runmap + 't_int_%s_ti%s.npy'%(sim_name,day))
            except WindowsError as e:
                print e
