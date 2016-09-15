# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 15:46:51 2016

@author: dkuitenbrouwer
Prepare velocities
inputs:                     basefieldname (refers to .mat from D3D)
                            advection time
                            
outputs:                    velocity fields as a function of position only. separate files for each timestep, files are saved.
                            Module returns 'nomatfile_error' if this is the case main should go to try next set.
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import logging
import os

def uv_no_cross_boun(UD3D,VD3D,basefolder): #this definition intends to adjust the D3D flow field such that tracers cannot cross the boundaries of the domain, which results in a nan and the code to crash.
    #positions needed for adjusting velocity field
    a_u_e_2d = np.load( basefolder + 'assign_u_nan_east_inlet.npy')
    a_u_w_2d = np.load( basefolder + 'assign_u_nan_west_inlet.npy')
    a_v_2d = np.load( basefolder + 'assign_v_nan.npy')
    sw_v_2d = np.load( basefolder + 'swap_v_grd0.npy')
    
    #Making 3D
    a_u_e = np.ones((UD3D.shape));a_u_e*=a_u_e_2d
    a_u_w = np.ones((UD3D.shape));a_u_w*=a_u_w_2d
    a_v = np.ones((UD3D.shape));a_v*=a_v_2d
    sw_v = np.ones((UD3D.shape));sw_v*=sw_v_2d
    
    #Adjusting the field
    UD3D = np.where(a_u_e,-abs(np.roll(UD3D,1,axis=0)),UD3D)
    UD3D = np.where(a_u_w, abs(np.roll(UD3D,1,axis=0)),UD3D)
    VD3D = np.where(a_v,  -abs(np.roll(VD3D,1,axis=0)),VD3D)
    VD3D = np.where(sw_v, -abs(VD3D),                  VD3D)
    UD3D[0,:] = UD3D[1,:]
    UD3D[:,:,72] = UD3D[:,:,73];UD3D[:,:,71] = UD3D[:,:,73]
    UD3D[:,:,-2] = UD3D[:,:,-3];UD3D[:,:,-1] = UD3D[:,:,-3]
    VD3D[:,0,:] = VD3D[:,1,:]
    VD3D[:,:,72] = VD3D[:,:,73];VD3D[:,:,71] = VD3D[:,:,73]
    VD3D[:,:,-2] = VD3D[:,:,-3];VD3D[:,:,71] = VD3D[:,:,73]
    UD3D[:,63,72] = UD3D[:,62,73];UD3D[:,63,71] = UD3D[:,62,73];UD3D[:,64,257] = UD3D[:,63,256];UD3D[:,64,258] = UD3D[:,63,256]
    VD3D[:,63,72] = -abs(VD3D[:,62,73]);VD3D[:,63,71] = -abs(VD3D[:,62,73]);VD3D[:,64,257] = -abs(VD3D[:,63,256]);VD3D[:,64,258] = -abs(VD3D[:,63,256]) #getting values for the NE and NW corners of the sea domain
    
    return UD3D,VD3D

def load_vel_dat(basefolder,fld_folder,basefilename,adj_time):
    print basefolder + fld_folder + basefilename + 'v_t.mat','basefolder + fld_folder + basefilename + v_t.mat'
    nomatfile_error = False    
    try:
        VD3D = loadmat( basefolder + fld_folder + basefilename + 'v_t.mat');VD3D = VD3D['vy'];VD3D = np.einsum('ijk -> ikj',VD3D)#take out runmap as all flds are saved in basefolder
        UD3D = loadmat( basefolder + fld_folder + basefilename + 'u_t.mat');UD3D = UD3D['ux'];UD3D = np.einsum('ijk -> ikj',UD3D)
        TD3D = loadmat( basefolder + fld_folder + basefilename + 't.mat');TD3D = TD3D['T'][0,:]-adj_time #minus adj_time to start at time zero

        UD3D,VD3D = uv_no_cross_boun(UD3D,VD3D,basefolder) #Fields at all times adjusted (adjusting is not time dependent)
    except IOError as e:
        logging.error('%s'%(e) + 'try to continue with next file')
        print e, 'try to continue with next file'
        nomatfile_error = True
        return 0,0,0,nomatfile_error
    print 'loading %s data OK'%basefilename
    return UD3D,VD3D,TD3D,nomatfile_error

def split_interp_fld(UD3D,VD3D,TD3D,t_stp_D3D,ndays,sim_name,basefolder,runmap):
    TD3D_l = np.zeros((TD3D.shape[0]+1));TD3D_l[:-1] = TD3D;TD3D_l[-1] = TD3D[0]+ndays#TD3D is made one step longer to avoid index errors. If we work with splitting up the D3D files into day parts, doing +1 works.
    if ndays < 1:
        ndays = int(1)
    for dy in range(ndays):                                                           #Splitting the flow fields per day
        print dy,'dy'
        ud_int = np.zeros(((t_stp_D3D*6*24 + t_stp_D3D),UD3D.shape[1],UD3D.shape[2]))
        vd_int = np.zeros(((t_stp_D3D*6*24 + t_stp_D3D),UD3D.shape[1],UD3D.shape[2]))
        for i in range(UD3D.shape[1]):
            for j in range(UD3D.shape[2]):
                ud_int[:,i,j] = np.interp(np.linspace(TD3D_l[dy*6*24],TD3D_l[(dy+1)*6*24],6*24*t_stp_D3D + t_stp_D3D),TD3D_l[dy*6*24:(dy+1)*6*24 + 1],UD3D[dy*6*24:(dy+1)*6*24+1,i,j]) #linearly interpolating in time for each position separately (might be faster with broadcasting)
                vd_int[:,i,j] = np.interp(np.linspace(TD3D_l[dy*6*24],TD3D_l[(dy+1)*6*24],6*24*t_stp_D3D + t_stp_D3D),TD3D_l[dy*6*24:(dy+1)*6*24 + 1],VD3D[dy*6*24:(dy+1)*6*24+1,i,j])
        td_int = np.interp(np.linspace(TD3D_l[dy*6*24],TD3D_l[(dy+1)*6*24],6*24*t_stp_D3D + t_stp_D3D),TD3D_l[dy*6*24:(dy+1)*6*24 + 1],TD3D_l[dy*6*24:(dy+1)*6*24+1]);
        np.save(basefolder + runmap + 'u_int_%s_ti%s'%(sim_name,dy),ud_int)#save name refers to initial time integer of day
        np.save(basefolder + runmap + 'v_int_%s_ti%s'%(sim_name,dy),vd_int)
        np.save(basefolder + runmap + 't_int_%s_ti%s'%(sim_name,dy),td_int)
    return

#Managing the above definitions
def prep_vel(basefolder,runmap,fld_folder,basefilename,sim_name,t_stp_D3D,ndays,adj_time,Uwnd,Vwnd):
    UD3D,VD3D,TD3D,nomatfile_error = load_vel_dat(basefolder,fld_folder,basefilename,adj_time)  #Loading velocity fields from D3D
    UD3D += Uwnd; VD3D += Vwnd                                                                  #adding windage to the velocity fields
    if nomatfile_error ==  True:
        return nomatfile_error
    split_interp_fld(UD3D,VD3D,TD3D,t_stp_D3D,ndays,sim_name,basefolder,runmap)                 #Splitting and interpolating velocity fields
    return nomatfile_error
    
def prep_vel_scope(basefolder,runmap,fld_folder,basefilename,sim_name,t_stp_D3D,ndays,adj_time,wdg):
    UD3D,VD3D,TD3D,nomatfile_error = load_vel_dat(basefolder,fld_folder,basefilename,adj_time)
    wnd = np.loadtxt('D:\dkuitenbrouwer/waves/fld_scope/scope_1m.txt')
    ind_st = 4*24*60; ind_end = 18*60 + 10 #Considering intital and final times of wind file in comparison to simulation
    wnd_scope = wnd[ind_st:-ind_end,:]    
    uwnd = -wnd_scope[:,1]*np.sin(wnd_scope[:,2]*np.pi/180.)
    vwnd = -wnd_scope[:,1]*np.cos(wnd_scope[:,2]*np.pi/180.)
    #averaging wnd timeseries to add them to D3D velocities
    uwnd_av = np.zeros((TD3D.shape[0]+1))
    vwnd_av = np.zeros((TD3D.shape[0]+1))
    for i in range(TD3D.shape[0]-1):
        uwnd_av[i] = np.mean(uwnd[i*10:(i*10) + 10])
        vwnd_av[i] = np.mean(vwnd[i*10:(i*10) + 10])
    uwnd_av[-2] = uwnd[-1];vwnd_av[-2] = vwnd[-1];
    uwnd_av[-1] = uwnd[-1];vwnd_av[-1] = vwnd[-1];
    UD3D = np.einsum('ijk -> kij',np.einsum('ijk -> jki',UD3D) + uwnd_av*wdg)#adding the windage to the surface flow
    VD3D = np.einsum('ijk -> kij',np.einsum('ijk -> jki',VD3D) + vwnd_av*wdg)
    if nomatfile_error ==  True:
        return nomatfile_error
    split_interp_fld(UD3D,VD3D,TD3D,t_stp_D3D,ndays,sim_name,basefolder,runmap)
    return nomatfile_error    