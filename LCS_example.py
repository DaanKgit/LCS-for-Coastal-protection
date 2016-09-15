# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:07:41 2016

@author: dkuitenbrouwer
This file is used to calculate LCS
input:              names that refer to simulation (basefolder,runmap,par1,par2)
                    load paths

output:             save FTLE fields
                    save EV fields                    
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import prep_vel as pv
import logging
import sys
from pylab import *
import numpy.ma as ma


def calc_jacobian(xo,xn,yo,yn):#Calculate the jacobian based on initial and final tracer positions
    A = ( np.roll(xn,-1,axis = 1) - np.roll(xn,1,axis = 1) ) / abs( np.roll(xo,-1,axis = 1) - np.roll(xo,1,axis = 1) )
    B = ( np.roll(xn,-1,axis = 0) - np.roll(xn,1,axis = 0) ) / abs( np.roll(yo,-1,axis = 0) - np.roll(yo,1,axis = 0) )
    C = ( np.roll(yn,-1,axis = 1) - np.roll(yn,1,axis = 1) ) / abs( np.roll(xo,-1,axis = 1) - np.roll(xo,1,axis = 1) )
    D = ( np.roll(yn,-1,axis = 0) - np.roll(yn,1,axis = 0) ) / abs( np.roll(yo,-1,axis = 0) - np.roll(yo,1,axis = 0) )
    #print abs( np.roll(xo,-1,axis = 0) - np.roll(xo,1,axis = 0) ), ( np.roll(xn,-1,axis = 0) - np.roll(xn,1,axis = 0) ), 'A'
    j = np.zeros((2,2,(xo.shape[0] - 2)**2)); # For each particle (third dimension) the jacobian (2x2) must be determined. The -1 & -2 here and in the next line refer to the fact that we only take core of the matrix (not the boundaries) into account, cannot deal with periodic boundary conditions
    j[0,0,:] = A[1:-1,1:-1].ravel();j[0,1,:] = B[1:-1,1:-1].ravel();j[1,0,:] = C[1:-1,1:-1].ravel(); j[1,1,:] = D[1:-1,1:-1].ravel()
    return j #outputs a raveled j  
    
def calc_FTLE_EV(j,ad_t):
    CGR = np.array([ [ ( j[0,0,:]*j[0,0,:] + j[1,0,:]*j[1,0,:] ) , ( j[0,0,:]*j[0,1,:] + j[1,0,:]*j[1,1,:] ) ] , \
                     [ ( j[0,1,:]*j[0,0,:] + j[1,1,:]*j[1,0,:] ) , ( j[0,1,:]*j[0,1,:] + j[1,1,:]*j[1,1,:] ) ] ]) #Calculating the Cauchy-Green-Tensor based on the jacobian j
    CGR = np.einsum('ijk -> kij',CGR) #reshaping CGR to allow for the calculationg of the eigenvalues and eigenvectors
    vl,vc = np.linalg.eig(CGR) #eigenvalues and eigenvectors
    vlmax = np.amax(vl,axis=1) #Finding the greatest eigenvalues
    FTLE = np.log(np.sqrt(vlmax))/(abs(ad_t)) #Calculating the FTLE values for all tracers at the same time.
    return FTLE,vl,vc #returns raveled array
    
def calc_FTLE_vl_vc(posa,ad_t,trac_len,trac_wdth):#trac_len should always be an integer greater than 1 times trac_wdth  
#We need square matrices for the LCS scheme, at this point posa is raveled(), we first reshape the arrays
    posa = posa.reshape(trac_wdth,trac_len,posa.shape[1],posa.shape[2]) 
    ratio = trac_len/trac_wdth #ratio of array length with respect to array width
    FTLE_field = np.zeros((posa.shape[0],posa.shape[1]));vl1_field = np.copy(FTLE_field);vc1_field = np.zeros((posa.shape[0],posa.shape[1],2))#Creating arrays for saving fields, FTLE and vl1 (smallest eigenvalue) are scalars, vc1 is a vector field
    wdth_1block = np.minimum(trac_len,trac_wdth)    
    for ind in range(ratio): #looping through all square blocks
        bl = ind*trac_wdth; br = (ind + 1)*trac_wdth #block left and right. Posa is subdivided in '#ratio' blocks, to get square matrices and make use of numpy linalg
        try:
            j = calc_jacobian(posa[:,bl:br,0,0],posa[:,bl:br,1,0],posa[:,bl:br,0,1],posa[:,bl:br,1,1]) #Calculating the jacobian j
            FTLE,vl,vc = calc_FTLE_EV(j,ad_t); #Calculating the FTLE and eigenvectors/values of the Cauchy-Green-tensor
            FTLE_field[1:-1,(1 + bl):(br - 1)] = FTLE.reshape(              br-bl-2,br-bl-2) #Reshaping and putting in 2D array 
            vl1_field[ 1:-1,(1 + bl):(br - 1)] = np.amin(vl,axis=1).reshape(br-bl-2,br-bl-2) #Taking smallest eigenvalue, reshaping and putting in 2D
            vl1_arg = np.ones((vc.shape[1],vc.shape[0]))*np.argmin(vl,axis=1) #Finding whether greatest eigenvalue is at position 0 or 1. and putting in array
            vc1 = np.amin(vl,axis=1)*np.einsum('ij->ji',np.where(np.einsum('ij->ji',vl1_arg).astype(bool),vc[:,:,1],vc[:,:,0])); #smallest eigenvector is smallest eigenvalue (where vl1_arg == True (bool(1)) --> True refers to smallest eigenvalue at position 1 because the former line asks for argmin) times the direction of the eigenvector (unit vector), index of ev; (1 (index) for true and 0 (index) for false). Einsums only change the axes of the matrix back and forth to allow for similar matrix shape in where statement (eigenvalues and eigenvectors)            
            vc1_field[1:-1,1 + wdth_1block*ind:wdth_1block*(ind+1) - 1,:] = np.einsum('ij -> ji',vc1).reshape(wdth_1block-2,wdth_1block-2,2)             
        except LinAlgError as e:
            logging.warning('%s'%(e))            
            print e
            break
    return FTLE_field, vc1_field
    
def repair_pos_fields(array,posa_or):
    array = array.reshape(100,300,2)
    array[:,:200,0] = np.where(array[:,:200,1]> 3363300,np.roll(array[:,:200,0],-1,axis=1)-100,array[:,:200,0])
    array[:,:200,1] = np.where(array[:,:200,1]> 3363300,np.roll(array[:,:200,1],-1,axis=1)-100,array[:,:200,1])
    array[:,200:,0] = np.where(array[:,200:,1]> 3365000,np.ones((100,100))*547500 + np.random.random((100,100))*10,array[:,200:,0])
    array[:,200:,1] = np.where(array[:,200:,1]> 3365000,np.ones((100,100))*3365000 + np.random.random((100,100))*10,array[:,200:,1])
    array = ma.masked_where(abs(array) > 4000000,array)  
    array = ma.masked_where(~np.isfinite(array),array)
    msk = np.copy(array.mask)
    array[:,:,:]    = np.where(array[:,:,:].mask,posa_or[:,:].reshape(100,300,2),array)    
    array = array.reshape(30000,2)
    return array,msk
    
def repair_FTLE(FTLE_field,msk):
    FTLE_rep = np.where(msk[:,:,0],np.zeros((FTLE_field.shape)),FTLE_field)
    return FTLE_rep
        

def main_LCS(basefolder,runmap,sim_name,t_adv_end,ad_t,trac_len,trac_wdth,posa_or):
    posa = np.zeros((posa_or.shape[0],2,2));posa[:,0,:] = posa_or #posa has shape (ntrac,2,2), where (ntrac,0,:) refers to original positions and (ntrac,1,:) refers to final positions. The last index refers to (x,y) position in UTM space
    for t in t_adv_end:
        tbs = int(t);tdec = t-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving/loading (time base and time decimal)
        newp = np.load(basefolder + runmap + 'aft_adv_%s_ti%s_%s_BT.npy'%(sim_name,tbs,tdec)) #loading the advected positions into the posa array        
        if np.any(~np.isfinite(newp)):
            posa[:,1,:],msk = repair_pos_fields(newp,posa_or)
            FTLE_field,vc1_field = calc_FTLE_vl_vc(posa,ad_t,trac_len,trac_wdth) 
            FTLE_field = repair_FTLE(FTLE_field,msk)
        else:
            posa[:,1,:] = np.copy(newp)
            FTLE_field,vc1_field = calc_FTLE_vl_vc(posa,ad_t,trac_len,trac_wdth) 
        #posa[:,1,:] = np.load(basefolder + runmap + 'aft_adv_%s_ti%s_%s_BT.npy'%(sim_name,tbs,tdec)) #loading the advected positions into the posa array
        np.save(basefolder + runmap + 'FTLE_field_%s_t_%s_%s'%(sim_name,tbs,tdec),FTLE_field)
        np.save(basefolder + runmap + 'vc1_field_%s_t_%s_%s'%(sim_name,tbs,tdec),vc1_field)
    return
    
def main_LCS_dg_unst(basefolder,runmap,sim_name,t_adv_end,ad_t,trac_len,trac_wdth,posa_or,time_dir):
    posa = np.zeros((posa_or.shape[0],2,2));posa[:,0,:] = posa_or #posa has shape (ntrac,2,2), where (ntrac,0,:) refers to original positions and (ntrac,1,:) refers to final positions. The last index refers to (x,y) position in UTM space
    for t in t_adv_end:
        tbs = int(t);tdec = t-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving/loading (time base and time decimal)
        posa[:,1,:] = np.load(basefolder + runmap + 'aft_adv_%s_ti%s_%s_%s.npy'%(sim_name,tbs,tdec,time_dir)) #loading the advected positions into the posa array
        FTLE_field,vc1_field = calc_FTLE_vl_vc(posa,ad_t,trac_len,trac_wdth) 
        np.save(basefolder + runmap + 'FTLE_field_%s_t_%s_%s'%(sim_name,tbs,tdec),FTLE_field)
        np.save(basefolder + runmap + 'vc1_field_%s_t_%s_%s'%(sim_name,tbs,tdec),vc1_field)
        
    return
