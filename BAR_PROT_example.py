# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 08:49:11 2016

@author: dkuitenbrouwer
This file aims to find the transport barriers and the coastal protection.

input:              names (FTLE, eigenvectorfield, Divergence, tracer position final Forward time)
                    Parameters: llim
                    
output              saved (Barrier positions, coastal protection)    

hard coded:         Position of the inlet mask                
"""

#Preambule
from scipy import stats
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime as dt
import numpy.ma as ma
import sys
import copy
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def interp_bands_msk_inlet(FTLE_or,inlmsk): #interpolate the vertical bands in the FTLE field with zero values due to square block method to obtain fully filled field
    FTLE_intb = np.copy(FTLE_or)
    ratio = FTLE_intb.shape[1]/FTLE_intb.shape[0] #Calculating the number of square blocks, this number minus one is the number of bands that has to be filled in
    boundaries = np.linspace(FTLE_intb.shape[0],(ratio-1)*FTLE_intb.shape[0],(ratio-1)).astype(int) #finding the horizontal (x) positions of the bands/boundaries
    FTLE_left_of_bound  = np.einsum('ij -> ji',FTLE_intb)[boundaries - 2] #Assuming that the width of the boundaries is two grid cells, this finds the FTLE value left of the boundary
    FTLE_right_of_bound = np.einsum('ij -> ji',FTLE_intb)[boundaries + 1]
    d_FTLE_dx = (FTLE_right_of_bound - FTLE_left_of_bound)/3. #Calculating the slope of the FTLE field
    for i in range(ratio -1): #plugging interpolated FTLE values in the FTLE array by making use of the side value and the slope
        FTLE_intb[:, ( boundaries[i] - 1) ] = FTLE_left_of_bound[i,:] + (1/3.)*d_FTLE_dx[i,:]
        FTLE_intb[:, ( boundaries[i]    ) ] = FTLE_left_of_bound[i,:] + (2/3.)*d_FTLE_dx[i,:]
#masking the inlet
    FTLE_intb_nomask = np.copy(FTLE_intb)
    FTLE_intb = np.where(inlmsk,0,FTLE_intb)        
    return FTLE_intb,FTLE_intb_nomask
    
def prep_rect_smoothed_height_field(posa_or,array,inc_num,sm,trac_len,trac_wdth):
    sm_err = False
    h_fld = np.zeros((posa_or.shape[0]*inc_num,posa_or.shape[1]*inc_num,3))
    h_fld_interp_no_smooth = np.zeros((posa_or.shape[0]*inc_num,posa_or.shape[1]*inc_num,3))
    x_pos_pre = np.linspace(posa_or[0,0,0],posa_or[0,-1,0],posa_or.shape[1])#creating grid as if curvilinear grid were rectangular
    x_pos_intp = np.linspace(posa_or[0,0,0],posa_or[0,-1,0],inc_num*posa_or.shape[1])#Creating refined grid as if curvilinear grid were rectangular
    if trac_len == trac_wdth:#This is valid for the double gyre experiments, not for the Choctawhatchee bay grid
        y_pos_pre = np.linspace(np.amin(posa_or[:,:,1]),np.amax(posa_or[:,:,1]),posa_or.shape[0])
        y_pos_intp = np.linspace(np.amin(posa_or[:,:,1]),np.amax(posa_or[:,:,1]),inc_num*posa_or.shape[0])
    else:
        y_pos_pre = np.linspace(posa_or[0,0,1],np.amax(posa_or[:,:,1]),posa_or.shape[0])
        y_pos_intp = np.linspace(posa_or[0,0,1],np.amax(posa_or[:,:,1]),inc_num*posa_or.shape[0])
    XX_intp,YY_intp = np.meshgrid(x_pos_intp,y_pos_intp) #rectangular mesh XX works for new positions, True field is not rectangular, hence YY only serves for plotting purposes here
    try:    
        z_spline_sm = interpolate.RectBivariateSpline(x_pos_pre,y_pos_pre,np.einsum('ij -> ji',array), bbox=[None, None, None, None], kx=3, ky=3, s=sm)     #Creating a function that interpolates and smooths at the same time. I'm not entirely sure why this s(mooth) value works out, but it does.
    except ValueError as e:
        print e
        logging.error(e)
        sm_err = True
        return 0,0,sm_err
        
    z_spline_non_sm = interpolate.RectBivariateSpline(x_pos_pre,y_pos_pre,np.einsum('ij -> ji',array), bbox=[None, None, None, None], kx=3, ky=3, s=0)     #Creating a function that interpolates and smooths at the same time. I'm not entirely sure why this s(mooth) value works out, but it does.    
    h_fld[:,:,-1] = np.einsum('ij -> ji',z_spline_sm(x_pos_intp,y_pos_intp)) #Finding the smoothed values on all interpolated points and assigning it to the h_fld array
    h_fld[:,:,0]  = XX_intp  ;h_fld[:,:,1]  = YY_intp #The actual positions will only be given in the end, such that working with a rectangular grid remains possible.
    h_fld_interp_no_smooth[:,:,-1] = np.einsum('ij -> ji',z_spline_non_sm(x_pos_intp,y_pos_intp)) #Finding the smoothed values on all interpolated points and assigning it to the h_fld array
    h_fld_interp_no_smooth[:,:,0]  = XX_intp  ;h_fld_interp_no_smooth[:,:,1]  = YY_intp #The actual positions will only be given in the end, such that working with a rectangular grid remains possible.    
    return h_fld,h_fld_interp_no_smooth,sm_err
        
def calc_grad_hes_vel_vec(array):
    g = np.zeros((array.shape[0],array.shape[1],2))
    gy,gx = np.gradient(array);g[:,:,0] = gx;g[:,:,1] = gy
    H = np.zeros((gy.shape[0],gy.shape[1],2,2))
    H22,H21 = np.gradient(gy);H12,H11 = np.gradient(gx) #Calculating Hessian from double gradient
    if np.amax(abs(H21-H12)) > 0.00001:
        logging.warning('possible problems with non symmetric Hessian')
        print 'possible problems with non symmetric Hessian'
    H[:,:,0,0] = H11;H[:,:,0,1] = H12;H[:,:,1,0] = H12;H[:,:,1,1] = H22#Assigning the values of the Hessian. Note that H21 is not used, H12 twice in order to assure symmetry.
#Calculating the eigenvectors from the Hessian
    vl,vc = np.linalg.eig(H)
    vl_min = np.amin(vl,axis=2); #Find smallest eigenvalue field
    vl1_arg = np.argmin(vl,axis=2) #Finding whether greatest eigenvalue is at position 0 or 1. and putting in array
    g_len = np.sqrt( g[:,:,0]**2 + g[:,:,1]**2)
    vc1 = np.zeros((vl.shape))
    vc1[:,:,0] = np.amin(vl,axis=2)*np.where(vl1_arg.astype(bool),vc[:,:,0,0],vc[:,:,0,1]); #Eigenvector that aligns with the ridge, only unit value is taken in order to avoid influence of size vector on the gdotz calculation
    vc1[:,:,1] = np.amin(vl,axis=2)*np.where(vl1_arg.astype(bool),vc[:,:,1,0],vc[:,:,1,1]); #giving them size of largest eigenvalue for comparison 
    GdotZ = abs( (g[:,:,0]/g_len) * (vc1[:,:,0]) - (g[:,:,1]/g_len) * (vc1[:,:,1]) )#absolute value of g dot zeta1 to easily calculate how far a value is off. minus sign is probably because of coordinate system?
    return vl_min,GdotZ,vl,vc1
    
def height_masker(heights,perc):  
    srted = np.sort(np.ravel(heights)) #sorting
    height_lim_n = int(perc*srted.shape[0]);height_lim = srted[height_lim_n] #determining the minimum height on the sorted array
    height_msk = np.where(heights > height_lim,0,1);height_msk = height_msk.astype(bool)
    return height_msk    
    
def only_peaks_all_req(heights_smoothed,vl_min,GdotZ,perc,lim_lab1,lim_GdotZ,heights_true):
# This definition returns an array with only the values that are not masked by the three (height, smallest eigenvalue (S&R) and GdotZ (S&R)) requirements
#Creating a height mask
    height_msk = height_masker(heights_true,perc)
#creating a mask based on requirment smallest eigenvalue hessian
    lab_msk = np.where(vl_min < lim_lab1, 0, 1); lab_msk = lab_msk.astype(bool)
#Creating a mask based on requirement GdotZ
    GdotZ_msk = np.where(GdotZ < lim_GdotZ,0,1); GdotZ_msk = GdotZ_msk.astype(bool)
#Creating total mask
    tot_msk = height_msk + lab_msk + GdotZ_msk
    tot_msk = np.where(tot_msk > 0, 1, 0)
    return tot_msk 
    
def find_local_peaks(h_fld,tot_msk,d_box,vc1,h_fld_interp_no_smooth):#h_fld is redundant if h_fld_interp_no_smooth will be used
    init_num_non_mask = float(len(ma.compressed(ma.masked_where(tot_msk,tot_msk)))) 
    xl = [];yl = []
    if init_num_non_mask != 0:
        surf_perc_masked = len(ma.compressed(ma.masked_where(tot_msk,tot_msk)))/init_num_non_mask
    else:
        surf_perc_masked = 0       
    while surf_perc_masked > 0.05:
        indy,indx = np.unravel_index(np.argmax(np.where(tot_msk,0,h_fld_interp_no_smooth[:,:,-1])),[tot_msk.shape[0],tot_msk.shape[1]])
        xl.append(indx);yl.append(indy) 
        min_ind_x = np.maximum(0,indx-d_box/2);max_ind_x = np.minimum(tot_msk.shape[1],indx+d_box/2);
        min_ind_y = np.maximum(0,indy-d_box/2);max_ind_y = np.minimum(tot_msk.shape[0],indy+d_box/2);
        tot_msk[min_ind_y:max_ind_y,min_ind_x:max_ind_x] = 1
        surf_perc_masked_new = len(ma.compressed(ma.masked_where(tot_msk,tot_msk)))/init_num_non_mask
        if surf_perc_masked_new == surf_perc_masked:
            print 'break out while loop, surf_perc_masked =',surf_perc_masked
            break
        else:
            surf_perc_masked = surf_perc_masked_new
    x_pos = np.array(xl);y_pos = np.array(yl)
    vc1x = np.zeros((x_pos.shape));vc1y = np.zeros((y_pos.shape))
    utm_h = np.zeros((x_pos.shape[0],3))
    for i in range(len(x_pos)):
        vc1x[i] = np.mean( vc1[  (y_pos[i]-2):(y_pos[i]+2) , (x_pos[i]-2):(x_pos[i]+2),0]  ) #getting the average eigenvector. Goes wrong if eigenvector direction suddenly changes (swaps)
        vc1y[i] = np.mean( vc1[  (y_pos[i]-2):(y_pos[i]+2) , (x_pos[i]-2):(x_pos[i]+2),1]  )
        utm_h[i,:-1] = h_fld_interp_no_smooth[y_pos[i],x_pos[i],:-1]
        utm_h[i,-1]  = np.amax(  h_fld_interp_no_smooth[  (y_pos[i]-2):(y_pos[i]+2) , (x_pos[i]-2):(x_pos[i]+2),-1]  ) #finding the highest value in the quadrant
    return x_pos,y_pos,vc1x,vc1y,utm_h
    
def grad_check(utm_h,x_pos,y_pos,vc1x,vc1y,perc_num_disr):#take out positions that have to great vertical gradients. Currently the number to take out is set at five, this could also be specified by a maximum gradient.
    grad_ar = np.zeros((utm_h.shape[0],utm_h.shape[0]))
    for i in range(len(utm_h)):
        for j in range(len(utm_h)):
            if i == j:
                continue
            grad_ar[i,j] = abs( utm_h[j,-1] - utm_h[i,-1] ) / np.sqrt( (utm_h[j,0] - utm_h[i,0])**2  + (utm_h[j,1] - utm_h[i,1])**2 )
    grad_sum = np.sum(grad_ar,axis=0)
    num_disr = int(round(perc_num_disr*utm_h.shape[0]))
    del_args = np.argsort(grad_sum[int(len(grad_sum)/2):])[-num_disr:] + int(len(grad_sum)/2)
    utm_h = np.delete(utm_h,del_args,0)
    x_pos = np.delete(x_pos,del_args)
    y_pos = np.delete(y_pos,del_args)
    vc1x = np.delete(vc1x,del_args)
    vc1y = np.delete(vc1y,del_args)
    return utm_h,x_pos,y_pos,vc1x,vc1y
    
def calc_delta_angles(ind,new_ind,vc1x,vc1y,utm_h):
    #Checking the difference in ev requirement
    ev_an_ind = np.arctan( vc1y[ind] / vc1x[ind]);ev_an_new_ind = np.arctan( vc1y[new_ind] / vc1x[new_ind])
    d_ev_an =  np.amin( [ abs( ev_an_ind - (ev_an_new_ind + np.pi) ), abs( ev_an_ind - ev_an_new_ind ), abs( ev_an_ind - (ev_an_new_ind - np.pi) ) ] ) #Finding the angle between the eigenvectors, taking into account that they may be in different quadrants
    quadr_param = np.argmin( [ abs( ev_an_ind - (ev_an_new_ind + np.pi) ), abs( ev_an_ind - ev_an_new_ind ), abs( ev_an_ind - (ev_an_new_ind - np.pi) ) ] ) #Finding the angle between the eigenvectors, taking into account that they may be in different quadrants            
    if quadr_param == 0:
        ev_an_av = ev_an_ind + d_ev_an/2.
    if quadr_param == 1:
        ev_an_av = np.amin([ev_an_ind,ev_an_new_ind]) + d_ev_an/2.
    if quadr_param == 2:
        ev_an_av = ev_an_ind - d_ev_an/2.                
    dy_prop = utm_h[ind,1] - utm_h[new_ind,1];dx_prop = utm_h[ind,0] - utm_h[new_ind,0]#distances to proposed new point
    ridge_an = np.arctan( dy_prop /  dx_prop )
    d_r_ev_av_an = np.amin( [ abs( ridge_an - (ev_an_av+ np.pi) ), abs( ridge_an - ev_an_av), abs( ridge_an - (ev_an_av - np.pi) ) ] ) #Finding the angle between the eigenvectors, taking into account that they may be in different quadrants
    return dx_prop,dy_prop,d_r_ev_av_an,d_ev_an
    
def create_ridge_req_SR(x_pos,y_pos,vc1x,vc1y,utm_h,h_fld,d_max,d_min,d_an_r_max,d_an_ev_max):#x_pos,y_pos (i,j indices of grid), eigenvectors at that point, h_ld, maximum distance between points on ridge, maximum angle between ridge and ev, maximum angle between two consecutive ev
    utm_h_unch = np.copy(utm_h)#unchanged array
    ridgesx = [];ridgex = []; 
    ridgesy = [];ridgey = []; 
    while np.sum(utm_h[:,-1]) > 0: #Check whether the summation of the heights > 0. If a position is checked, it's height will be made zero.
        ind = np.argmax(utm_h[:,-1]) 
        utm_h[ind,:] = np.array([-1,-1,0])#Positions in dg field get close to zero, yields problems with finding closest points. Altering from [0,0,0] to [-1,-1,0] should not influence choctawhatchee bay runs.        
        ind_orig = copy.copy(ind);
        ridgex.append(x_pos[ind])
        ridgey.append(y_pos[ind])
        npd = np.sqrt( (utm_h[:,0] - utm_h_unch[ind,0])**2 + (utm_h[:,1] - utm_h_unch[ind,1])**2 ) #npd -> next_point_dist
        dist_accept = len(ma.compressed(ma.masked_where( ~( (npd > 0) & (npd < d_max) ), npd ) ) )
        new_indl = []
        while dist_accept > 0:
            utm_h[ind,:] = np.array([-1,-1,0])
            for q in new_indl: #Masking the values that were tried (failed or passed) (new_ind), but not for entire array utm_h, so these positions can be used for new ridges
                npd[q] = d_max*10
            new_ind = np.argmin(np.where( (npd > d_min) & (npd < d_max), npd, np.amax(npd)*10)) #Finding the lowest distance (npd) of a set of npd's that are greater than 0 and smaller than a maximum value.
            new_indl.append(new_ind)
            dx_prop,dy_prop,d_r_ev_av_an,d_ev_an = calc_delta_angles(ind,new_ind,vc1x,vc1y,utm_h_unch)            
            if len(ridgex) == 1: #If the first point on the ridge still has to be found, there is no previous direction
                dy_prev = dy_prop;dx_prev = dx_prop;
            if np.sign(dx_prev*dx_prop) < 0 and np.sign(dy_prev*dy_prop) < 0:
                ridge_dir_not_rotates = False
            else:
                ridge_dir_not_rotates = True                
            if d_r_ev_av_an <= d_an_r_max and d_ev_an <= d_an_ev_max and ridge_dir_not_rotates:
                ind = copy.copy(new_ind)
                ridgex.append(x_pos[ind])
                ridgey.append(y_pos[ind])
                dy_prev = copy.copy(dy_prop);copy_prev = np.copy(dx_prop)
                utm_h[ind,:] = np.array([-1,-1,0])
                npd = np.sqrt( (utm_h[:,0] - utm_h_unch[ind,0])**2 + (utm_h[:,1] - utm_h_unch[ind,1])**2 ) #npd -> next_point_dist
                for q in new_indl: 
                    npd[q] = d_max*10
                dist_accept = len(ma.compressed(ma.masked_where( ~( (npd > 0) & (npd < d_max) ), npd ) ) )    
            else:
                new_indl.append(new_ind)                
                npd = np.sqrt( (utm_h[:,0] - utm_h_unch[ind,0])**2 + (utm_h[:,1] - utm_h_unch[ind,1])**2 ) #npd -> next_point_dist
                for q in new_indl: 
                    npd[q] = d_max*10
                dist_accept = len(ma.compressed(ma.masked_where( ~( (npd > 0) & (npd < d_max) ), npd ) ) )
        utm_h[ind,:-1] = utm_h_unch[ind,:-1]
        utm_h[ind_orig,:-1] = utm_h_unch[ind_orig,:-1]
        ridgesx.append(ridgex);ridgex = []
        ridgesy.append(ridgey);ridgey = []
    return ridgesx,ridgesy
    
def link_ridge(current_ridge_x,current_ridge_y):
    new_ridge_x = [];new_ridge_y = []
    while len(current_ridge_x) > 0:
        coupled_x = [];coupled_y = []
        init_x = current_ridge_x[0][0];init_y = current_ridge_y[0][0]
        fin_x = current_ridge_x[0][-1];fin_y = current_ridge_y[0][-1]
        coupled_x.append(current_ridge_x[0]);coupled_y.append(current_ridge_y[0])
        current_ridge_x.pop(0);current_ridge_y.pop(0)
        for i in range(len(current_ridge_x)):
            new_in_x = current_ridge_x[i][0];new_in_y = current_ridge_y[i][0]
            new_fin_x = current_ridge_x[i][-1];new_fin_y = current_ridge_y[i][-1]
            if new_in_x == fin_x and new_in_y == fin_y:
                coupled_x.append(current_ridge_x[i]);coupled_y.append(current_ridge_y[i])
                current_ridge_x.pop(i);current_ridge_y.pop(i)
                break
            if new_fin_x == fin_x and new_fin_y == fin_y:
                coupled_x.append(current_ridge_x[i][::-1]);coupled_y.append(current_ridge_y[i][::-1])#reversing new ridge direction
                current_ridge_x.pop(i);current_ridge_y.pop(i)
                break
            if new_in_x == init_x and new_in_y == init_y:
                coupled_x = current_ridge_x[i][::-1] + coupled_x
                coupled_y = current_ridge_y[i][::-1] + coupled_y
                current_ridge_x.pop(i);current_ridge_y.pop(i)
                break
            if new_fin_x == init_x and new_fin_y == init_y:
                coupled_x = current_ridge_x[i] + coupled_x
                coupled_y = current_ridge_y[i] + coupled_y
                current_ridge_x.pop(i);current_ridge_y.pop(i)
                break                                
        new_ridge_x.append(flatten(coupled_x));new_ridge_y.append(flatten(coupled_y))
    return new_ridge_x,new_ridge_y
 
def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l] #don't get this line, but it works           
    
def find_eq_ind(x_eq_ind,y_eq_ind):
    for i in x_eq_ind[0]:
        for j in y_eq_ind[0]:
            if i == j:
                no_equal = False
                return i,no_equal
    no_equal = True
    return 0,no_equal
                        
def closed_gap_ridge(ridgesx,ridgesy,x_pos,y_pos,vc1x,vc1y,utm_h,d_max,d_an_r_max,d_an_ev_max,h_fld):
    new_ridgesx = copy.copy(ridgesx);new_ridgesy = copy.copy(ridgesy)
    while len(ridgesx) > 0:
        x_pos_init = ridgesx[0][0];y_pos_init = ridgesy[0][0];
        init_x = h_fld[y_pos_init,x_pos_init,0];init_y = h_fld[y_pos_init,x_pos_init,1];
        ridgesx.pop(0);ridgesy.pop(0)
#Select closest init position
        dist_pos_ar = np.zeros((len(ridgesx),3))
        for i in range(len(ridgesx)):
            x_pos_sec = ridgesx[i][0];y_pos_sec = ridgesy[i][0];
            dist = np.sqrt( ( h_fld[y_pos_sec,x_pos_sec,0] - init_x )**2 + ( h_fld[y_pos_sec,x_pos_sec,1] - init_y )**2 )
            if dist < d_max:
                dist_pos_ar[i,0] = x_pos_sec;dist_pos_ar[i,1] = y_pos_sec;dist_pos_ar[i,2] = dist
        while np.sum(dist_pos_ar[:,-1],axis=0) > 0:
            no_equal = False
            dist_pos_ar_masked = ma.masked_where(dist_pos_ar == 0,dist_pos_ar)
            arg_min_dist = np.argmin(dist_pos_ar_masked[:,-1])
            dist_pos_ar[arg_min_dist,-1] = 0
            x_pos_sec = int(dist_pos_ar[arg_min_dist,0]);y_pos_sec = int(dist_pos_ar[arg_min_dist,1])
            x_eq_ind = np.where(x_pos_init == x_pos);y_eq_ind = np.where(y_pos_init == y_pos);#finding if there are equal x values to x_pos_init, same for y. Now must see whether they form a pair
            ind,no_equal = find_eq_ind(x_eq_ind,y_eq_ind)
            if no_equal == True:
                break
            x_eq_new_ind = np.where(x_pos_sec == x_pos);y_eq_new_ind = np.where(y_pos_sec == y_pos);
            new_ind,no_equal = find_eq_ind(x_eq_new_ind,y_eq_new_ind)            
            if no_equal == True:
                break
            dx_prop,dy_prop,d_r_ev_av_an,d_ev_an = calc_delta_angles(ind,new_ind,vc1x,vc1y,utm_h)
            if d_r_ev_av_an <= d_an_r_max and d_ev_an <= d_an_ev_max:
                add_l_x = [x_pos_init,x_pos_sec];add_l_y = [y_pos_init,y_pos_sec];
                new_ridgesx.append(add_l_x);new_ridgesy.append(add_l_y)
                ridgesx.pop(arg_min_dist);ridgesy.pop(arg_min_dist)
                break
    return new_ridgesx,new_ridgesy
    
def merge_connected_ridges(ridgesx,ridgesy):   
#First combine ridges that already share a comon point
    current_ridge_x = copy.copy(ridgesx);current_ridge_y = copy.copy(ridgesy);
    ln_nr = 0;ln_cr = 1 #length new ridge and length current ridge
    while ln_nr != ln_cr:
        ln_cr = len(current_ridge_x)        #length of current ridge
        new_ridge_x,new_ridge_y = link_ridge(current_ridge_x,current_ridge_y)
        current_ridge_x = new_ridge_x;current_ridge_y = new_ridge_y        
        ln_nr = len(new_ridge_x) #length of the new ridge
    return new_ridge_x,new_ridge_y
                
def merge_ridges_main(ridgesx,ridgesy,x_pos,y_pos,vc1x,vc1y,utm_h,d_max,d_an_r_max,d_an_ev_max,h_fld):
    connected_ridges_x,connected_ridges_y = merge_connected_ridges(ridgesx,ridgesy)
    closed_gap_ridgesx,closed_gap_ridgesy = closed_gap_ridge(connected_ridges_x,connected_ridges_y,x_pos,y_pos,vc1x,vc1y,utm_h,d_max,d_an_r_max,d_an_ev_max,h_fld)
    entire_ridges_x,entire_ridges_y = merge_connected_ridges(closed_gap_ridgesx,closed_gap_ridgesy)    
    return entire_ridges_x,entire_ridges_y

def disreg_short_ridges(entire_ridges_x,entire_ridges_y): #Disregard ridges which are too short
    long_ridges_x = copy.copy(entire_ridges_x)    ;long_ridges_y = copy.copy(entire_ridges_y)
    for ind,rx in enumerate(long_ridges_x):
        if len(rx) <= 2:
            long_ridges_x.pop(ind);long_ridges_y.pop(ind)
    return long_ridges_x,long_ridges_y
    
def find_closest_point(xs,ys,x_grid,y_grid):#Needs raveled xs
    x_rav = np.ravel(xs);y_rav = np.ravel(ys)
    x_loc = np.zeros((x_grid.shape[0],x_grid.shape[1],len(x_rav))) + x_rav
    y_loc = np.zeros((y_grid.shape[0],y_grid.shape[1],len(y_rav))) + y_rav
    dists = (np.einsum('ijk -> kij',x_loc) - x_grid)**2 + (np.einsum('ijk -> kij',y_loc) - y_grid)**2
    y_inds_ar = np.zeros((dists.shape[0]));x_inds_ar = np.zeros((dists.shape[0]))
    for i in range(y_inds_ar.shape[0]):
        y_inds_ar[i],x_inds_ar[i] = np.unravel_index(np.argmin(np.ravel(dists[i,:,:])),(x_grid.shape[0],x_grid.shape[1]))
    x_inds = x_inds_ar.reshape(xs.shape); y_inds = y_inds_ar.reshape(ys.shape)  
    return x_inds,y_inds
    
def determine_angle_between_axes(set_locs_repositioned):
    cm = np.zeros((set_locs_repositioned.shape[0],set_locs_repositioned.shape[1],set_locs_repositioned.shape[3],set_locs_repositioned.shape[4]))#center of mass
    cm[:,:,:,:] = np.sum(set_locs_repositioned,axis=2)/3.
    slope_start = np.zeros((set_locs_repositioned.shape[0],set_locs_repositioned.shape[1],set_locs_repositioned.shape[3],set_locs_repositioned.shape[4]))#ridge,point on ridge,(slope,b),(old/new)
    for r in range(slope_start.shape[0]):
        for p in range(slope_start.shape[1]):
            if np.all(set_locs_repositioned[r,p,:,:,:] == np.zeros((3,2,2))): #If there are no positions, there is no ridge, hence no polyfit calculation
                break
            if np.all(set_locs_repositioned[r,p,:,0,0] == np.zeros((3))): #polyfit has problem with vertical line, this makes the line deviate slightly from vertical
                set_locs_repositioned[r,p,-1,0,0] = 0.0000000001
            try:
                (slope_start[r,p,0,0],slope_start[r,p,1,0]) =  np.polyfit(set_locs_repositioned[r,p,:,0,0],set_locs_repositioned[r,p,:,1,0],1)
                (slope_start[r,p,0,1],slope_start[r,p,1,1]) =  np.polyfit(set_locs_repositioned[r,p,:,0,1],set_locs_repositioned[r,p,:,1,1],1)
            except ValueError:
                print set_locs_repositioned[r,p,:,:,:],'set_locs_repositioned[r,p,:,:,:]'
                #sys.exit()
    rot_to_base_pre = np.arctan(slope_start[:,:,0,:])
    rot_same_dir = np.where( abs( rot_to_base_pre[:,:,0] - rot_to_base_pre[:,:,1] ) < np.pi/2.,1,0 ) #determining in what direction the initial positions should rotate, does not work for rotation greater than 90 dg. However, one could argue that there is not much protection anymore anyway.
    rot_to_base =  np.zeros((rot_to_base_pre.shape))   
    rot_to_base[:,:,0] = rot_to_base_pre[:,:,0] + np.pi/2. #adding half pi to align with vertical axis
    rot_to_base[:,:,1] = np.where(rot_same_dir.astype(bool), rot_to_base_pre[:,:,1] + np.pi/2.,rot_to_base_pre[:,:,1] - np.pi/2.)    
    set_rep_rot = np.zeros((set_locs_repositioned.shape))
    set_rep_rot[:,:,:,0,0] = np.einsum('ijk -> jki', ( np.einsum('ijk -> kij', set_locs_repositioned[:,:,:,0,0])*np.cos(rot_to_base[:,:,0]) + np.einsum('ijk -> kij',set_locs_repositioned[:,:,:,1,0])*np.sin(rot_to_base[:,:,0]) ) )
    set_rep_rot[:,:,:,1,0] = np.einsum('ijk -> jki', ( np.einsum('ijk -> kij',-set_locs_repositioned[:,:,:,0,0])*np.sin(rot_to_base[:,:,0]) + np.einsum('ijk -> kij',set_locs_repositioned[:,:,:,1,0])*np.cos(rot_to_base[:,:,0]) ) )
    set_rep_rot[:,:,:,0,1] = np.einsum('ijk -> jki', ( np.einsum('ijk -> kij', set_locs_repositioned[:,:,:,0,1])*np.cos(rot_to_base[:,:,1]) + np.einsum('ijk -> kij',set_locs_repositioned[:,:,:,1,1])*np.sin(rot_to_base[:,:,1]) ) )
    set_rep_rot[:,:,:,1,1] = np.einsum('ijk -> jki', ( np.einsum('ijk -> kij',-set_locs_repositioned[:,:,:,0,1])*np.sin(rot_to_base[:,:,1]) + np.einsum('ijk -> kij',set_locs_repositioned[:,:,:,1,1])*np.cos(rot_to_base[:,:,1]) ) )
    return set_rep_rot
 
    
def find_ridge_orthogonal_gridpoint_curvilinear(long_ridges_x,long_ridges_y,inc_num,posa_or,posa_adv,FTLE_intb,orth_dist,t,sim_name):
#Making a ridge array
    ln=0
    for r in long_ridges_x:
        if len(r) > ln:
            ln = len(r)
    if ln == 0:
        logging.warning('ln = 0, no barriers for ' + sim_name + ' time %s'%t)
        print 'ln = 0, no barriers for ' + sim_name + ' time %s'%t
        no_bar_err = True
        return 0,0,0,0,0,0,0,no_bar_err
    ridge_ar = np.zeros((len(long_ridges_x),ln,3))#all ridges with the lenght of the longest ridge, x_pos, y_pos, length of ridge at position (i,0,2)
    for ind,ridgex in enumerate(long_ridges_x):
        ridge_ar[ind,0:len(ridgex),0] = np.array(ridgex);
        ridge_ar[ind,0:len(ridgex),1] = np.array(long_ridges_y[ind])
        ridge_ar[ind,0,2] = len(ridgex)#.astype(int)
#Findging the gridpoints in the original grid and finding (interpolated positions from which to calculate ridge angle)        
    base_ind_ar = np.floor_divide(ridge_ar,inc_num)
    remainder_ind_ar = np.remainder(ridge_ar,inc_num)/float(inc_num)
    round_remainder_ind_ar = np.where(remainder_ind_ar >= 0.5,1,0)
    base_in_ar_masked = ma.masked_where(base_ind_ar == 0, base_ind_ar + round_remainder_ind_ar) #adding the remainder to get closer to the original index.
    base_in_ar_masked[:,:,0] = np.where(base_in_ar_masked[:,:,0] == 300,299,base_in_ar_masked[:,:,0])
    base_in_ar_masked[:,:,1] = np.where(base_in_ar_masked[:,:,1] == 100, 99,base_in_ar_masked[:,:,1])
    pos_for_angles = np.zeros((ridge_ar.shape))
    base_ind_ar_next = np.zeros((base_ind_ar.shape))
    base_ind_ar_next[:,:,0] = np.where(base_ind_ar[:,:,0] + 1 >= posa_or.shape[1],base_ind_ar[:,:,0],base_ind_ar[:,:,0] + 1);
    base_ind_ar_next[:,:,1] = np.where(base_ind_ar[:,:,1] + 1 >= posa_or.shape[0],base_ind_ar[:,:,1],base_ind_ar[:,:,1] + 1)#Making sure that the system does not look outside the domain of posa_or
    for i in range(pos_for_angles.shape[0]):
        pos_for_angles[i,:ridge_ar[i,0,2],0] = posa_or[base_ind_ar[i,:ridge_ar[i,0,2],1].astype(int),base_ind_ar[i,:ridge_ar[i,0,2],0].astype(int),0]*(1-remainder_ind_ar[i,:ridge_ar[i,0,2],0]) \
        + posa_or[base_ind_ar_next[i,:ridge_ar[i,0,2],1].astype(int) ,base_ind_ar_next[i,:ridge_ar[i,0,2],0].astype(int),0]*remainder_ind_ar[i,:ridge_ar[i,0,2],0]
        pos_for_angles[i,:ridge_ar[i,0,2],1] = posa_or[base_ind_ar[i,:ridge_ar[i,0,2],1].astype(int),base_ind_ar[i,:ridge_ar[i,0,2],0].astype(int),1]*(1-remainder_ind_ar[i,:ridge_ar[i,0,2],1]) \
        + posa_or[base_ind_ar_next[i,:ridge_ar[i,0,2],1].astype(int) ,base_ind_ar_next[i,:ridge_ar[i,0,2],0].astype(int) ,1]*remainder_ind_ar[i,:ridge_ar[i,0,2],1]
#Determining angle of ridge using central differences in the middle and forward/backward differences at the edges
    dp_pos = np.zeros((pos_for_angles.shape[0],pos_for_angles.shape[1]-1,2))
    for r in range(dp_pos.shape[0]):
        dp_pos[r,0:ridge_ar[r,0,2].astype(int) - 1,:] = pos_for_angles[r,1:ridge_ar[r,0,2].astype(int),:-1] - pos_for_angles[r,0:ridge_ar[r,0,2].astype(int) - 1,:-1]
    dp_fwd_cnt_bwd = np.zeros((ridge_ar.shape[0],ridge_ar.shape[1],2))
    dp_fwd_cnt_bwd[:,1:,:] += dp_pos;dp_fwd_cnt_bwd[:,:-1,:] += dp_pos;
    orient_ridge = np.arctan(dp_fwd_cnt_bwd[:,:,1]/dp_fwd_cnt_bwd[:,:,0])
    orthog_ridge = orient_ridge + np.pi/2.
    y_show=np.zeros((orient_ridge.shape[0],orient_ridge.shape[1],2))
    x_show=np.zeros((orient_ridge.shape[0],orient_ridge.shape[1],2))
    y_orthog=np.zeros((orthog_ridge.shape[0],orthog_ridge.shape[1],2))
    x_orthog=np.zeros((orthog_ridge.shape[0],orthog_ridge.shape[1],2))
    y_show[:,:,0] = -orth_dist*np.sin(orient_ridge) + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),1]
    y_show[:,:,1] = orth_dist*np.sin(orient_ridge)  + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),1]
    x_show[:,:,0] = -orth_dist*np.cos(orient_ridge) + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),0]
    x_show[:,:,1] = orth_dist*np.cos(orient_ridge)  + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),0]
    y_orthog[:,:,0] = -orth_dist*np.sin(orthog_ridge) + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),1]
    y_orthog[:,:,1] = orth_dist*np.sin(orthog_ridge)  + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),1]
    x_orthog[:,:,0] = -orth_dist*np.cos(orthog_ridge) + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),0]
    x_orthog[:,:,1] = orth_dist*np.cos(orthog_ridge)  + posa_or[base_in_ar_masked[:,:,1].astype(int),base_in_ar_masked[:,:,0].astype(int),0]   
    x_ind,y_ind = find_closest_point(x_orthog,y_orthog,posa_or[:,:,0],posa_or[:,:,1]);x_ind = x_ind.astype(int);y_ind = y_ind.astype(int)
    set_indcs = np.zeros((x_ind.shape[0],x_ind.shape[1],3,2))#ridges,points on ridge,both sides and mid (on ridge), (x,y)
    set_indcs[:,:,0,0] = x_ind[:,:,0];set_indcs[:,:,-1,0] = x_ind[:,:,1]
    set_indcs[:,:,0,1] = y_ind[:,:,0];set_indcs[:,:,-1,1] = y_ind[:,:,1]
    set_indcs[:,:,1,0] = base_in_ar_masked[:,:,0];set_indcs[:,:,1,1] = base_in_ar_masked[:,:,1]
    no_bar_err = False
    return set_indcs,orient_ridge,x_show,x_orthog,y_show,y_orthog,pos_for_angles,no_bar_err
 
def no_single_points(set_indcs,orient_ridge,x_show,x_orthog,y_show,y_orthog,pos_for_angles):
    sete_no_sin = [];orns=[];xs=[];xo=[];ys=[];yo=[];pfa=[]
    for i in range(set_indcs.shape[0]):
        if len(ma.compressed(ma.masked_where(set_indcs[i,:,1,0] == 0,set_indcs[i,:,1,0]))) > 1:
            sete_no_sin.append(set_indcs[i,:,:,:])
            orns.append(orient_ridge[i,:])
            xs.append(x_show[i,:,:])
            xo.append(x_orthog[i,:,:])
            ys.append(y_show[i,:,:])
            yo.append(y_orthog[i,:,:])            
            pfa.append(pos_for_angles[i,:,:])
    set_indcs2 = np.array([sete_no_sin])[0]
    ora = np.array([orns])[0]
    xsa = np.array([xs])[0]
    xoa = np.array([xo])[0]
    ysa = np.array([ys])[0]
    yoa = np.array([yo])[0]
    pfaa = np.array([pfa])[0]
    return set_indcs2,ora,xsa,xoa,ysa,yoa,pfaa
    
def determine_bar_ridge_point(set_indcs,posa_or,posa_adv,min_oneside_stretch_rat):  
    set_locs_err = False        
    try:
        set_locs = np.zeros((set_indcs.shape[0],set_indcs.shape[1],set_indcs.shape[2],set_indcs.shape[3],2))#same as set_indcs, last is pre and after advection
    except IndexError as e:
        print e,set_indcs.shape,'set_indcs.shape,'
        set_locs_err = True
        return 0,0,0,0,0,set_locs_err
    set_locs[:,:,:,0,0] = posa_or[np.ravel(set_indcs[:,:,:,1].astype(int)),np.ravel(set_indcs[:,:,:,0]).astype(int),0].reshape(set_indcs.shape[0],set_indcs.shape[1],set_indcs.shape[2])
    set_locs[:,:,:,1,0] = posa_or[np.ravel(set_indcs[:,:,:,1].astype(int)),np.ravel(set_indcs[:,:,:,0]).astype(int),1].reshape(set_indcs.shape[0],set_indcs.shape[1],set_indcs.shape[2])
    set_locs[:,:,:,0,1] = posa_adv[np.ravel(set_indcs[:,:,:,1].astype(int)),np.ravel(set_indcs[:,:,:,0]).astype(int),0].reshape(set_indcs.shape[0],set_indcs.shape[1],set_indcs.shape[2])
    set_locs[:,:,:,1,1] = posa_adv[np.ravel(set_indcs[:,:,:,1].astype(int)),np.ravel(set_indcs[:,:,:,0]).astype(int),1].reshape(set_indcs.shape[0],set_indcs.shape[1],set_indcs.shape[2])
# Repositioned each set to the origin
    set_locs_repositioned = np.zeros((set_locs.shape))    
    set_locs_repositioned[:,:,:,0,:] = np.einsum('ijkl -> jkil', ( np.einsum('ijkl -> kijl',set_locs[:,:,:,0,:]) - set_locs[:,:,1,0,:]))
    set_locs_repositioned[:,:,:,1,:] = np.einsum('ijkl -> jkil', ( np.einsum('ijkl -> kijl',set_locs[:,:,:,1,:]) - set_locs[:,:,1,1,:]))
# Rotating the tracers, both the ridge (final) and initial tracer sets with their own orientation
    set_rep_rot = determine_angle_between_axes(set_locs_repositioned)
# Calculating ratios etc for characterizing ridge
    set_rat_ros = np.zeros((set_rep_rot.shape[0],set_rep_rot.shape[1],7)) #ridges, point on ridge,( (2x orth stretch ratio),(2x shear ratio), dy_av, dx_av, r_centre)
    set_rat_ros[:,:,0] = set_rep_rot[:,:, 0,1,0]/set_rep_rot[:,:, 0,1,1]
    set_rat_ros[:,:,1] = set_rep_rot[:,:,-1,1,0]/set_rep_rot[:,:,-1,1,1]
    set_rat_ros[:,:,2] = set_rep_rot[:,:, 0,0,0]/set_rep_rot[:,:, 0,0,1]
    set_rat_ros[:,:,3] = set_rep_rot[:,:,-1,0,0]/set_rep_rot[:,:,-1,0,1]    
    set_rat_ros[:,:,4] = (abs( set_rep_rot[:,:,0,1,0] - set_rep_rot[:,:,0,1,1] ) + abs( set_rep_rot[:,:,-1,1,0] - set_rep_rot[:,:,-1,1,1] ))/2
    set_rat_ros[:,:,5] = (abs( set_rep_rot[:,:,0,0,0] - set_rep_rot[:,:,0,0,1] ) + abs( set_rep_rot[:,:,-1,0,0] - set_rep_rot[:,:,-1,0,1] ))/2
    set_rat_ros[:,:,6] = np.sqrt( ( set_locs[:,:,1,0,0] - set_locs[:,:,1,0,1] )**2 + ( set_locs[:,:,1,1,0] - set_locs[:,:,1,1,1] )**2 )
#Calculating factors
    set_fact = np.zeros((set_rat_ros.shape[0],set_rat_ros.shape[1],3)) #ridges, points on ridge, (sweeping, shearing, type)
    set_fact[:,:,0] = set_rat_ros[:,:,4] / set_rat_ros[:,:,-1]        
    set_fact[:,:,1] = set_rat_ros[:,:,5] / set_rat_ros[:,:,-1]
    set_fact[:,:,2] = set_rat_ros[:,:,4] / set_rat_ros[:,:,5]
#Characterize ridge: shearing or con/di- vergence
    ok_convergence = np.where(set_fact[:,:,2] > 0.2,0,2)#We are looking at convergence, if the ridge is built up from some convergence, I'll check the one/two sidedness through this. 0 for convergence, because that relates to indices :,:,0 and :,:,1 in set_rat_ros    
# Finding whether FTLE field is indcued by two sided convergence or by just one
    one_sided = np.zeros((set_rat_ros.shape[0],set_rat_ros.shape[1]))#-1 --> consider side -1, 0 consider both sides, 1, consider side 1
    for r in range(ok_convergence.shape[0]):
        for p in range(ok_convergence.shape[1]):
            if abs(set_rat_ros[r,p,0]/set_rat_ros[r,p,1]) > min_oneside_stretch_rat:
                one_sided[r,p] = 1
            if abs(set_rat_ros[r,p,0]/set_rat_ros[r,p,1]) < 1./min_oneside_stretch_rat: #not taking ok_convergence into account, because I still want to do two-sided test for shearing induced FTLE
                one_sided[r,p] = -1
#Calculating background advection independent barrier (2sided)
    set_rat_bar = np.copy(set_rat_ros[:,:,:2])
    set_rat_bar = np.where(set_rat_bar < 0,0,set_rat_bar)
    set_rat_bar = np.where( (set_rat_bar > 0) & (set_rat_bar < 1),-1,set_rat_bar)
    set_rat_bar = np.where( set_rat_bar > 1,1,set_rat_bar)
    set_bai_bar = np.where( set_rat_bar[:,:,0]*set_rat_bar[:,:,1] == 1,1,0)#This also accepts divergence as a barrier. However divergence cannot be measured, there should hence be somehting wrong.
    set_bai_bar_inc_1sided = np.where(one_sided != 0,1,set_bai_bar)
    cl = np.where(set_bai_bar_inc_1sided == 0,'red','green')
    return cl,set_bai_bar_inc_1sided,set_locs,set_locs_repositioned,set_rep_rot,set_locs_err
    
def coast_stretch_prot(set_locs,set_bai_bar_inc_1sided,num_stretch):
    sbb = set_bai_bar_inc_1sided #too long name
    x_acc_ridge = ma.compressed(ma.masked_where(np.where(sbb.astype(bool),set_locs[:,:,1,0,0],0) == 0,set_locs[:,:,1,0,0]))#Getting all accepted x positions in a flattened array
#each point in a ridge protects a coastal stretch. if at least two positions in a row are not accepted as barrier, these two are not protecting the coast.
    prot_bins = np.zeros((num_stretch,2))    
    prot_bins[:,0] = np.linspace(516950,551450,num_stretch)    #Locations are based on grid posa_or
    loop_ar = np.arange(1,num_stretch - 1,1)    
    for ind in loop_ar:
        prot_bins[ind,1] = np.where( (x_acc_ridge > prot_bins[ind-1,0]) & (x_acc_ridge < prot_bins[ind +1,0]))[0].shape[0]
    prot_bins_disc = np.zeros((prot_bins.shape))
    prot_bins_disc[:,0] = prot_bins[:,0]    
    prot_bins_disc[:,1] = np.where(prot_bins[:,1] >= 1,1,0)
    return prot_bins_disc
    
def plot(posa_or,x_show,x_orthog,y_show,y_orthog,FTLE_intb,pos_for_angles,cl,t,prot_bins_disc,save_plot,plot_save_name,fig_size,pos_r_x,pos_r_y):  
    plt.ioff()
    x_show = ma.masked_where(~np.isfinite(x_show),x_show);y_show = ma.masked_where(~np.isfinite(y_show),y_show)
    x_orthog = ma.masked_where(~np.isfinite(x_orthog),x_orthog);y_orthog = ma.masked_where(~np.isfinite(y_orthog),y_orthog)
    figure = plt.gcf()    
    figure.set_size_inches(fig_size,int(round(fig_size/np.sqrt(4))))#6
    plt.pcolor(posa_or[:,:,0],posa_or[:,:,1],FTLE_intb)#;plt.colorbar()
    for i in range(x_show.shape[0]):
        plt.scatter(pos_for_angles[i,:,0],pos_for_angles[i,:,1],c='black',s=40) 
        for j in range(len(ma.compressed(ma.masked_where(~np.isfinite(x_show[i,:,0]),x_show[i,:,0])))):
            plt.plot(x_show[i,j,:],y_show[i,j,:],c=cl[i,j],linewidth=5)
            plt.plot(x_orthog[i,j,:],y_orthog[i,j,:],c='black',linewidth=5)#c=cl2[q],linewidth=5)
      
    for r_ind,r in enumerate(pos_r_x):
        plt.plot(r,pos_r_y[r_ind])

    plt.axis('equal')        
    plt.xlim([516750,551700]);plt.ylim([3349000,3363000])
    
    plt.title('t = %s'%format(t,'.3f'))
    for ind,x in enumerate(prot_bins_disc[:,0]):
        if prot_bins_disc[ind,1] == 1:
            col = 'green'
        else:
            col = 'red'
        plt.scatter(x,3362800,c=col)
    plt.xlabel(r'Easting $\times 10^{5} m$ ',fontsize=30);plt.ylabel(r'Northing $\times 10^{6} m$',fontsize=30)
    plt.xticks([520000,530000,540000,550000],['5.2','5.3','5.4','5.5'],fontsize=20)
    plt.yticks([3350000,3354000,3358000,3362000],[3.35,3.354,3.358,3.362],fontsize=20)
    if save_plot == False:
        plt.show()
        return
    if save_plot == True:
        print 'saving figure'
        plt.savefig(plot_save_name,dpi=200)        
        plt.cla();plt.clf();plt.close()
    return
    
def plot_dg(posa_or,x_show,x_orthog,y_show,y_orthog,FTLE_intb,pos_for_angles,cl,t,prot_bins_disc,save_plot,plot_save_name,fig_size,vc1,set_indcs):  
    plt.ioff()
    plt.figure()
    x_show = ma.masked_where(~np.isfinite(x_show),x_show);y_show = ma.masked_where(~np.isfinite(y_show),y_show)
    x_orthog = ma.masked_where(~np.isfinite(x_orthog),x_orthog);y_orthog = ma.masked_where(~np.isfinite(y_orthog),y_orthog)
    figure = plt.gcf()    
    figure.set_size_inches(fig_size,fig_size)#int(round(fig_size/np.sqrt(6))))
    figure.suptitle('FTLE field of double gyre flow, advection time %s'%t)
    plt.subplot(2,1,1)
    plt.pcolor(posa_or[:,:,0],posa_or[:,:,1],FTLE_intb)#;plt.colorbar()
    plt.xlim([0,2]);plt.ylim([0,1])    
    plt.title('FTLE field')
    plt.xlabel('x');plt.ylabel('y')    
    plt.subplot(2,1,2)
    plt.title('FTLE field overlayed with transport barriers')
    plt.pcolor(posa_or[:,:,0],posa_or[:,:,1],FTLE_intb)#;plt.colorbar()    
    for i in range(x_show.shape[0]):
        plt.scatter(pos_for_angles[i,:,0],pos_for_angles[i,:,1],c='black') 
        for j in range(len(ma.compressed(ma.masked_where(~np.isfinite(x_show[i,:,0]),x_show[i,:,0])))):
            plt.plot(x_show[i,j,:],y_show[i,j,:],c=cl[i,j],linewidth=5)
            plt.plot(x_orthog[i,j,:],y_orthog[i,j,:],c='black',linewidth=5)#c=cl2[q],linewidth=5)
    plt.xlim([0,2]);plt.ylim([0,1])
    plt.xlabel('x');plt.ylabel('y')
    if save_plot == False:
        plt.show()
        return
    if save_plot == True:
        print 'saving figure'
        plt.savefig(plot_save_name,dpi=200)        
        plt.cla();plt.clf();plt.close()
    return

   
def main_BAR_PROT(basefolder,runmap,sim_name,t_adv_end,trac_len,trac_wdth,posa_or,sm,perc,lim_lab1,lim_GdotZ,inc_num,d_box,d_an_r_max,d_an_ev_max,perc_num_disr,d_max,orth_dist,min_oneside_stretch_rat,num_stretch,save_plot,fig_size):
    inlmsk = np.zeros((100,300));inlmsk[-5:,:] = 1;inlmsk[92:-1,250:266] = 1;
    posa_orig = np.copy(posa_or)
    posa_or = posa_or.reshape(trac_wdth,trac_len,2)    
    for index_t,t in enumerate(t_adv_end):
        tbs = int(t);tdec = t-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving/loading (time base and time decimal)
        FTLE_or = np.load(basefolder + runmap + 'FTLE_field_%s_t_%s_%s.npy'%(sim_name,tbs,tdec)) #Loading FTLE
        posa_adv = np.load(basefolder + runmap + 'aft_adv_%s_ti%s_%s_BT.npy'%(sim_name,tbs,tdec)).reshape(trac_wdth,trac_len,2)
        FTLE_intb,FTLE_intb_nomask = interp_bands_msk_inlet(FTLE_or,inlmsk) #Interpolating bands of FTLE = 0 values due to block calculation
        h_fld,h_fld_interp_no_smooth,sm_err = prep_rect_smoothed_height_field(posa_or,FTLE_intb,inc_num,sm,trac_len,trac_wdth)#,FTLE_peaks_msk)
        if sm_err == True:
            continue        
        sm_adj = np.copy(sm)#If the heightfield is wrong due to smoothing, the smoothing paramater is increased untill the heightfield is reasonable        
        while np.amax(h_fld[:,:,-1])/np.amax(FTLE_intb) >= 100:
            sm_adj += 20000
            print 'adjusting smoothening factor'
            h_fld,h_fld_interp_no_smooth,sm_err = prep_rect_smoothed_height_field(posa_or,FTLE_intb,inc_num,sm_adj,trac_len,trac_wdth)
            if sm_err == True:
                break
        if sm_err == True:
            continue
        vl_min,GdotZ,vl,vc1 = calc_grad_hes_vel_vec(h_fld[:,:,-1])
        tot_msk = only_peaks_all_req(h_fld[:,:,-1],vl_min,GdotZ,perc,lim_lab1,lim_GdotZ,h_fld_interp_no_smooth[:,:,-1])
        tot_msk_orig = np.copy(tot_msk)
        x_pos,y_pos,vc1x,vc1y,utm_h = find_local_peaks(h_fld,tot_msk,d_box,vc1,h_fld_interp_no_smooth)
        utm_h,x_pos,y_pos,vc1x,vc1y = grad_check(utm_h,x_pos,y_pos,vc1x,vc1y,perc_num_disr)
        utm_h_orig = np.copy(utm_h)
        d_min = 0.1 #minimum distance to accept next point on ridge, hard coded here to avoid hard coding in all main files
        ridgesx,ridgesy = create_ridge_req_SR(x_pos,y_pos,vc1x,vc1y,utm_h,h_fld,d_max,d_min,d_an_r_max,d_an_ev_max)
        utm_h = utm_h_orig        
        ridgesx_or = ridgesx[:];ridgesy_or = ridgesy[:]
        entire_ridges_x,entire_ridges_y = merge_ridges_main(ridgesx,ridgesy,x_pos,y_pos,vc1x,vc1y,utm_h_orig,d_max,d_an_r_max,d_an_ev_max,h_fld)
        long_ridges_x,long_ridges_y = disreg_short_ridges(entire_ridges_x,entire_ridges_y)

        pos_r_x = [];pos_r_y = []
        for ind_r,r in enumerate(long_ridges_x):
            rx = [];ry = []
            for ind_p,p in enumerate(r):
                #print ind_r,ind_p,'ind_r,ind_p,',len(r),len(long_ridges_y[ind_r]),r[ind_p],'r'
                rx.append(posa_or[int(round( (long_ridges_y[ind_r][ind_p]) /8.)),int(round(r[ind_p]) / 8.),0])
                ry.append(posa_or[int(round(long_ridges_y[ind_r][ind_p]/8.)),int(round(r[ind_p]) / 8.),1])
            pos_r_x.append(rx);pos_r_y.append(ry)
                

        print sim_name#long_ridges_x,'long_ridges_x',len(long_ridges_x),'len'
        var = perc
        set_indcs,orient_ridge,x_show,x_orthog,y_show,y_orthog,pos_for_angles,no_bar_err = find_ridge_orthogonal_gridpoint_curvilinear(long_ridges_x,long_ridges_y,inc_num,posa_or,posa_adv,FTLE_intb,orth_dist,t,sim_name)
        if no_bar_err == True:
            print 'no_bar_err = True, go to next field (%s,%s)'%(sim_name,t)
            continue
        
        set_indcs,orient_ridge,x_show,x_orthog,y_show,y_orthog,pos_for_angles = no_single_points(set_indcs,orient_ridge,x_show,x_orthog,y_show,y_orthog,pos_for_angles)        

        cl,set_bai_bar_inc_1sided,set_locs,set_locs_repositioned,set_rep_rot,set_locs_err = determine_bar_ridge_point(set_indcs,posa_or,posa_adv,min_oneside_stretch_rat)     
        if set_locs_err == True:
            print 'set_locs_err = True, go to next field (%s,%s)'%(sim_name,t)
            continue
        prot_bins_disc = coast_stretch_prot(set_locs,set_bai_bar_inc_1sided,num_stretch)
        np.save(basefolder + runmap + 'prot_bins_%s_t_%s_%s.npy'%(sim_name,tbs,tdec),prot_bins_disc)        
        np.save(basefolder + runmap + 'set_indcs_%s_t_%s_%s.npy'%(sim_name,tbs,tdec),set_indcs)   #non-barrier positions must be deleted from set_indcs before saving
        plot_save_name = basefolder + runmap + 'FTLE_realbar_prot_%s_t_%s_%s.png'%(sim_name,tbs,tdec)
        plot(posa_or,x_show,x_orthog,y_show,y_orthog,FTLE_intb_nomask,pos_for_angles,cl,t,prot_bins_disc,save_plot,plot_save_name,fig_size,pos_r_x,pos_r_y)
        
    posa_or = np.copy(posa_orig) #should not be necessary, but if memory leaks the systme will crash, this prevents that.
    return 
  
 
def main_BAR_PROT_dg(basefolder,runmap,sim_name,t_adv_end,trac_len,trac_wdth,posa_or,sm,perc,lim_lab1,lim_GdotZ,inc_num,d_box,d_an_r_max,d_an_ev_max,perc_num_disr,d_max,orth_dist,min_oneside_stretch_rat,num_stretch,save_plot,fig_size,time_dir,wd_bn):
    posa_orig = np.copy(posa_or)
    posa_or = posa_or.reshape(trac_wdth,trac_len,2)  
    t_adv = t_adv_end[0]
    tbs = int(t_adv_end[0]);tdec = t_adv_end[0]-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving/loading (time base and time decimal)
    FTLE_or = np.load(basefolder + runmap + 'FTLE_field_%s_t_%s_%s.npy'%(sim_name,tbs,tdec)) #Loading FTLE
    posa_adv = np.load(basefolder + runmap + 'aft_adv_%s_ti%s_%s_%s.npy'%(sim_name,tbs,tdec,time_dir)).reshape(trac_wdth,trac_len,2)
    inlmsk = np.zeros((FTLE_or.shape));inlmsk[:,:wd_bn] = 1;inlmsk[:,-wd_bn:] = 1;inlmsk[:wd_bn,:] = 1;inlmsk[-wd_bn:,:] = 1;
    FTLE_intb,FTLE_intb_nomask = interp_bands_msk_inlet(FTLE_or,inlmsk)
    h_fld,h_fld_interp_no_smooth,sm_err = prep_rect_smoothed_height_field(posa_or,FTLE_intb,inc_num,sm,trac_len,trac_wdth)#;print np.amax(h_fld[:,:,-1])#,FTLE_peaks_msk)
    if sm_err == True:
        print 'sm_err =',sm_err
        sys.exit()
    sm_adj = float(np.copy(sm))#If the heightfield is wrong due to smoothing, the smoothing paramater is increased untill the heightfield is reasonable        
    while np.amax(h_fld[:,:,-1])/np.amax(FTLE_intb) >= 100:
        sm_adj *= 1.3
        print 'adjusting smoothening factor'
        h_fld,h_fld_interp_no_smooth,sm_err = prep_rect_smoothed_height_field(posa_or,FTLE_intb,inc_num,sm_adj,trac_len,trac_wdth)
        if sm_err == True:
            break
    if sm_err == True:
        print 'sm_err =',sm_err
        sys.exit()
    print sm_adj,'sm_adj',np.amax(h_fld[:,:,-1]),np.amax(FTLE_intb),'np.amax(h_fld[:,:,-1]),np.amax(FTLE_intb)'
    vl_min,GdotZ,vl,vc1 = calc_grad_hes_vel_vec(h_fld[:,:,-1])
    tot_msk = only_peaks_all_req(h_fld[:,:,-1],vl_min,GdotZ,perc,lim_lab1,lim_GdotZ,h_fld_interp_no_smooth[:,:,-1])
    tot_msk_orig = np.copy(tot_msk)
    x_pos,y_pos,vc1x,vc1y,utm_h = find_local_peaks(h_fld,tot_msk,d_box,vc1,h_fld_interp_no_smooth)
    utm_h,x_pos,y_pos,vc1x,vc1y = grad_check(utm_h,x_pos,y_pos,vc1x,vc1y,perc_num_disr)
    utm_h_orig = np.copy(utm_h)
    d_min = 0.01
    ridgesx,ridgesy = create_ridge_req_SR(x_pos,y_pos,vc1x,vc1y,utm_h,h_fld,d_max,d_min,d_an_r_max,d_an_ev_max)
    utm_h = utm_h_orig        
    ridgesx_or = ridgesx[:];ridgesy_or = ridgesy[:]
    entire_ridges_x,entire_ridges_y = merge_ridges_main(ridgesx,ridgesy,x_pos,y_pos,vc1x,vc1y,utm_h_orig,d_max,d_an_r_max,d_an_ev_max,h_fld)
    long_ridges_x,long_ridges_y = disreg_short_ridges(entire_ridges_x,entire_ridges_y)
    print sim_name#long_ridges_x,'long_ridges_x',len(long_ridges_x),'len'
    var = perc
    set_indcs,orient_ridge,x_show,x_orthog,y_show,y_orthog,pos_for_angles,no_bar_err = find_ridge_orthogonal_gridpoint_curvilinear(long_ridges_x,long_ridges_y,inc_num,posa_or,posa_adv,FTLE_intb,orth_dist,t_adv,sim_name)
    if no_bar_err == True:
        print 'no_bar_err = True, go to next field (%s,%s)'%(sim_name,t)
        
    cl,set_bai_bar_inc_1sided,set_locs,set_locs_repositioned,set_rep_rot = determine_bar_ridge_point(set_indcs,posa_or,posa_adv,min_oneside_stretch_rat)     
    prot_bins_disc = coast_stretch_prot(set_locs,set_bai_bar_inc_1sided,num_stretch)
    np.save(basefolder + runmap + 'prot_bins_%s_t_%s_%s.npy'%(sim_name,tbs,tdec),prot_bins_disc)        
     
    np.save(basefolder + runmap + 'set_indcs_%s_t_%s_%s.npy'%(sim_name,tbs,tdec),set_indcs)   
    plot_save_name = basefolder + runmap + 'FTLE_realbar_prot_%s_t_%s_%s.png'%(sim_name,tbs,tdec)
    plot_dg(posa_or,x_show,x_orthog,y_show,y_orthog,FTLE_intb_nomask,pos_for_angles,cl,t_adv,prot_bins_disc,save_plot,plot_save_name,fig_size,vc1,set_indcs)
    posa_or = np.copy(posa_orig) #should not be necessary, but if memory leaks the systme will crash, this prevents that.
    return     
