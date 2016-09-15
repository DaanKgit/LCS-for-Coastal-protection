# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 15:46:51 2016

@author: dkuitenbrouwer
This file manages the tracer advection for both forward and backward time

inputs:                 flowfield (name in folder), 
                        advection time, 
                        days of advection, 
                        timestep
                        
hard-coded:             Extended grid name                        
                        
outputs:                paths (saved)
                        error if something goes wrong                        
"""

#Pre-ambule
import numpy as np
import scipy
import matplotlib.pyplot as plt
import prep_vel as pv
import logging
import sys

def load_vel_dat(basefolder,runmap,sim_name,t,time_dir):  #Loading D3D prepared D3D data FT or BT, specific set etc.
    UD3D = np.load(basefolder + runmap + 'u_int_%s_ti%s.npy'%(sim_name,int(t)))
    VD3D = np.load(basefolder + runmap + 'v_int_%s_ti%s.npy'%(sim_name,int(t)))
    TD3D = np.load(basefolder + runmap + 't_int_%s_ti%s.npy'%(sim_name,int(t)))
    if time_dir == 'BT':#For backwards time we start at the end and go to the begin, hence order is reversed
        UD3D = UD3D[::-1,:,:]
        VD3D = VD3D[::-1,:,:]
        TD3D = TD3D[::-1]
    return UD3D,VD3D,TD3D
    
def law_cosines(ln_prt,lt_prt,ln_sr1,lt_sr1,ln_sr2,lt_sr2): #Based on the law of cosines, the angle between the tracer and two neighbouring gridpoints within one gridcell is calculated for all combinations. (4x4=16)
    a = np.sqrt( (ln_sr1 - ln_prt)**2 + (lt_sr1 - lt_prt)**2 )
    b = np.sqrt( (ln_sr2 - ln_prt)**2 + (lt_sr2 - lt_prt)**2 )
    c = np.sqrt( (ln_sr1 - ln_sr2)**2 + (lt_sr1 - lt_sr2)**2 )
    gamma = np.arccos( (a**2 + b**2 - c**2)/(2*a*b) )
    return gamma
    
def tot_an(lnp,ln1,ln2,ln3,ln4,ltp,lt1,lt2,lt3,lt4):#Calculating the sum of angles for each grid cell. where each separate angle is calculated and discussed in law_cosines
    lns = np.array([ln1,ln2,ln3,ln4])               #l(n/t)(1,2,3,4) have shape (4,ntracers) each combination of ln(i)[x,trac],lt(i)[x,trac] with i E (1 - 4) spans up a grid cell. x refers to what grid cell. x=0 is grid cell left upper in zcl,ecl space (rotation counter clockwise) and right lower in UTM space (rotation clockwise). This is because the D3D coordinate system goes 'up (N),left (E)' directions are between brackets as system is curvilinear so it is mostly aligned with these axes, but not entirely. Python makes use of 'right,down', hence both rotation and positions are swapped.
    lts = np.array([lt1,lt2,lt3,lt4])
    an = law_cosines(lnp,ltp,lns,lts,np.roll(lns,-1,axis=0),np.roll(lts,-1,axis=0));an_tot = np.sum(an,axis=0) #calculating the angles and summing the angles per grid cell. If a tracer is within a gridcell, the angles sum up to 360 dg.
    return an_tot    

def FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac):
# Finding the closest gridpoints based on former location of particle.
    x_loc = np.zeros((3,3,posat.shape[0])) + posat[:,0];
    y_loc = np.zeros((3,3,posat.shape[0])) + posat[:,1];
    nxa = np.isnan(xall);xall[nxa] = 1000000000.; nya = np.isnan(yall);yall[nya] = 1000000000. #Finding closest gridpoint is based on distances. Nan is considered zero distance. By giving nans high values, distances are always large, so gridpoints will never be closest.
    zcl = zcl.astype(int);ecl = ecl.astype(int)    
    try:  #Indexerrors may occur. Only one try except statement to increase velocity
        x_surp = np.array([    [ xall[zcl+1,ecl+1],xall[zcl+1,ecl],xall[zcl+1,ecl-1] ], \
                               [ xall[zcl  ,ecl+1],xall[zcl  ,ecl],xall[zcl  ,ecl-1] ], \
                               [ xall[zcl-1,ecl+1],xall[zcl-1,ecl],xall[zcl-1,ecl-1] ]  ]) #surrounding x locations before new gridpoints
    
        y_surp = np.array([    [ yall[zcl+1,ecl+1],yall[zcl+1,ecl],yall[zcl+1,ecl-1] ], \
                               [ yall[zcl  ,ecl+1],yall[zcl  ,ecl],yall[zcl  ,ecl-1] ], \
                               [ yall[zcl-1,ecl+1],yall[zcl-1,ecl],yall[zcl-1,ecl-1] ]  ]) #surrounding y locations before new gridpoints
        pyth = (x_loc - x_surp)**2 + (y_loc - y_surp)**2 #Calculating the distance between the tracer and each of its 9 surrounding gridpoints. This is done for all tracers at the same time, shape = (3,3,ntracers)
        pyth_rav = pyth.reshape(9,nm_trac);min_rev = np.argmin(pyth_rav,axis=0);marg,narg = np.unravel_index(min_rev,(3,3))#reshape to (9,ntracers), find closest gridpoint (9) for each tracer (ntracers). Lastly find index in x/y_surp setup
        zcl = zcl - marg + 1;ecl = ecl - narg +1 #New gridpositions (zcl/ecl) based on old one. -m/narg +1 because ecl, zcl counting is in opposite direction to python counting. The +1 is because if ecl does not change (middle) python counting gives it +1
  
# Define four possible surrounding cells  
        x_sur = np.array([    [ xall[zcl+1,ecl+1],xall[zcl+1,ecl],xall[zcl+1,ecl-1] ], \
                              [ xall[zcl  ,ecl+1],xall[zcl  ,ecl],xall[zcl  ,ecl-1] ], \
                              [ xall[zcl-1,ecl+1],xall[zcl-1,ecl],xall[zcl-1,ecl-1] ]  ]) #surrounding x locations before new gridpoints

        y_sur = np.array([    [ yall[zcl+1,ecl+1],yall[zcl+1,ecl],yall[zcl+1,ecl-1] ], \
                              [ yall[zcl  ,ecl+1],yall[zcl  ,ecl],yall[zcl  ,ecl-1] ], \
                              [ yall[zcl-1,ecl+1],yall[zcl-1,ecl],yall[zcl-1,ecl-1] ]  ]) #surrounding y locations before new gridpoints
    except IndexError as e:
        logging.debug('e = False, %s'%(e))
        return np.array([[[False]]]),zcl + marg -1,ecl + narg -1
 
# Calculate total angle going round
    b = np.array([1,1,1,1]);e11 = np.array([2,2,0,0]);e22=np.array([2,0,0,2])#defining 4 surrounding grid cells based on base (b and x_sur,y_sur)
    an_tot = tot_an(posat[:,0],x_sur[b,b],x_sur[e11,b],x_sur[e11,e22],x_sur[b,e22],posat[:,1],y_sur[b,b],y_sur[e11,b],y_sur[e11,e22],y_sur[b,e22])#Calculating sum of 4 angles from tracer to 4 possible surrounding gridpoints
    region = np.argmax(an_tot,axis=0);#defining region depending on highest sum of angles. number region is given in figure in appendix (starting left down, rotation clockwise)

# Find UTM location of four surrounding grid points for each tracer  
    pr0 = np.where(region == 0);pr1 = np.where(region == 1); pr2 = np.where(region == 2); pr3 = np.where(region ==3)
    n = np.zeros((posat.shape[0],4,2));
    n[pr0[0],:,0] = np.einsum('ij -> ji',np.array([zcl[pr0[0]],zcl[pr0[0]],zcl[pr0[0]]-1,zcl[pr0[0]]-1]))#Changed region redirection, problem is hopefully in this.
    n[pr0[0],:,1] = np.einsum('ij -> ji',np.array([ecl[pr0[0]],ecl[pr0[0]]-1,ecl[pr0[0]]-1,ecl[pr0[0]]]))
    n[pr1[0],:,0] = np.einsum('ij -> ji',np.array([zcl[pr1[0]],zcl[pr1[0]],zcl[pr1[0]]-1,zcl[pr1[0]]-1]))
    n[pr1[0],:,1] = np.einsum('ij -> ji',np.array([ecl[pr1[0]]-1,ecl[pr1[0]],ecl[pr1[0]],ecl[pr1[0]]-1]))  
    n[pr2[0],:,0] = np.einsum('ij -> ji',np.array([zcl[pr2[0]]+1,zcl[pr2[0]]+1,zcl[pr2[0]],zcl[pr2[0]]]))
    n[pr2[0],:,1] = np.einsum('ij -> ji',np.array([ecl[pr2[0]]-1,ecl[pr2[0]],ecl[pr2[0]],ecl[pr2[0]]-1]))
    n[pr3[0],:,0] = np.einsum('ij -> ji',np.array([zcl[pr3[0]]+1,zcl[pr3[0]]+1,zcl[pr3[0]],zcl[pr3[0]]]))
    n[pr3[0],:,1] = np.einsum('ij -> ji',np.array([ecl[pr3[0]],ecl[pr3[0]]+1,ecl[pr3[0]]+1,ecl[pr3[0]]]))    
    return n,zcl,ecl    
    
def InterpVelToTrac(x,y,n,vals,xall,yall,e,t,f):#e refers to the set that is interpolated e.g. (u1,v1,u2...). It is added to be able to locate errors.
    #start with recalculating every time. Think about saving withou for loop later Could do something with first finding the arguments in tpi_c_ar where the value is zero and only work out for those. np.linalg.solve happens to be quite fast, so don't think it's neccesary
    E = np.zeros((x.shape[0],6,6));Epre = np.identity(6);Epre[3,3]=0;Epre[4,4]=0;Epre[5,5]=0; E = E + Epre#part of identity matrix
    n = n.astype(int);
    X = np.array([  [xall[n[:,0,0],n[:,0,1]]**2,xall[n[:,0,0],n[:,0,1]]*yall[n[:,0,0],n[:,0,1]],yall[n[:,0,0],n[:,0,1]]**2,xall[n[:,0,0],n[:,0,1]],yall[n[:,0,0],n[:,0,1]],np.ones(x.shape[0])],  \
                    [xall[n[:,1,0],n[:,1,1]]**2,xall[n[:,1,0],n[:,1,1]]*yall[n[:,1,0],n[:,1,1]],yall[n[:,1,0],n[:,1,1]]**2,xall[n[:,1,0],n[:,1,1]],yall[n[:,1,0],n[:,1,1]],np.ones(x.shape[0])],  \
                    [xall[n[:,2,0],n[:,2,1]]**2,xall[n[:,2,0],n[:,2,1]]*yall[n[:,2,0],n[:,2,1]],yall[n[:,2,0],n[:,2,1]]**2,xall[n[:,2,0],n[:,2,1]],yall[n[:,2,0],n[:,2,1]],np.ones(x.shape[0])],  \
                    [xall[n[:,3,0],n[:,3,1]]**2,xall[n[:,3,0],n[:,3,1]]*yall[n[:,3,0],n[:,3,1]],yall[n[:,3,0],n[:,3,1]]**2,xall[n[:,3,0],n[:,3,1]],yall[n[:,3,0],n[:,3,1]],np.ones(x.shape[0])]   ])
    z = np.array([vals[n[:,0,0],n[:,0,1]],vals[n[:,1,0],n[:,1,1]],vals[n[:,2,0],n[:,2,1]],vals[n[:,3,0],n[:,3,1]]]); z = np.einsum('ij -> ji',z)
    A = np.zeros((x.shape[0],10,10));A[:,0:6,0:6]=E[:,:,:];A[:,0:6,6:10]=np.einsum('ijk -> kji',X);A[:,6:10,0:6]=np.einsum('ijk -> kij',X)
    B = np.zeros((x.shape[0],10)); B[:,6:10] = z
    try:
        g = np.linalg.solve(A,B);a=g[:,0:6]
    except np.linalg.linalg.LinAlgError as lae:

        logging.warning('%s for %s, at t = %s, using former f'%(lae,e,t))
        return f,f
    f=a[:,0]*x**2 + a[:,1]*x*y + a[:,2]*y**2 + a[:,3]*x + a[:,4]*y + a[:,5]#the interpolation function evaluated at the desired location (particle)
    return f,f
    
def RK4_1step(posat,dtRK4,xall,yall,uvals,vvals,zcl,ecl,t,vel_ar,nm_trac): 
    err = False
    #print dtRK4,'dtRK4'
    x1=np.copy(posat[:,0]);  y1=np.copy(posat[:,1]); n,zcl1,ecl1 = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac) #Determining the indices utm positions of gridcells
    if n[0,0,0] == False:
        logging.debug('break1')
        err = True
        return posat,zcl,ecl,err,vel_ar
    e_u = 'u1';e_v = 'v1'        
    u1,vel_ar[:,0,0] = InterpVelToTrac(x1,y1,n,uvals,xall,yall,e_u,t,vel_ar[:,0,0]); v1,vel_ar[:,0,1]=InterpVelToTrac(x1,y1,n,vvals,xall,yall,e_v,t,vel_ar[:,0,1])#Interpolating the velocity to the position of the tracer and therefore calculating its velocity
    u1 = np.where(u1 == 999999,np.nan,u1); v1 = np.where(v1 == 999999,np.nan,v1);
    
    x2=x1+0.5*dtRK4*u1;  y2=y1+0.5*dtRK4*v1; posat[:,0] = x2; posat[:,1] = y2; n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl1,ecl1,nm_trac) #Determining new intermediate positions of tracers to find their velocity at these positions
    if n[0,0,0] == False:
        logging.debug('break2')
        err = True
        return posat,zcl,ecl,err,vel_ar  
    e_u = 'u2';e_v = 'v2'        
    u2,vel_ar[:,1,0] = InterpVelToTrac(x2,y2,n,uvals,xall,yall,e_u,t,vel_ar[:,1,0]); v2,vel_ar[:,1,1]=InterpVelToTrac(x2,y2,n,vvals,xall,yall,e_v,t,vel_ar[:,1,1])
    u2 = np.where(u2 == 999999,np.nan,u2); v2 = np.where(v2 == 999999,np.nan,v2);

    x3=x1+0.5*dtRK4*u2;  y3=y1+0.5*dtRK4*v2; posat[:,0] = x3; posat[:,1] = y3;  n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac)
    if n[0,0,0] == False:
        logging.debug('break3')
        err = True
        return posat,zcl,ecl,err,vel_ar
    e_u = 'u3';e_v = 'v3'        
    u3,vel_ar[:,2,0] = InterpVelToTrac(x3,y3,n,uvals,xall,yall,e_u,t,vel_ar[:,2,0]); v3,vel_ar[:,2,1]=InterpVelToTrac(x3,y3,n,vvals,xall,yall,e_v,t,vel_ar[:,2,1])
    u3 = np.where(u3 == 999999,np.nan,u3); v3 = np.where(v3 == 999999,np.nan,v3);

    x4=x1+    dtRK4*u3;  y4=y1+    dtRK4*v3; posat[:,0] = x3; posat[:,1] = y3; n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac)
    if n[0,0,0] == False:
        logging.debug('break4')
        err = True
        return posat,zcl,ecl,err,vel_ar   
    e_u = 'u4';e_v = 'v4'        
    u4,vel_ar[:,3,0] = InterpVelToTrac(x4,y4,n,uvals,xall,yall,e_u,t,vel_ar[:,3,0]); v4,vel_ar[:,3,1]=InterpVelToTrac(x4,y4,n,vvals,xall,yall,e_v,t,vel_ar[:,3,1])
    u4 = np.where(u4 == 999999,np.nan,u4); v4 = np.where(v4 == 999999,np.nan,v4);

    #print u1,u2,u3,u4,v1,v2,v3,v4,'u1,u2,u3,u4,v1,v2,v3,v4, grid interp'
   # print x1,x1+dtRK4/6.*(u1+2.*u2+2.*u3+u4),'x1,x1+dtRK4/6.*(u1+2.*u2+2.*u3+u4)'
    posat[:,0]=x1+dtRK4/6.*(u1+2.*u2+2.*u3+u4)
    posat[:,1]=y1+dtRK4/6.*(v1+2.*v2+2.*v3+v4)
    
    n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl1,ecl1,nm_trac)   
    
    if np.amax(ecl) == 258:
        logging.debug('ecl max = 258')
        err = True
        print 'ecl max = 258'
        return posat,zcl,ecl,err,vel_ar
    return posat,zcl,ecl,err,vel_ar

def main_RK4_unsteady(basefolder,runmap,sim_name,ndays,time_dir,t_adv_end,dtRK4,trac_grid,ad_t,posa_or):
    #Loading grid data and original tracer positions
    if trac_grid == '30000shore':
        xall_ext = np.load( basefolder + 'x_all_extended.npy')
        yall_ext = np.load( basefolder + 'y_all_extended.npy')
        nm_trac = posa_or.shape[0]
        zcl_or = np.load( basefolder + 'zcl30000_cls.npy');ecl_or = np.load( basefolder + 'ecl30000_cls.npy') 
    
    minYD = 1./(60*24)
    TD3D = np.array([1000])#Giving TD3D a high value so at the first iteration it will start looking for the right timeseries
    #Looping through the entire timeseries with at each step the posat reset
    if time_dir == 'BT':
        t_adv_end = t_adv_end[::-1]
        t_adv = -ad_t/(60*24.)                     #Determining the advection time, this should be negative for backward time
        dtRK4 *= -1                 
        print time_dir,'time_dir'
    if time_dir == 'FT':
        t_adv = ad_t/(60*24.)                     #Determining the advection time, this should be positive for forward time        
        print time_dir,'time_dir'
    for ind,t in enumerate(t_adv_end):                          #In this loop every time a new advection gets started
        print 't =',t,'set = ', sim_name
        logging.info('t = %s, set = %s'%(t,sim_name))
        posat = np.copy(posa_or);zcl = np.copy(zcl_or);ecl = np.copy(ecl_or) #Resetting original positions
        t_int_ar = np.linspace(t - t_adv,t - dtRK4/(3600.*24), abs((t_adv*3600*24)/dtRK4) -1 )#Creating internal loop timeseries
        vel_ar = np.zeros((zcl.shape[0],4,2))#If a singular matrix is found in the interpolation method, the former velocity set (for each tracer for each of the 4 RK4 parts) is used to see whether the singular matrix occurs in the next timestep as well. Note, this is a workaround, it does not solve the origin of the problem.

        
        for t_int in t_int_ar:                                  #Performing all timesteps within one time integration
            print t_int,'t_int'
            if int(t-0.00000001) != int(np.amin(TD3D)):         #Checking whether the flow field file must be reset, the minus to make sure that the last step is still day 5
                UD3D,VD3D,TD3D = load_vel_dat(basefolder,runmap,sim_name,t,time_dir) #Loading new velocity field if necessary
                
            k = np.argmin(abs(TD3D - t_int))                    #finding the index (time) of the flow field
            
            posat,zcl,ecl,err,vel_ar = RK4_1step(posat,dtRK4,xall_ext,yall_ext,UD3D[k,:,:],VD3D[k,:,:],zcl,ecl,t,vel_ar,nm_trac)
            if err == True:
                logging.warning('stopped before end, at %s out of %s'%(t_int,t_int_ar))
                break
        tbs = int(t);tdec = t-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving
        np.save(basefolder + runmap + 'aft_adv_%s_ti%s_%s_%s'%(sim_name,tbs,tdec,time_dir),posat)
    return
    
def main_RK4_steady(basefolder,runmap,sim_name,time_dir,t_adv,dtRK4,Windage,Uwnd,Vwnd,trac_grid,ln,rnd,ntrac,posa_or):
#Load velocity field on grid and initial tracer positions
    xall = np.load(basefolder + 'xx_dg_ln_%s_rand_%s.npy'%(ln,int(rnd)))
    yall = np.load(basefolder + 'yy_dg_ln_%s_rand_%s.npy'%(ln,int(rnd)))
    uu = np.load(basefolder + 'uu_dg_ln_%s_rand_%s.npy'%(ln,int(rnd)))
    vv = np.load(basefolder + 'vv_dg_ln_%s_rand_%s.npy'%(ln,int(rnd)))
    posat = np.copy(posa_or)
    zcl = np.load(basefolder + 'zcl_dg_ln%s_rand_%s.npy'%(ln,int(rnd)))
    ecl = np.load(basefolder + 'ecl_dg_ln%s_rand_%s.npy'%(ln,int(rnd))) 
    vel_ar = np.zeros((zcl.shape[0],4,2))#If a singular matrix is found in the interpolation method, the former velocity set (for each tracer for each of the 4 RK4 parts) is used to see whether the singular matrix occurs in the next timestep as well. Note, this is a workaround, it does not solve the origin of the problem.

    print 'here'
    if time_dir == 'BT':
        uu = -uu;vv=-vv#Swapping around velocities. Still actual time does not really matter.
    

#Loop through timesteps
    t = 0
    while t < t_adv:
        print t,'t'
        posat,zcl,ecl,err,vel_ar = RK4_1step(posat,dtRK4,xall,yall,uu,vv,Windage,Uwnd,Vwnd,zcl,ecl,t,vel_ar,ntrac)
        t += dtRK4
    tbs = int(t_adv);tdec = t-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving
    np.save(basefolder + runmap + 'aft_adv_%s_ti%s_%s_%s'%(sim_name,tbs,tdec,time_dir),posat)
    return    
    
def dg_vel_on_grid(xx,yy,t,B,gamma,omega):
    b = 1-2*gamma*np.sin(omega*t)
    c = gamma*np.sin(omega*t)
    h = c*xx**2 + b*xx
    dhdx = (np.roll(h,-1,axis=1) - h) / (np.roll(xx,-1,axis=1) - xx)
    uu = -np.pi*B*np.sin(np.pi*h)*np.cos(np.pi*yy)
    vv =  np.pi*B*np.cos(np.pi*h)*np.sin(np.pi*yy)*dhdx
    return uu,vv

def main_RK4_dg_unsteady(basefolder,runmap,time_dir,t_adv,dtRK4,sim_name,xall,yall,posa_or,zcl,ecl,B,gamma,omega,ntrac):
#Load velocity field on grid and initial tracer positions
    posat = np.copy(posa_or)
    vel_ar = np.zeros((zcl.shape[0],4,2))#If a singular matrix is found in the interpolation method, the former velocity set (for each tracer for each of the 4 RK4 parts) is used to see whether the singular matrix occurs in the next timestep as well. Note, this is a workaround, it does not solve the origin of the problem.

    print 'here'


#Loop through timesteps
    t = 0
    while t < t_adv:
        print t,'t'
        uu,vv = dg_vel_on_grid(xall,yall,t,B,gamma,omega)
        if time_dir == 'BT':
            uu = -uu;vv=-vv#Swapping around velocities. Still actual time does not really matter.
        posat,zcl,ecl,err,vel_ar = RK4_1step(posat,dtRK4,xall,yall,uu,vv,zcl,ecl,t,vel_ar,ntrac)
        t += dtRK4
    tbs = int(t_adv);tdec = t_adv-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving
    np.save(basefolder + runmap + 'aft_adv_%s_ti%s_%s_%s'%(sim_name,tbs,tdec,time_dir),posat)
    return    

def h_def(x,b,c):
    h = c*x**2 + b*x
    return h

def uv_def(x,y,t,B,gamma,omega,time_dir):
    b = 1-2*gamma*np.sin(omega*t)
    c = gamma*np.sin(omega*t)
    h = h_def(x,b,c)
    dhdx = (h_def(x+0.1,b,c) - h_def(x - 0.1,b,c)) / (0.2)
    u = -np.pi*B*np.sin(np.pi*h)*np.cos(np.pi*y)
    v = np.pi*B*np.cos(np.pi*h)*np.sin(np.pi*y)*dhdx
    if time_dir == 'BT':
        u =  np.pi*B*np.sin(np.pi*h)*np.cos(np.pi*y)
        v = -np.pi*B*np.cos(np.pi*h)*np.sin(np.pi*y)*dhdx
    return u,v
    
def RK4_1step_analytic_flow(posat,dtRK4,B,gamma,omega,time_dir,t): 
    err = False
    #print dtRK4,'dtRK4'
    x1=posat[:,0];  y1=posat[:,1];
    u1,v1 = uv_def(x1,y1,t,B,gamma,omega,time_dir)
    
    x2=x1+0.5*dtRK4*u1;  y2=y1+0.5*dtRK4*v1; 
    u2,v2 = uv_def(x2,y2,t,B,gamma,omega,time_dir)

    x3=x1+0.5*dtRK4*u2;  y3=y1+0.5*dtRK4*v2;
    u3,v3 = uv_def(x3,y3,t,B,gamma,omega,time_dir)

    x4=x1+    dtRK4*u3;  y4=y1+    dtRK4*v3;
    u4,v4 = uv_def(x4,y4,t,B,gamma,omega,time_dir)
    
    #print u1,u2,u3,u4,v1,v2,v3,v4,'u1,u2,u3,u4,v1,v2,v3,v4, analytical'
    #print x1,x1+dtRK4/6.*(u1 + 2.*u2 + 2.*u3 + u4),'x1,x1+dtRK4/6.*(u1 + 2.*u2 + 2.*u3 + u4)'
    
    posat[:,0]=x1+dtRK4/6.*(u1 + 2.*u2 + 2.*u3 + u4)
    posat[:,1]=y1+dtRK4/6.*(v1 + 2.*v2 + 2.*v3 + v4)

    return posat
    
def main_RK4_dg_unsteady_analytic_flow(basefolder,runmap,time_dir,t_adv,dtRK4,sim_name,posa_orig,B,gamma,omega,ntrac):
#Load velocity field on grid and initial tracer positions
    posat_ana = np.copy(posa_orig)

#Loop through timesteps
    t = 0
    while t < t_adv:
        print t,'t'
        posat_ana = RK4_1step_analytic_flow(posat_ana,dtRK4,B,gamma,omega,time_dir,t)
        t += dtRK4
    tbs = int(t_adv);tdec = t_adv-tbs;tdec = format(tdec,'.4f');tdec=tdec[2:6] #splitting time up in strings without dots for saving
    np.save(basefolder + runmap + 'aft_adv_%s_ti%s_%s_%s'%(sim_name,tbs,tdec,time_dir),posat_ana)
    return      
    
    
    
    
    
    
    
    
#Method to avoid crash when index error in location finder
'''
def RK4_1step(posat,dtRK4,xall,yall,uvals,vvals,Windage,Uw,Vw,zcl,ecl,t,vel_ar,nm_trac): #In the original files there is a method to avoid the simulation to crash when the interpolation scheme finds a singular matrix, this method is given below. 
    err = False
    x1=posat[:,0];  y1=posat[:,1]; n,zcl1,ecl1 = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac)
    if n[0,0,0] == False:
        logging.debug('break1')
        err = True
        return posat,zcl,ecl,err,vel_ar
    e_u = 'u1';e_v = 'v1'        
    u1,vel_ar[:,0,0] = InterpVelToTrac(x1,y1,n,uvals,xall,yall,e_u,t,vel_ar[:,0,0]); v1,vel_ar[:,0,1]=InterpVelToTrac(x1,y1,n,vvals,xall,yall,e_v,t,vel_ar[:,0,1])
    u1 = np.where(u1 == 999999,np.nan,u1); v1 = np.where(v1 == 999999,np.nan,v1);

    x2=x1+0.5*dtRK4*u1;  y2=y1+0.5*dtRK4*v1; posat[:,0] = x2; posat[:,1] = y2; n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl1,ecl1,nm_trac)
    if n[0,0,0] == False:
        logging.debug('break2')
        err = True
        return posat,zcl,ecl,err,vel_ar  
    e_u = 'u2';e_v = 'v2'        
    u2,vel_ar[:,1,0] = InterpVelToTrac(x2,y2,n,uvals,xall,yall,e_u,t,vel_ar[:,1,0]); v2,vel_ar[:,1,1]=InterpVelToTrac(x2,y2,n,vvals,xall,yall,e_v,t,vel_ar[:,1,1])
    u2 = np.where(u2 == 999999,np.nan,u2); v2 = np.where(v2 == 999999,np.nan,v2);

    x3=x1+0.5*dtRK4*u2;  y3=y1+0.5*dtRK4*v2; posat[:,0] = x3; posat[:,1] = y3;  n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac)
    if n[0,0,0] == False:
        logging.debug('break3')
        err = True
        return posat,zcl,ecl,err,vel_ar
    e_u = 'u3';e_v = 'v3'        
    u3,vel_ar[:,2,0] = InterpVelToTrac(x3,y3,n,uvals,xall,yall,e_u,t,vel_ar[:,2,0]); v3,vel_ar[:,2,1]=InterpVelToTrac(x3,y3,n,vvals,xall,yall,e_v,t,vel_ar[:,2,1])
    u3 = np.where(u3 == 999999,np.nan,u3); v3 = np.where(v3 == 999999,np.nan,v3);

    x4=x1+    dtRK4*u3;  y4=y1+    dtRK4*v3; posat[:,0] = x3; posat[:,1] = y3; n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl,ecl,nm_trac)
    if n[0,0,0] == False:
        logging.debug('break4')
        err = True
        return posat,zcl,ecl,err,vel_ar   
    e_u = 'u4';e_v = 'v4'        
    u4,vel_ar[:,3,0] = InterpVelToTrac(x4,y4,n,uvals,xall,yall,e_u,t,vel_ar[:,3,0]); v4,vel_ar[:,3,1]=InterpVelToTrac(x4,y4,n,vvals,xall,yall,e_v,t,vel_ar[:,3,1])
    u4 = np.where(u4 == 999999,np.nan,u4); v4 = np.where(v4 == 999999,np.nan,v4);

    posat[:,0]=x1+dtRK4/6.*(u1+2.*u2+2.*u3+u4) #+ Uw*Windage*dtRK4
    posat[:,1]=y1+dtRK4/6.*(v1+2.*v2+2.*v3+v4) #+ Vw*Windage*dtRK4 
    n,zcl,ecl = FindClGrdPntAndCellForTrac(posat,xall,yall,zcl1,ecl1,nm_trac)      


    if np.amax(ecl) == 258:
        logging.debug('ecl max = 258')
        err = True
        return posat,zcl,ecl,err,vel_ar

        
    return posat,zcl,ecl,err,vel_ar
    

cl = ['b','r','m','y']
for i in range(4):
    plt.scatter(ln1[i,0],lt1[i,0],c=cl[i]);plt.scatter(ln4[i,0],lt4[i,0],c=cl[i]);plt.scatter(ln2[i,0],lt2[i,0],c=cl[i]);plt.scatter(ln3[i,0],lt3[i,0],c=cl[i]);
    plt.scatter(lnp[0],ltp[0],c='r')
    plt.scatter(lns[i,:,0],lts[i,:,0],c='y')        

#'''    
            

