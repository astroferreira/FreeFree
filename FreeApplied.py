
# coding: utf-8

# In[3]:


#Libraries
import numpy as np
import time
import math as mt

#------------------------------------------------------------------------------------------------------------
#Call Freefree Function
from FreeFunction import freefree
from FreeFunction import set_frequencies

#------------------------------------------------------------------------------------------------------------
#Constants

mh   = 1.67372e-24 
#h = 6.62606885e-27 #Planck's constant in CGS
#k = 1.3807e-16 #Boltzmann's constant in CGS
cl   = 2.99792458e10
#------------------------------------------------------------------------------------------------------------
#Calls and convert input file

C = np.loadtxt('cat.001.100.dat', unpack = True)

Crc = C[0]*cl #r/c 
Cpr = C[1]*mh*(cl**2)  #pressure/mh/c^2,
Cvx = C[2]*cl #vx/c, 
Cvy = C[3]*cl #vy, 
Cvz = C[4]*cl #vz, 
Cbx = C[5]*cl #bx, 
Cby = C[6]*cl #by, 
Cbz = C[7]*cl #bz, 
Cde = C[8]*mh #density/mh
#------------------------------------------------------------------------------------------------------------
#proton number

Zn = 1.
#------------------------------------------------------------------------------------------------------------
#range frequencies

nfreq = 2
nu_min = 1e11
nu_max = 1e15

nu = set_frequencies(nu_min,nu_max,nfreq)
#------------------------------------------------------------------------------------------------------------
#distance from the source

dist = 3.086e18*15.1e6 #cm
#------------------------------------------------------------------------------------------------------------
#difference between distances: creates the cells for R = 0. 
#These cells will be projected over positions outside the axis

Crc2 = np.insert(Crc, 0, 0)
Crc2 = np.delete(Crc2, -1)
dR = Crc - Crc2
#------------------------------------------------------------------------------------------------------------

#Cylindrical coordinates
R = Crc
#R = np.insert(R, 0, 0)
Z = Crc
Z = np.insert(Z, 0, 0)

print(len(dR), len(Z))


# In[4]:


#----------------------------------------------------------------------------------------------------------
#Runs the freefree function over each cell 1D and saves it

#Fluxo0 = np.zeros([len(Crc),nfreq])

t1 = time.time()


#iterates over Z
I0 = np.zeros([len(Crc),nfreq])
stacked_results = np.zeros_like(I0)
for i in range(len(Z)-1):
    #r of the cell
    r = np.sqrt(Z[i+1]**2+R**2)
    
    #condition, so it doenst calculate outside the data
    theta = np.arcsin(Z[i+1]/r) #angle 
    #Cell size
    Cell = (Z[i+1] - Z[i])*np.cos(theta)

    
    #Find the index for values of r in the data
    g = np.where(Crc <= r)
    g = g[0][-1]
    
    #FreeFree Function
    nu = nu.reshape([2, 1])

    I_ff, F_ff, F, tau_ff = freefree(Cde, Cpr, R, dR, Cell, I0[i-1].reshape([2, 1]), Zn, nu, dist)
    I0 = I_ff.reshape([I0.shape[0], I0.shape[1]])
    stacked_results = np.dstack((stacked_results, I0))
#------------------------------------------------------------------------------------------------------------
#Show if the for is finished and measure its time    
print('ok')
t2 = time.time()
tt = t2-t1
print(stacked_results.shape)
print(tt, 'seconds')
#END OF THE CODE
#------------------------------------------------------------------------------------------------------------

