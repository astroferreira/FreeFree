from __future__ import division
import numpy as np
import mpmath as mp

#Constants
mh   = 1.67372e-24 #hydrogen mass CGS
h = 6.62606885e-27 #Planck's constant in CGS
k = 1.3807e-16 #Boltzmann's constant in CGS
cl   = 2.99792458e10 #speed of light CGS

#------------------------------------------------------------------------------------------------------------    
#This function increases accuracy for calculation
def DC(V):
    if type(V) is np.ndarray:
        return V.astype(np.float64)
    else:
        V = np.array([V]).astype(np.float64)
        return V


#---------------------------------------------------------------------------------------------------------------    
#range frequencies
def set_frequencies(nu_min,nu_max,nfreq):
    ifraq = np.linspace(0, nfreq, nfreq)  
    nu = nu_min*(nu_max/nu_min)**(ifraq/nfreq)
    return nu

#Variables:

#-------------------------------------------------------------------------------------------------
#Physical:

#de = density,
#pr = pressure, Z = atomic number
#I0 = intensity from previous cell

#-------------------------------------------------------------------------------------------------
#Frequency range and resolution:

#nfreq = number of frequencies, nu_min = min frequency, nu_max = max frequency

#-------------------------------------------------------------------------------------------------
#Geometrical Variables:
#dist = distance from the source
#Cells = Difference between distances. It's function of cylindrical Z and its projection
#R = cylindrical coordinate, dR = differential R

#------------------------------------------------------------------------------------------------
#FreeFree Function. Where modules are applied:

def freefree(de, pr, R, dR, Cells, I0, Z, nu, dist): 
    #Creating Modules to apply on the FreeFree Function
    #defines electron and ion density from general density 
    def density(de):
        ne = DC(0.2*de) #Electron Density
        ni = DC(0.8*de) #Ion Density
        return ne, ni

    #---------------------------------------------------------------------------------------------------------------    
    #where goes the gaunt factor function (quantum correction), might be changed in the future
    def GauntFactor():
        G = 1
        return G

    #---------------------------------------------------------------------------------------------------------------    
    #Temperatura for an ideal gas in CGS
    def EOSIdeal(pr, de):
        mu = 1.
        T = DC(mu*pr*mh/(de*k))
        return T

    #---------------------------------------------------------------------------------------------------------------
    #Emission and Absorption Coeficient
    def find_jff_and_aff(nu, Z, ne, ni, T, h, k, G):  
        H =-h*nu/(k*T)
        mH = np.mean(H)
        Eff = np.zeros(nu.shape[0])

        ga = DC(G*(6.8e-38*(Z*Z)*ne*ni*(T**-0.5)))
        ct1 = DC(3.7e8*(Z**2.)*ne*ni*(nu**-3.)*T**(-0.5)*G)

        if abs(mH) > 1e-8:
            Eff = DC(ga*np.exp(H))
            aff = ct1*(1. - np.exp(H))
        else:
            Eff = DC(ga*(1 + H))
            aff = DC(ct1*(-H))

        jff = Eff/(4.*np.pi)
        return (jff, aff)
    #---------------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------------
    #Intensity
    def emission(de, nu, Cells, aff, jff):
        dz = Cells            # set the size of the emitting region in each interaction
        tau_ff = DC(aff*dz)
        H = -tau_ff
        mtau = np.mean(H)

        if abs(mtau) > 1e-8:

            I_ff = DC(jff/aff)*(1. - np.exp(H)) + (I0*np.exp(H))

        else:
            I_ff = DC((jff/aff)*(-H)) + (I0*(1+H))

        return I_ff, tau_ff

    #---------------------------------------------------------------------------------------------------------------
    #Net Flux
    def Net_Flux(I_ff, dist, R, dR):
        F1_ff = np.zeros(I_ff.shape)
        F1_ff = DC(I_ff*2.*np.pi*R*dR)
        F1_ff = DC(F1_ff/(dist**2))
        return F1_ff
    #--------------------------------------------------------------------------------------------------------------- 
    #Flux
    def Flux(F1_ff, nu):
        F = np.trapz(F1_ff.T, nu.T)
        return F
    #--------------------------------------------------------------------------------------------------------------- 
    
#Apply modules

    #Gaunt Factor
    G = GauntFactor()
    #Temperature of Ideal Gas EOS
    T = EOSIdeal(pr, de)
    #Electron and Ions densities
    ne, ni = density(de)
    #Emission and Absorption coeficient
    (jff, aff) = find_jff_and_aff(nu, Z, ne, ni, T, h, k, G)
    #Intensity
    I_ff, tau_ff = emission(de, nu, Cells, aff, jff)
    #Net Flux
    F_ff = Net_Flux(I_ff, dist, R, dR)
    #Flux    
    F = Flux(F_ff, nu)
    
    return I_ff, F_ff, F, tau_ff
#---------------------------------------------------------------------------------------------------------------    

