#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
import utilities as ut
import radvel
from PyAstronomy import pyasl

def pyarome(t, t0, aRs, per, ecc, i, omega, lda, u1, u2,  Vsini, Rp, beta0, sigma0, zeta, units='radian'):
    """
	Function to calculate and return the value of
	stellar radial velocity (RV) anomaly due to a
	transiting exoplanet, i.e the Rossiter-McLaughlin
	effect (RME).  This function is written only with
    the quadratic limb darkening in mind.
	
	Python translation of the ARoME library.
	http://www.astro.up.pt/resources/arome/
	
	Inputs:
	-------
	t: float
	time of observation
	
    t0:float
    mid-transit time
    
	aRs: float
	Scaled semi-major axis
    
    per: float
    Period of the orbit, same units as t
    
    ecc: float
    eccentricity of the orbit
	
	i: float
	Orbital inclination in degrees or radians
	
	lam: float
	Obliquity (Spin-Orbit) angle in degrees or radians
	
	u1, u2: floats 
	Limb darkening coefficients for the quadratic law (Kopal 1950)
	I_mu / I_0 = 1 - ld1 * (1-u) - ld2 * (1-u)**2 
	
	Vsini: float
	Projected stellar rotation velocity at the equator in km/s
	
	Rp: float
	Radius of the planet in solar radius
	
	beta0: float
	FWHM of a delta function, essentially the instrumental profile in km/s 
	(defaults to ESPRESSO. For HARPS: beta0 = 2.6km/s)
	
	sigma0: float
	FWHM of the CCF in km/s
	
	zeta: float
	Macroturbulence velocity in km/s
	set to -1 to ignore it.
		
	units: string
	The unit used for all the input angles
	Possible values: 'degree' or 'radian' (default)
	
	
	Output:
	--------
	
	rv_ccf: float
	the RV during transit, i.e. the RM effect measured by
	the CCF technique in km/s
	
	"""
    if units not in ['degree', 'radian']:
        raise ValueError('The units should be either of "degree" or "radian".')
    elif units == 'degree':
        i            *= np.pi/180.
        lda          *= np.pi/180.
        omega        *= np.pi/180.
    
    Tper       = radvel.orbit.timetrans_to_timeperi(t0, per, ecc, omega)
    ke         = pyasl.KeplerEllipse(a=aRs, per=per, e=ecc, i=i*180./np.pi, Omega=0., w=omega*180./np.pi, tau=Tper)    

    ecc_anom   = ke.eccentricAnomaly(t)
    true_anom  = 2.*np.arctan(np.sqrt((1.+ecc)/(1.-ecc))*np.tan(ecc_anom/2.))
    radius_vec = aRs * (1.-ecc**2)/(1.+ecc*np.cos(true_anom))
    node       = lda    # this is np.pi+lda in the C-function?!?! 
    true_lat   = omega + ke.trueAnomaly(t)

    x_ = radius_vec*(-np.cos(node)*np.cos(true_lat)+np.sin(node)*np.sin(true_lat)*np.cos(i))
    y_ = radius_vec*(-np.sin(node)*np.cos(true_lat)-np.cos(node)*np.sin(true_lat)*np.cos(i))
    z_ = radius_vec*np.sin(true_lat)*np.sin(i)
        
    rho   = np.sqrt(x_**2+y_**2) 
    dmin  = rho-Rp #minimal distance between the planet ellipse and the origin
    dmax  = rho+Rp #maximal distance between the planet ellipse and the origin

    # when the planet is behind the star
    if z_ <= 0.: return 0.
    # when the planet does not overlap the stellar disk
    elif dmin >= 1. - 1e-10: return 0.
    #planet transiting
    else:
        dic = {'Vsini':Vsini,'Rp':Rp,'beta0':beta0,'sigma0':sigma0,'zeta':zeta}
        
        # limb darkening parameters
        coefficients, powers, kernel_coeffs = arome_alloc_quad(u1,u2)
        dic['coefficients']                 = coefficients
        dic['powers']                       = powers
        dic['kernel_coeffs']                = kernel_coeffs
        dic['LD_order']                     = int(2)
        
        dic['gauss_a0']                     = setGaussfit_a0(dic)
            
        dic                                 = arome_calc_fvpbetap(x_,y_,z_,dic)
        
        return arome_get_RM_CCF_e(dic)    
    
    
def arome_alloc_quad(u1,u2):
    """
    Translation of "arome_alloc_quad" C function
    """
    denom         = np.pi*(1.0-u1/3.0-u2/6.0)
    #quadratic limb darkening coefficients for pyarome
    coeffs        = [(1.0-u1-u2)/denom,(u1+2.0*u2)/denom,-u2/denom]
    powers        = [0,2,4]
    
    kernel_coeffs = setrotkernel(coeffs, powers)
    
    return coeffs, powers, kernel_coeffs


def setrotkernel(coefficients, powers):
    """
    "setrotkernel" C function
    """
    
    Im0, Im1, Im2 = 2.0, 2.3962804694711844, np.pi
    Ip1           = 1.7480383695280799
    
    ntabI = 4
    for i in range(len(powers)):
        ntabI = max(ntabI, powers[i]+4)
    
    tabI         = np.zeros(int(ntabI), float)
    tabI[-2]     = Im2
    tabI[-1]     = Im1
    tabI[0]      = Im0
    tabI[1]      = Ip1
    
    for i in range(2,int(ntabI)-2):
        tabI[i]  = i / (i+2) * tabI[i-4]
    
    kernel_coeffs = [coefficients[i]*tabI[int(powers[i])] for i in range(len(powers))]
    
    return kernel_coeffs
    
    
    
def setGaussfit_a0(dic, lim=20):
    """
    "setGaussfit_a0" C function    
    """
    integral = 4.0*dic['sigma0']*np.sqrt(np.pi)*integrate.quad(funcAmp_a0,0.,1.,dic,lim)[0]
    return integral
    

def funcAmp_a0(x, dic):
    """
    "funcAmp_a0" C function
    """
    
    sig2 = dic['sigma0']**2+dic['beta0']**2+dic['zeta']**2/2.
    mu   = np.sqrt(1-x**2)
    smu  = np.sqrt(mu)
    
    # Rotational kernel
    Rx   = 0.
    for i in range(2): # in future could be extended to other LD functions
        Rx += dic['kernel_coeffs'][i]*smu**dic['powers'][i]
    Rx  *= mu

    # Gaussian kernel
    Gx   = 1./np.sqrt(2*np.pi*sig2)*np.exp(-(x*dic['Vsini'])**2 / (2*sig2) )
    
    return Rx*Gx




def arome_calc_fvpbetap(x,y,z,dic):
    """
    translation of the C function "arome_calc_fvpbetap" that
    calculates the flux f, subplanet velocity vp and dispersion
    betapR and betapT.
    
    """

    # case 1: planet is behind the star
    if z <= 0.:
        dic['flux']   = 0.
        dic['vp']     = 0.
        dic['betapR'] = dic['beta0']
        dic['betapT'] = dic['beta0']
        
        return dic

    # planet parameters
    rx, ry, r         = dic['Rp'],dic['Rp'],dic['Rp']
    phi0              = np.arctan2(y, x)
    rho               = np.sqrt(x**2+y**2)
    dmin              = rho-r
    dmax              = rho+r
    
    # case 2: planet does not overlap the stellar disk
    if dmin >= 1.-1e-10:
        dic['flux']   = 0.
        dic['vp']     = 0.
        dic['betapR'] = dic['beta0']
        dic['betapT'] = dic['beta0']
        
        return dic
        

    # case 3: planet completely inside the stellar disk
    elif dmax <= 1.:
        xbar          = x
        ybar          = y
        a00           = np.pi*rx*ry
        axx           = rx**2/4.
        ayy           = ry**2/4.
        axy           = 0.

    else : #during ingress and egress 
		
		#stellar boundary
        psi   = np.arccos((1.0+rho**2-r**2)/(2.0*rho))
        phi1  = phi0-psi
        phi2  = phi0+psi
				
        a00  = ut.funcF00(0.0,0.0,1.0,1.0,phi2)-ut.funcF00(0.0,0.0,1.0,1.0,phi1)
        xbar = ut.funcF10(0.0,0.0,1.0,1.0,phi2)-ut.funcF10(0.0,0.0,1.0,1.0,phi1)
        ybar = ut.funcF01(0.0,0.0,1.0,1.0,phi2)-ut.funcF01(0.0,0.0,1.0,1.0,phi1)
        axx  = ut.funcF20(0.0,0.0,1.0,1.0,phi2)-ut.funcF20(0.0,0.0,1.0,1.0,phi1)
        ayy  = ut.funcF02(0.0,0.0,1.0,1.0,phi2)-ut.funcF02(0.0,0.0,1.0,1.0,phi1)
        axy  = ut.funcF11(0.0,0.0,1.0,1.0,phi2)-ut.funcF11(0.0,0.0,1.0,1.0,phi1)
		
		# planet boundary
        psi   = np.arccos(-1.*(1.0-rho**2-r**2)/(2.0*r*rho))
        phi1  = phi0+np.pi-psi
        phi2  = phi0+np.pi+psi
		
        a00  += (ut.funcF00(x,y,rx,ry,phi2)-ut.funcF00(x,y,rx,ry,phi1))
        xbar += (ut.funcF10(x,y,rx,ry,phi2)-ut.funcF10(x,y,rx,ry,phi1))
        ybar += (ut.funcF01(x,y,rx,ry,phi2)-ut.funcF01(x,y,rx,ry,phi1))
        axx  += (ut.funcF20(x,y,rx,ry,phi2)-ut.funcF20(x,y,rx,ry,phi1))
        ayy  += (ut.funcF02(x,y,rx,ry,phi2)-ut.funcF02(x,y,rx,ry,phi1))
        axy  += (ut.funcF11(x,y,rx,ry,phi2)-ut.funcF11(x,y,rx,ry,phi1))
		
        xbar /= a00
        ybar /= a00
        axx   = axx/a00 - xbar**2
        ayy   = ayy/a00 - ybar**2
        axy   = axy/a00 - xbar*ybar
		
    II = ut.funcIxn(xbar,ybar,dic,0)
    Hxx0,Hyy0,Hxy0 = ut.HessIxn(xbar,ybar,dic,0)
    ff = a00*(II+0.5*(Hxx0*axx+Hyy0*ayy+2.0*Hxy0*axy))
	
    Hxx1,Hyy1,Hxy1 = ut.HessIxn(xbar,ybar,dic,1)
    Hxx1 -= xbar*Hxx0
    Hyy1 -= xbar*Hyy0
    Hxy1 -= xbar*Hxy0
    vv    = xbar + 0.5/II*(Hxx1*axx+Hyy1*ayy+2.0*Hxy1*axy)
	
    Hxx2,Hyy2,Hxy2 = ut.HessIxn(xbar,ybar,dic,2)

    Hxx2 -= xbar**2*Hxx0
    Hyy2 -= xbar**2*Hyy0
    Hxy2 -= xbar**2*Hxy0
    v2    = xbar**2 + 0.5/II*(Hxx2*axx+Hyy2*ayy+2.0*Hxy2*axy)
	
	# results
	
    dic['flux'] = ff
    dic['vp']   = vv
    dbetaR         = np.sqrt(v2-vv**2)
    dbetaT         = dbetaR
	
	# set the units
	
    dic['vp']    *= dic['Vsini']
    dbetaR       *= dic['Vsini']
    dbetaT       *= dic['Vsini']

    if dic['zeta']>0.0 : #take into account macro turbulence
        powers = np.zeros(dic['LD_order'], float)
        mu2bar = 1.0-xbar**2-ybar**2
		
        for i in range(dic['LD_order']):
            powers[i] = dic['powers'][i]
            dic['powers'][i] += 4
		
        Hxx2,Hyy2,Hxy2 = ut.HessIxn(xbar,ybar,dic,0)
        Hxx2 -= mu2bar*Hxx0
        Hyy2 -= mu2bar*Hyy0
        Hxy2 -= mu2bar*Hxy0
        zetaR2 = mu2bar + 0.5/II*(Hxx2*axx+Hyy2*ayy+2.0*Hxy2*axy)
        zetaT2 = 1.0-zetaR2
		
        zetaR2 *= dic['zeta']**2
        zetaT2 *= dic['zeta']**2
		
        dbetaR = np.sqrt(dbetaR**2+zetaR2)
        dbetaT = np.sqrt(dbetaT**2+zetaT2)
		
		# retrieve the initial limb-darkening law
        for i in range(dic['LD_order']):
            dic['powers'][i] = powers[i]
		
	
	# add to the width of the non-rotating star
    dic['betapR'] = np.sqrt(dbetaR**2+dic['beta0']**2)
    dic['betapT'] = np.sqrt(dbetaT**2+dic['beta0']**2)
	
    return dic
    
    
def arome_get_RM_CCF_e(dic):
    """
    translation of "arome_get_RM_CCF_e" C function
    Computes the RM effect measured by the CCF technique.
    v = 1/den * (2*sig0**2/(sig0**2+betap**2))**(3/2)*f*vp*exp(-vp**2/(2*(sig0**2+betap**2)))
    """ 
    den   = dic['gauss_a0']
    f     = dic['flux']
    vp    = dic['vp']
    bs2   = dic['sigma0']**2
    bpR2  = dic['betapR']**2
    bpT2  = dic['betapT']**2
    
    if f<0.: return 0.
    # with macro turbulence
    if dic['zeta']>0.:
        return -0.5/den*(2.*bs2/(bs2+bpR2))**(3./2)*f*vp*np.exp(-vp**2/(2.0*(bs2+bpR2)))-0.5/den*(2.*bs2/(bs2+bpT2))**(3./2)*f*vp*np.exp(-vp*vp/(2.0*(bs2+bpT2)))
    #without macro turbulence
    else:
        return -1.0/den*(2.0*bs2/(bs2+bpR2))**(3./2)*f*vp*np.exp(-vp*vp/(2.0*(bs2+bpR2)))
