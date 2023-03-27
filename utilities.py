#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def funcF00(x0, y0, rx, ry, phi):
	return 0.5*(rx*ry*phi+x0*ry*np.sin(phi)-y0*rx*np.cos(phi))


def funcF10(x0,  y0, rx, ry, phi):
	return -x0*y0*rx*np.cos(phi)\
		-0.5*y0*rx**2*(np.cos(phi))**2\
		+0.25*x0*rx*ry*(2.0*phi-np.sin(2.0*phi))\
		+1.0/12.0*rx**2*ry*(3.0*np.sin(phi)-np.sin(3.0*phi))


def funcF01(x0, y0, rx, ry, phi):
	return  x0*y0*ry*np.sin(phi)\
		+0.5*x0*ry**2*(np.sin(phi))**2\
		+0.25*y0*rx*ry*(2.0*phi+np.sin(2.0*phi))\
		-1.0/12.0*rx*ry**2*(3.0*np.cos(phi)+np.cos(3.0*phi))


def funcF20(x0, y0, rx, ry, phi):
	return -1.*x0**2*y0*rx*np.cos(phi)\
		-x0*y0*rx**2*(np.cos(phi))**2\
		+0.25*x0**2*rx*ry*(2.0*phi-np.sin(2.0*phi))\
		-1.0/12.0*y0*rx**3*(3.0*np.cos(phi)+np.cos(3.0*phi))\
		+1.0/6.0*x0*rx**2*ry*(3.0*np.sin(phi)-np.sin(3.0*phi))\
		+1.0/32.0*rx**3*ry*(4.0*phi-np.sin(4.0*phi))


def funcF02(x0, y0, rx, ry, phi):
	return  x0*y0**2*ry*np.sin(phi)\
		+x0*y0*ry**2*(np.sin(phi))**2\
		+0.25*y0**2*rx*ry*(2.0*phi+np.sin(2.0*phi))\
		+1.0/12.0*x0*ry**3*(3.0*np.sin(phi)-np.sin(3.0*phi))\
		-1.0/6.0*y0*rx*ry**2*(3.0*np.cos(phi)+np.cos(3.0*phi))\
		+1.0/32.0*ry**3*rx*(4.0*phi-np.sin(4.0*phi))


def funcF11(x0, y0, rx, ry, phi):
	return 0.25*x0*y0*(2.0*rx*ry*phi+x0*ry*np.sin(phi)-y0*rx*np.cos(phi))\
		+0.125*(x0*ry*np.sin(phi))**2\
		-0.125*(y0**2+ry**2)*(rx*np.cos(phi))**2\
		+1.0/48.0*y0*rx**2*ry*(15.0*np.sin(phi)-np.sin(3*phi))\
		-1.0/48.0*x0*rx*ry**2*(15.0*np.cos(phi)+np.cos(3*phi))


def funcIxn(x0, y0, dic, n):
	return x0**n*funcLimb(x0,y0,dic)


def funcLimb(x0, y0, dic):
	R   = x0**2+y0**2
	mus = (1.0-R)**(1/4)
	out = 0.
	for i in range(2):
		out += dic['coefficients'][i]*mus**dic['powers'][i]
	return out


def HessIxn(x0, y0, dic, n):
	xn            = x0**n
	
	L             = funcLimb(x0,y0,dic)
	Lx, Ly        = dfuncLimb(x0,y0,dic)
	Lxx, Lyy, Lxy = ddfuncLimb(x0,y0,dic)
		
	Hxx = xn*Lxx
	if n>0 : Hxx += 2.0*float(n)*x0**(n-1)*Lx
	if n>1 : Hxx += L*float(n)*float(n-1)*x0**(n-2)
	Hyy           = xn*Lyy
	Hxy           = xn*Lxy
	if n>0 : Hxy += Ly*float(n)*x0**(n-1)
	
	return Hxx, Hyy, Hxy

def dfuncLimb(x0, y0, dic):
	
	R             = x0**2+y0**2
	mu2           = 1.-R
	mus           = mu2**(1/4)
	dIdR          = 0.

	for i in range(2):
		dIdR -= 0.25*dic['powers'][i]*dic['coefficients'][i]*mus**dic['powers'][i]/mu2
	Jx            = 2.*x0*dIdR
	Jy            = 2.*y0*dIdR
	
	return Jx, Jy


def ddfuncLimb(x0, y0, dic):
	
	R            = x0**2+y0**2
	mu2          = 1.-R
	mus          = mu2**(1/4)
	IR           = 0.
	IRR          = 0.
		
	for i in range(2):
		var  = 0.25*dic['powers'][i]*dic['coefficients'][i]*mus**dic['powers'][i]/mu2
		IR  -= var
		IRR += var*(0.25*dic['powers'][i]-1.)/mu2
	
	
	Hxx          = 2.*IR+4.*x0**2*IRR
	Hyy          = 2.*IR+4.*y0**2*IRR
	Hxy          = 4.*x0*y0*IRR
	
	return Hxx, Hyy, Hxy
