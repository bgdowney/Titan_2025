'''
NOTES
This file (python_code_DowneyNimmo2024.py) is intended to reproduce the graphs and numbers of Downey & Nimmo (2024)
Uses Python
Last modified March 25, 2024
Email bgdowney@ucsc.edu

TABLE OF CONTENTS

* General parameters*
- conversions, G, etc.

* Satellite parameters *
- returns list of satellite parameters used
Moon() 
Titan()
Io()
Europa()
Ganymede()
Callisto()

* Orbital and tidal equations *
Edote_solid() - dissipation from solid-body eccentricity tides
Edoto_solid() - dissipation from solid-body obliquity tides
deriv_a() - da/dt due to eccentricity and obliquity tides
deriv_ecc() - de/dt due to eccentricity tides
deriv_inc() - di/dt due to obliquity tides	
heat_flux() - surface heat flux due to satellite tidal heating
mean_motion() - satellite's orbital mean motion
timescale_ecc() - eccentricity damping timescale due to satellite tidal heating
timescale_inc() - inclination damping timescale due to satellite tidal heating

* Ocean tidal heating *
Edot_obl_ocean_HM19() - ocean obliquity tidal heating
nu_obl_HM19() - turbulent viscous diffusivity from obliquity tides
vel_obl_Chen14() - average velocity of flow from obliquity tides
cd_Fan19() - scaling law to relate bottom drag coefficient to Reynolds number 

* Moment of inertia *
moi_core() - moment of inertia of a core
moi_mantle() - moment of inertia of the mantle/shell

* Spin angle equations *
prec_fig() - term to help calculate precessional torques on the satellite's permanent figure
tid() - term to help calculate tidal torques on satellite
cmb() - term to help calculate CMB torques on satellite    
tan_phi() - equilibrium spin axis: tan(phi), phi is azimuth
sin_phi() - equilibrium spin axis: sin(phi), phi is azimuth
cos_phi() - equilibrium spin axis: cos(phi), phi is azimuth
CS_spin_axis() - equilibrium spin axis: sx, sy, sz

* Dissipation parameters *
k2oQ_from_offset() - solve for k2/Q from Cassini plane offset
KoC_from_offset() - solve for K/C from Cassini plane offset

* Uncertainty *
dk2oQ() - uncertainty in Titan's k2/Q

* Obliquity solver *
CSR_T_KoC() - Cassini state relation (nonlinear equation to numerically solve for obliquity)
obliquity() - Solves for the obliquity

* Functions to print information/generate graphs *
spin_dissipation_calculations() - print predicted angles/end-member dissipation/heating rates
satellite_offset() - graph predicted spin angles vs k2/Q for Jovian satellites
satellite_subplots() - used satellite_offset() to make 4 subplots (Io, Europa, Ganymede, and Callisto)
k2oQ_remainder() - helper function for graph_dissipation_partition()
graph_dissipation_partition() - graph curves of dissipation solutions for Moon's and Titan's Cassini plane offsets
calculate_Titan_ocean_tides() - calculate Titan's bottom drag coefficient and ocean obliquity tidal heating
graph_heat_damping() - Titan's tidal heating rate and eccentricity/inclination damping timescales
graph_Titan_free_ocean_nutation() - Titan's free ocean nutation period vs cmb flattening

* Replicate numbers and graphs in Downey & Nimmo (2024) paper *
[List of functions to call]
'''

#################
#               #
# START OF CODE #
#               #
#################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

''' General parameters '''
# JPL SSD: J2000 = (12h on January 1, 2000) = JD 2451545.0
# All JPL SSD Horizons ephemeris numbers taken from JD 2451545.0
G = 6.674e-11 # [N m^2 / kg^2] gravitational constant, ref: Wikipedia
Msun = 1.9885e30 # [kg] mass of the Sun, ref: JPL SSD, Horizons physical properties
s_in_yr = 365*24*3600 # [s] number of seconds in a year
rd = 180/np.pi # [deg] conversion rad -> deg

def Moon():
	''' Return's a list of the Moon's parameters '''
	
	''' Earth parameters, p is for planet '''
	Mp = 5.972e24 # [kg] Mass of Earth, ref: JPL SSD, Planetary Physical Parameters/Horizons ephemeris
	Rp = 6.378e6 # [m] Radius (equatorial), ref: JPL SSD, Planetary Physical Parameters/Horizons ephemeris
	ap = 1.489e11 # [m] Semimajor axis, ref: JPL SSD, Horizons ephemeris
	k2p = 0.97 # [ ] Fluid k2 Love number, ref: ???
	Omegap = 7.292e-5 # [rad/s] Rotation frequency, ref: JPL SSD, Horizons geophysical properties

	''' Moon physical parameters '''
	M = 7.349e22 # [kg] Mass of Moon, ref: JPL SSD, Horizons geophysical data
	R = 1738.0e3 # [m] Radius of Moon (from gravity), ref: JPL SSD, Horizons geophysical data

	''' Konopliv et al. 1998 - Lunar Prospector '''
	#c = 0.393 # [ ] Normalized polar moment of interia, ref: Konopliv+1998

	''' Konopliv et al. 2013 - GRAIL JPL (vs GSFC analysis Lemoine et al. 2013) '''
	J2 = 203.305e-6 # [ ] Degree-2 gravity, ref: GRAIL - Konopliv+2013 Table 4
	C22 = 22.4261e-6 # [ ] Degree-2 gravity, ref: GRAIL - Konoplic+2013 Table 4

	''' Williams et al. 2014 - LLR + GRAIL DE430'''
	# Gravity from GRAIL, physical librations from LLR
	#J2 = 203.21568e-6 # [ ] degree-2 gravity, ref: Williams+2014, Table 3, fixed from GRAIL
	#C22 = 1/4 * 0.8971e-4 # [ ] degree-2 gravity from (B-A)/MR^2, ref: Williams+2014, page 1556, (B-A)/MR^2 = 0.8971e-4
	c = 0.392863 # [ ] normalized polar MOI, ref: Williams+2014, GRAIL estimate, page 1556, C/MR^2 = 0.392863 +/- 0.000012
	#fc = 2.46e-4 # [ ] core oblateness, ref: GRAIL + LLR - Williams+2014, Table 3, (2.46+/-1.4)e-4
	#k2oQ = 6.4e-4 # [ ] tidal dissipation, ref: Williams+2014, page 1558, (6.4+/-0.6)e-4 at monthly tidal frequency
	#KoC = 1.64e-8 / (3600*24) # [1/day -> 1/s] CMB friction, ref: GRAIL + LLR - Williams+2014, Table 3 & 5, (1.64+/-0.17)e-8 days
	#k2 = 0.024059 # [ ] tidal k2 Love number, ref: GRAIL + LLR - Williams+2014, Table 3 & 5, fixed from GRAIL
	#k2 = 0.02416 # [ ] tidal k2 Love number, ref: GRAIL + LLR - Williams+2014, summary averaged GRAIL-determined value

	''' Williams & Boggs 2015 - LLR '''
	k2oQ = 6.4e-4 # [ ] Tidal dissipation factor, ref: LLR - Williams & Boggs 2015, Table 7 (6.4 +/- 0.6 e-4)
	KoC = 1.41e-8 / (3600*24) # [1/day -> 1/s] Core-mantle boundary friction, ref: LLR - Williams & Boggs 2015, Table 4 (1.41 +/- 0.34 e-8/day) 
	k2 = 0.02416 # [ ] tidal k2 Love number, ref: LLR - Williams & Boggs 2015, Table 7

	''' Viswanathan et al. 2019 - LLR INPOP17a '''
	fc = 2.17e-4 # [ ] core oblateness, ref: Viswanathan+2019, Fig. 1 (2.2+/-0.6)e-4, Table B3 Rcmb=380 so fc = 2.17e-4
	Rc = 380e3 # [m] core radius, ref: Viswanathan+2019, 381 +/- 12 km
	rhoc = 5811.6 # [kg/m^3] core density, ref: Viswanathan+2019, Table B3 pick R_cmb = 380 km
	#KoC = 7.72e-9 / (3600*24) # [1/day->1/s] visous cmb friction ref: Viswanathan+2019, Table B3, pick R_cmb = 380 km
	#c = 0.39322 # [ ] normalized polar MOI, ref: Viswanathan+2019, Table B3, pick Rcmb = 380 km so c=.39322
	
	''' Moon orbital/rotational parameters '''
	a = 3.844e8 # [m] Semimajor axis, ref: JPL SSD, Planetary satellite mean elements
	e = 0.0554 # [ ] Eccentricity, ref: JPL SSD, Planetary satellite mean elements
	inc = 5.16/rd # [deg -> rad] Inclination, ref: JPL SSD, Planetary satellite mean elements
	theta = 6.67/rd # [deg -> rad] Obliquity, ref: JPL SSD, Horizons geophysical data
	n = 2.662e-6 # [rad/s] Orbital frequency/mean motion, ref: JPL SSD, Horizons geophysical data
	T = 18.6*s_in_yr # [yr -> s] Nodal precession period, ref: JPL SSD, Horizons Planetary satellite mean elements
	eta = -2*np.pi/T # [rad/s] Precession frequency of the longitude of the ascending node, neg b/c clockwise
	gamma = -0.27/3600/rd # [arcseconds -> deg -> rad] Cassini plane offset, ref: LLR - Williams+2014, page 1557, 0.27" or 7.5e-5 deg
	
	name = 'Moon'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name)

def Titan():
	''' Return's a list of Titan's parameters '''

	''' Saturn parameters, p is for planet '''
	Mp = 5.6834e26 # [kg] mass of Saturn, ref: JPL SSD, Horizons physical data
	Rp = 60268e3 # [m] eq. radius at 1 bar, ref: JPL SSD, Horizons physical data (60268 +/- 4 km)
	ap = 1.4261e12 # [m] semimajor axis, ref: JPL SSD, Horizons ephemeris
	Omegap = 1.63785e-4 # [rad/s] siderial rotation rate, ref: JPL SSD, Horizons physical data

	''' Titan physical parameters '''
	M = 1345.53e20 # [kg] mass of Titan, ref: JPL SSD, Horizons physical properties
	R = 2575.5e3 # [m] mean radius, ref: JPL SSD, Horizons physical properties

	''' Durante et al. 2019 - Cassini gravity data '''
	c = 0.341 # [ ] normalized polar moment of inertia, ref: Durante+2019, Section 4
	J2 = 33.089e-6 # [ ] degree-2 gravity, ref: Durante+2019, Table 2, (error bars)
	C22 = 10.385e-6 # [ ] degree-2 gravity, ref: Durante+2019, Table 2, (error bars)
	k2 = 0.616 # [ ] tidal k2 Love number, ref: Durante+2019, Table 2, 0.616 +/- 0.067

	''' Titan orbital/rotational parameters '''
	a = 1221.87e6 # [m] semimajor axis, ref: JPL SSD, Horizons orbital data
	e = 0.0288 # [ ] eccentricity, ref: JPL SSD, Horizons orbital data
	inc = 0.28/rd # [deg->rad] inclination, ref: JPL SSD, Horizons orbital data
	n = 2*np.pi/(15.945421*24*3600) # [day->rad/s] orbital frequency from period, ref: JPL SSD, Horizons orbital data
	#n = mean_motion(Mp, a) # [rad/s] orbital frequency
	T = 687.37*s_in_yr # [yr->s] nodal precession period, ref: JPL SSD, Planetary satellite mean elements
	eta = -2*np.pi/T # [1/s] nodal precession frequency
	
	''' Different published rotation states (some explicitly state negative) '''
	#theta = -0.323/rd # [deg->rad] obliquity, ref: Stiles et al. 2008, Section 5
	#gamma = 0.091/rd # [deg->rad] Cassini plane offset, ref: Stiles et al. 2008 
	#theta = 0.31/rd # [deg->rad] obliquity, ref: Meriggiola et al 2016
	theta = -0.32/rd # [deg->rad] obliquity, ref: Baland+2011, 0.32 +/- 0.02 deg
	gamma = -0.12/rd # [deg->rad] Cassini plane offset, ref: Baland+2011, 0.12 +/- 0.02 deg

	''' Not measured, c is for core '''
	k2oQ = 0 # [ ] tidal dissipation factor
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	fc = 0 # [ ] oblateness of ice shell-ocean interface
	
	name = 'Titan'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name)

def Io():
	''' Return's a list of Io's parameters '''
	
	''' Jupiter parameters, p is for planet '''
	Mp = 1.89819e27 # [kg] Jupiter's mass, ref: JPL SSD, Horizons physical data

	''' Io physical properties '''
	M = 8.9299e22 # [kg] mass, ref: JPL Horizons physical parameters
	R = 1821.5e3 # [m] mean radius, ref: JPL SSD, Horizons physical properties
	k2oQ = 0.015 # [ ] tidal dissipation factor, ref: Lainey et al. 2009, from astrometry, 0.015 +/- 0.003

	''' Anderson et al. 2001 - Galileo gravity data '''
	c = 0.377 # [ ] normalized polar moment of inertia, ref: Anderson et al. 2001
	J2 = 1845.9e-6 # [ ] degree-2 gravity coefficients, ref: Anderson et al. 2001
	C22 = 553.7e-6 # [ ] degree-2 gravity coefficients, ref: Anderson et al. 2001

	''' Io orbital properties '''
	a = 421.8e6 # [m] semi-major axis, ref: JPL SSD, satellite mean elements
	e = 0.00472 # [ ] eccentricity, ref: JPL SSD, Horizons satellite orbit
	inc = 0.0375/rd # [deg->rad] inclination, ref: JPL SSD, Horizons satellite orbit
	#n = 2*np.pi/(1.77*24*3600) # [day->rad/s] orbital frequency from period, ref: JPL SSD, Horizons satellite orbit
	n = mean_motion(Mp, a) # [rad/s] orbital frequency
	T = 7.426*s_in_yr # [yr->s] nodal precession period, ref: Noyelles 2009, Table 5, (JPL SSD has i=0)
	eta = -2*np.pi/T # [rad/s] nodal precession frequency

	''' Not measured '''
	KoC = 0 # [1/s] crust-magma ocean boundary dissiption factor
	theta = 0 # [rad] obliquity
	gamma = 0 # [rad] Cassini plane offset
	fc = 0 # [ ] oblateness of crust-magma ocean boundary

	name = 'Io'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name)

def Europa():
	''' Jupiter parameters, p is for planet '''
	Mp = 1.89819e27 # [kg] Jupiter's mass, ref: JPL SSD, Horizons physical data

	''' Europa physical properties '''
	M = 4.80e22 # [kg] mass, ref: JPL SSD, Horizons physical properties
	R = 1560.8e3 # [m] radius, ref: JPL SSD, Horizons physical properties

	''' Gomez Casajus et al. 2021 - Galileo gravity data (after Juno) '''
	c = 0.3547 # [ ] normalized polar moment of inertia, ref: Gomez Casajus et al. 2021, SOL B Table 2
	J2 = 461.39e-6 # [ ] degree-2 gravity coefficients, ref: Gomez Casajus et al. 2021, SOL B Table 2
	C22 = 138.42e-6 # [ ] degree-2 gravity coefficients, ref: Gomez Casajus et al. 2021, SOL B Table 2
	
	''' Europa orbital properties '''
	a = 671.1e6 # [m] semi-major axis, ref: JPL SSD, satellite mean elements
	e = 0.00981 # [ ] eccentricity, ref: JPL SSD, Horizons satellite orbit
	inc = 0.462/rd # [deg->rad] inclination, ref: JPL SSD, Horizons satellite orbit
	#n = 2*np.pi/(3.55*24*3600) # [day->rad/s] orbital frequency from period, ref: JPL SSD, Horizons satellite orbit
	n = mean_motion(Mp, a) # [rad/s] orbital frequency/mean motion
	T = 29.741*s_in_yr # [yr->s] nodal precession period, ref: Noyelles 2009, sum Table 5
	eta = -2*np.pi/T # [rad/s] nodal precession frequency

	''' Not measured '''
	k2oQ = 0 # [ ] tidal dissipation factor
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	fc = 0 # [ ] oblateness of ice shell-ocean interface
	theta = 0 # [rad] obliquity
	gamma = 0 # [rad] Cassini plane offset
	
	name = 'Europa'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name)

def Ganymede():
	''' Jupiter parameters, p is for planet '''
	Mp = 1.89819e27 # [kg] Jupiter's mass, ref: JPL SSD, Horizons physical data

	''' Ganymede physical properties '''
	M = 1481.5e20 # [kg] mass, ref: JPL SSD, Horizons physical parameters
	R = 2631.2e3 # [m] radius, ref: JPL SSD, Horizons physical parameters

	''' Gomez Casajus et al. 2022 - Galileo and Juno gravity data '''
	c = 0.3159 # [ ] normalized polar moment of inerta, ref: Casajus et al. 2022
	J2 = 133.0e-6 # [ ] degree-2 gravity coefficients, ref: Casajus et al. 2022
	C22 = 39.56e-6 # [ ] degree-2 gravity coefficients, ref: Casajus et al. 2022
	
	''' Alternative: Anderson et al. 1999 - Galileo
	c = 0.311
	J2 = 127.8e-6
	C22 = 38.3e-6 '''
	
	''' Ganymede orbital properties '''
	a = 1070.4e6 # [m] semi-major axis, ref: JPL SSD, satellite mean elements
	e = 0.00146 # [ ] eccentricity, ref: JPL SSD, Horizons satellite orbit
	inc = 0.207/rd # [deg->rad] inclination, ref: JPL SSD, Horizons satellite orbit
	#n = 2*np.pi/(7.155*24*3600) # [day->rad/s] orbital frequency from period, ref: JPL SSD, Horizons satellite orbit
	n = mean_motion(Mp, a) # [rad/s] orbital frequency/mean motion
	T = 136.13*s_in_yr # [yr->s] nodal precession period, ref: Noyelles 2009, sum Table 5
	#T = 137.812*s_in_yr # ref: JPL SSD, Horizons mean elements
	eta = -2*np.pi/T # # [rad/s] nodal precession frequency

	''' Not measured '''
	theta = 0 # obliquity
	gamma = 0 # Cassini plane offset
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	k2oQ = 0 # [ ] tidal dissipation factor
	fc = 0 # [ ] oblateness of ice shell-ocean interface
	
	name = 'Ganymede'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name)

def Callisto():
	''' Jupiter parameters, p is for planet '''
	Mp = 1.89819e27 # [kg] Jupiter's mass, ref: JPL SSD, Horizons physical data

	''' Callisto physical parameters '''
	M = 1.0757e23 # [kg] mass, ref: JPL SSD, Horizons physical properties 
	R = 2410.3e3 # [m] radius, ref: JPL SSD, Horizons physical properties

	''' Anderson et al. 2001 - Galileo '''
	c = 0.3549 # [ ] normalized polar MoI, ref: Anderson et al. 2001
	J2 = 32.7e-6 # [ ] degree-2 gravity coefficients, ref: Anderson et al. 2001
	C22 = 10.2e-6 # [ ] degree-2 gravity coefficients, ref: Anderson et al. 2001
	
	''' Callisto orbital parameters '''
	a =  1882.7e6 # [m] semi-major axis, ref: JPL SSD, satellite mean elements
	e = 0.00744 # [ ] eccentricity, ref: JPL SSD, Horizons ephemeris
	inc = 0.2/rd # [def->rad] inclination, ref: JPL SSD, Horizons ephemeris
	#n = 2*np.pi/(16.691*24*3600) # [day->rad/s] orbital frequency from period, ref: JPL SSD, Horizons satellite orbit
	n = mean_motion(Mp, a) # [rad/s] orbital frequency	
	T = 521.37*s_in_yr # # [yr->s] nodal precession period, ref: Noyelles 2009, sum Table 5
	#T = 577.264*s_in_yr # [yr->s] ref: JPL SSD, Horizons mean elements
	eta = -2*np.pi/T # [rad/s] nodal precession frequency
	
	''' Not measured '''
	theta = 0 # obliquity
	gamma = 0 # Cassini plane offset
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	k2oQ = 0 # [ ] tidal dissipation factor
	fc = 0 # [ ] oblateness of ice shell-ocean interface
	
	name = 'Callisto'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name)


###############################
#                             #
# ORBITAL AND TIDAL EQUATIONS #
#                             #
###############################

def Edote_solid(e, R, n, k2oQ):
	'''
	Tidal dissipation - solid-body eccentricity [W]
	Original reference: Peale & Cassen 1978
	Standard form: e.g., Wisdom 2004
	Manuscript: Eq. 2, used in Fig. 3
	'''
	Edote = 3/2 * k2oQ * (n*R)**5 / G * 7*e**2
	return(Edote)

def Edoto_solid(theta, R, n, k2oQ):
	'''
	Tidal dissipation - solid-body obliquity [W]
	Original reference: Peale & Cassen 1978
	Standard form: e.g., Wisdom 2004
	Manuscript: Eq. 2, used in Fig. 3
	'''
	Edoto = 3/2 * k2oQ * (n*R)**5 / G * np.sin(theta)**2
	return(Edoto)

def deriv_a(Mp, M, a, Edot_tid):
	'''
	da/dt due to tidal dissipation in the satellite [m/s]
	Ex reference: Chyba et al. 1989
	Notes: Edot (dissipation) is positive number, so negative sign to decrease a
	Manuscript: used to find Titan's semi-major axis shrinking rate (Dynamical Implications for Titan)
	'''
	dadt = - 2 * a**2 * Edot_tid / (G*Mp*M)
	return(dadt)

def deriv_ecc(Mp, M, a, e, Edote):
	'''
	de/dt due to eccentricity tidal heating in the satellite [1/s]
	Ex reference: Chyba et al. 1989
	Notes: Edot (dissipation) is positive number, so negative sign to decrease e
	Manuscript: Eq. 4, used in Fig 3b
	'''
	dedt = - a/(G*Mp*M) * (1-e**2)/e * Edote
	return( dedt )

def deriv_inc(Mp, M, a, inc, Edoto):
	'''
	di/dt due to obliquity tidal heating in the satellite
	Ex reference: Chyba et al. 1989
	Notes: Edot (dissipation) is positive number, so negative sign to decrease i
	Manuscript: Eq. 3, used in Fig 3b
	'''
	didt = - a/(G*Mp*M) / np.tan(inc) * Edoto
	return(didt)
	
def heat_flux(Mp, R, n, a, e, theta, k2oQ):
	'''
	Surface heat flux due to tidal heating in the satellite [W/m^2]
	Notes: solid-body eccentricity & obliquity tides only
	Manuscript: Fig 3a right y-axis
	'''
	Edote = Edote_solid(e, R, n, k2oQ)
	Edoto = Edoto_solid(theta, R, n, k2oQ)
	Edot = Edote + Edoto
	flux = Edot / (4*np.pi*R**2)
	return(Edote, Edoto, flux)
    
def mean_motion(Mp, a):
	''' Orbital mean motion '''
	n2 = G*Mp/a**3
	return( np.sqrt(n2) )

def timescale_ecc(Mp, M, R, a, n, e, k2oQ):
	'''
	Eccentricity damping timescale due to tidal heating in the satellite [s]
	e / (de/dt)
	Manuscript: Fig 3b 
	'''
	Edote = Edote_solid(e, R, n, k2oQ)
	dedt = deriv_ecc(Mp, M, a, e, Edote)
	taue = np.abs( e / dedt ) # because de/dt is negative
	return(Edote, taue)

def timescale_inc(Mp, M, R, a, n, inc, theta, k2oQ):
	'''
	Inclination damping timescale due to tidal heating in the satellite [s]
	tan(inc) / (di/dt)
	Manuscript: Fig 3b
	'''
	Edoto = Edoto_solid(theta, R, n, k2oQ)
	didt = deriv_inc(Mp, M, a, inc, Edoto)
	taui = np.abs( np.tan(inc) / didt ) # because di/dt is negative
	return(Edoto, taui)
	
#######################
#                     #
# OCEAN TIDAL HEATING #
#                     #
#######################

def Edot_obl_ocean_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2, rho):
	'''
	[W] Energy dissipation due to obliquity tides in a subsurface ocean on a synchronous satellite
	Ref: Hay & Matsuyama 2019 - scaling laws
	Notes: nu is the turbulent viscous diffusivity, other parameters described in Ref
	Manuscript: plotted in Fig 3a, used in calculate_Titan_ocean_tides()
	'''
	nu = nu_obl_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2)
	A = 12*np.pi * rho * h * nu * (Omega*R*theta*u2*(R/rt))**2
	B = 1 + (20*u2*beta2*nu*g*h/(Omega**3*rt**4))**2
	return(A/B)

def nu_obl_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2):
	'''
	Turbulent viscous diffusivity from obliquity tide flow
	Ref: Hay & Matsuyama 2019 - scaling laws
	Notes: to be used in Edot_obl_ocean_HM19 and vel_obl_Chen14
	Manuscript: None, helper function for Edot_obl_ocean_HM19
	'''
	C = Omega**3 * rt**4 / (20 * np.sqrt(2) * beta2 * g * h)
	D = 200 / 3 * 0.4 * cd * beta2 * u2 * g * R**2 * theta / (Omega**2 * rt**3)
	return(C * np.sqrt(-1+np.sqrt(1+D**2)))

def vel_obl_Chen14(R, Omega, theta, g, h, cd, beta2, u2, rt):
	'''
	Average obliquity flow velocity
	Ref: Chen et al. 2014, Table 4 - scaling laws
	Notes: uses turbulent viscous diffusivity from HM19
	Manuscript: None, helper function for Edot_obl_ocean_HM19   
	'''
	u0 = (Omega*R)**3 / (g*h)
	num = np.sqrt(3/2) * Omega * R * theta
	denom = np.sqrt( 1 + 400*nu_obl_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2)**2 / (u0**2 * R**2) )
	return(num/denom)

def cd_Fan19(cd, R, n, theta, g, h, eta, b2, u2, d, rho):
	'''
	Ocean bottom drag coefficient cD using empirical scaling laws from Earth oceans
	Ref: Fan et al. 2019
	Notes:
	- cD is a fcn of Reynolds number which is a fcn of flow velocity, which is a fcn of cD (circular dependency)
	- Circular dependency means need to solve numerically
	- Uses Chen+2014 velocity flow equation and Hay & Matsuyama 2019 viscous diffusivity equation
	Manuscript: used in calculate_Titan_ocean_tides(), helper function for Edot_obl_ocean_HM19
	'''
	rhs = cd
	vel = np.abs(vel_obl_Chen14(R, n, theta, g, h, cd, b2, u2, R-d))
	lhs = 0.395*(vel*h*rho/eta)**(-0.518) # scaling law for cD: cD as a function of the Reynolds number
	return(rhs - lhs)

#####################
#                   #
# MOMENT OF INERTIA #
#                   #
#####################

def moi_core(rho_c, R_c, f_c):
	''' Moment of inertia of the core [kg m^2], one-layer
	Notes: f_c is the oblateness of the core
	Manuscript: none '''
	return( 8*np.pi/15 * rho_c * R_c**5 * (1+2/3*f_c) )

def moi_mantle(rho_m, R, R_cmb, f_s, f_cmb):
	''' Moment of inertia of the mantle (Moon)/ice shell (Titan) [kg m^2], one-layer
	Notes: f_s is the oblateness of the surface, f_cmb is the oblateness of the core-mantle/ocean-shell boundary
	Manuscript: Eq. 6 of Supplemental Materials, used in Titan's Free Ocean Nutation and Fig. S1 '''
	return( 8*np.pi/15 * rho_m * (R**5 * (1+2/3*f_s) - R_cmb**5 * (1+2/3*f_cmb)) )
	
########################
#                      #
# SPIN ANGLE EQUATIONS #
#                      #
########################

def prec_fig(n, inc, theta, eta, c, J2, C22):
	''' (Helper function)
	- Component of precession of spin axis due to torques on permanent figure
	Manuscript: E.g., in Eq. 17 and is the denominator of Eq. 18, 19 in '''
	fig = 3/2 * n/c * ((J2+C22)*np.cos(theta)+C22)
	orb = eta*np.cos(inc)
	return( fig + orb )
	
def tid(Mp, M, R, a, n, theta, c, k2oQ):
	''' (Helper function)
	- Component of tidal torque exerted on satellite
	Ref: Gladman et al. 1996, Appendix A, Eq. 49
	Manuscript: Eqs. 9 and 10 '''
	T = 3 * k2oQ * n/c * Mp/M * (R/a)**3
	return( T * (1-1/2*np.cos(theta) ) )

def cmb(KoC, de):
	''' (Helper function) 
	- Component of torque due to differential rotation at core-mantle boundary
	Ref: Williams et al. 2001
	Manuscript: Eqs. 4 of Supplemental Materials '''
	return( KoC * np.sin(de) )
    
def tan_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Steady-state spin axis: dsx/dt = 0 -> tan(phi)
	Manuscript: Eq. 18 '''
	tid_term = tid(Mp, M, R, a, n, theta, c, k2oQ)
	cmb_term = cmb(KoC, de) / np.sin(theta)
	prec = prec_fig(n, inc, theta, eta, c, J2, C22)
	return( (tid_term+cmb_term)*np.cos(theta)/prec )

def sin_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Steady-state spin axis: dsz/dt = 0 -> sin(phi)
	Manuscript: sy in Eqns. 16 '''
	tid_term = tid(Mp, M, R, a, n, theta, c, k2oQ) * np.sin(theta)
	cmb_term = cmb(KoC, de)
	prec = eta * np.sin(inc)
	return( (tid_term+cmb_term)/prec )

def cos_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Steady-state spin axis: cos(phi) = sin(phi)/tan(phi)
	Manuscript: sx in Eqns. 16 '''
	prec_figure = prec_fig(n, inc, theta, eta, c, J2, C22)
	prec_orbit = eta * np.sin(inc)
	return( np.tan(theta)*prec_figure/prec_orbit )

def CS_spin_axis(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Components of the equilibrium spin axis: sx, sy, sz
	Manuscript: Eqns. 16, sy = sin(gamma) is Eq. 1'''
	sx = np.sin(theta) * cos_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de)
	sy = np.sin(theta) * sin_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de)
	sz = np.cos(theta)
	return(sx, sy, sz)

##########################
#                        #
# DISSIPATION PARAMETERS #
#                        #
##########################

def k2oQ_from_offset(Mp, M, R, a, n, inc, theta, eta, c, gamma):
	''' Solve for k2/Q from sin(gamma) (Eq. 1)
	Manuscript: used to find endmember k2/Q
	- Figure 2
	- Eq. 8 in SM + uncertainty analysis '''
	tid_mod = 3 * n/c * Mp/M * (R/a)**3 * (1-1/2*np.cos(theta)) # NO k2/Q
	prec_orb = eta * np.sin(inc)
	return( np.sin(gamma) * prec_orb / np.sin(theta)**2 / tid_mod )

def KoC_from_offset(theta, inc, eta, gamma, de):
	''' Solve for K/C from sin(gamma) (Eq. 1)
	Manuscript: used to find endmember K/C, Figure 2 '''
	cmb_mod = np.sin(de) # NO K/C
	prec_orb = eta * np.sin(inc)
	return( np.sin(gamma) * prec_orb / np.sin(theta) / cmb_mod )

###############
#             #
# UNCERTAINTY #
#             #
###############

def dk2oQ(Mp, M, R, a, n, inc, theta, eta, c, gamma, k2oQ, dgamma, dtheta, dc):
	'''
	Finds the uncertainty in Titan's k2/Q endmember from uncertainty in gamma/theta/c
	Manuscript:
	- Titan's k2/Q in the form (nominal) +/- (uncertainty) appears through Manuscript
	- Algebra in "Uncertainty analysis for k2/Q" section in Supplemental Materials
	- Called in spin_dissipation_calculations(moon)
	'''
	# Nominal value, Eq 8 in SM
	k2oQ = k2oQ_from_offset(Mp, M, R, a, n, inc, theta, eta, c, gamma)

	# Partial derivatives: d(k2/Q)/d(variable), Eq 10 in SM
	dk_dg = k2oQ * np.cos(gamma)/np.sin(gamma)
	dk_dt = -k2oQ * (np.sin(theta)/(2-np.cos(theta)) + 2*np.cos(theta)/np.sin(theta))
	dk_dc = k2oQ / c

	# Partial derivs * uncertainties add in quadrature, Eq 9 in SM
	dk2oQ_2 = (dk_dg * dgamma)**2 + (dk_dt * dtheta)**2 + (dk_dc * dc)**2
	
	return( np.sqrt(dk2oQ_2) )

####################
#                  #
# OBLIQUITY SOLVER #
#                  #
####################

def CSR_T_KoC(o, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ, KoC, de):
	'''
	Cassini state relation with tidal and CMB dissipation
	Found by:
	- dsy/dt = 0
	- sx^2 - sy^2 - sz^2 = 0
	Manuscript: Eq. 17 '''
	orb = eta * np.sin(inc)
	fig = prec_fig(n, inc, o, eta, c, J2, C22)
	tid_term = tid(Mp, M, R, a, n, o, c, k2oQ) * np.sin(o)
	cmb_term = cmb(KoC, de)
	return( orb**2 - fig**2*np.tan(o)**2 - (tid_term+cmb_term)**2 )

def obliquity(guess, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ, KoC, de):
	''' Numerically solve the Cassini state relation to get obliquity b/c nonlinear '''
	theta = root_scalar(CSR_T_KoC,
			    args=(Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ, KoC, de),
			    x0=guess, x1=guess/2).root
	return(theta)

###########################################
#                                         #
# FUNCTIONS TO PRINT INFO/GENERATE GRAPHS #
#                                         #
###########################################

def spin_dissipation_calculations(moon):
	'''
	PURPOSE: Print out predicted angles/end-member dissipation/heating rates
	CHOOSE: moon = Moon or Titan
	MANUSCRIPT: Prints endmember dissipative parameters & Titan's predicted Cassini state obliquity
	'''

	''' Set up parameters '''
	# Save parameters specific to Moon or Titan
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, fc0, gamma, name = moon()

	# Define azimuth phi from offset gamma and obliquity theta
	sinphi = np.sin(gamma)/np.sin(theta0)
	phi = np.arcsin(sinphi)

	# Trying to get phi to be > 180 deg if in quadrant 3
	if gamma*theta0 > 0 : phi = np.abs(phi) + np.pi

	# Definitions commented for Moon
	if moon == Moon:
		guess = inc # seed value for obliquity solver, theta = OOM(inc)
		de = theta0-inc # \Delta\epsilon = angle b/w core spin axis (ecliptic normal) and mantle spin axis
		dgamma = 0 # uncertainty in offset (might exist, but I didn't look for)
		dtheta = 0 # uncertainty in obliquity (might exist, but I didn't look for)
		dc = 0 #uncertainty in polar moment of inertia (might exist, but I didn't look for)
		
	elif moon == Titan:
		guess = -inc # Titan is in CS1 so obliquity is negative
		de = theta0-inc # angle between core and mantle not observed for Titan, but maximum is this
		dgamma = 0.02/rd # [rad] Baland+2011 Sec 3.2 derive own uncertainties for Cassini plane offset, 0.02 deg
		dtheta = 0.02/rd # [rad] Baland+2011 Sec 3.2 derive own uncertainties for obliquity, 0.02 deg
		dc = 0.03 # this is an estimate: nominal = 0.341, highest = 0.36 (Hemingway+2013), lowest = 0.31 (Baland+2014)
		
	''' Predict obliquity '''
	# YES dissipation
	# Use measured/observed satellite values
	# * this will be incorrect for Titan b/c obl not that of a purely rigid body *
	theta_pred = obliquity(guess, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ0, KoC0, de=0)

	''' Predict spin axis '''
	# YES dissipation
	# Use measured/observed satellite values & observed (not predicted) obliquity
	# * primarily used for sy component to get Cassini plane offset *
	sx_p, sy_p, sz_p = CS_spin_axis(Mp, M, R, a, n, inc, theta0, eta, c, J2, C22, k2oQ0, KoC0, de)

	''' Predict azimuth '''
	# YES dissipation
	# Use measured/observed satellite values & observed (not predicted) obliquity
	# * tan(phi)*sin(theta) similar to Organowski & Dumberry (2020)
	# * tan(phi)*sin(theta) ~ sin(gamma) for Moon only b/c 1) small phi 2) purely rigid body
	# * incorrect for Titan b/c 1) large phi 2) obliquity not reproduced from expected figure torques which this eq includes
	tanphi_pred = tan_phi(Mp, M, R, a, n, inc, theta0, eta, c, J2, C22, k2oQ0, KoC0, de) # Fix theta
	
	''' PRINTING CHECKS '''
	print('%s'%name)
	#print('theta-inc %.3g deg'%(de*rd))
	#print('polar MoI %.3g'%c)
	print('Observed angles: theta %.3g deg, phi %.3g deg, gamma %.3g deg'%(theta0*rd, phi*rd, gamma*rd))
	print('Observed diss. factors: k2/Q %.3g, K/C %.3g'%(k2oQ0, KoC0))
	print('Predicted angles: theta %.3g deg, phi from tanphi %.4g deg, gamma from sy %.3g deg'%(theta_pred*rd, np.arctan(tanphi_pred)*rd, np.arcsin(sy_p)*rd))
	print('Predicted spin axis: sx %.3g, sy %.3g, sz %.3g'%(sx_p, sy_p, sz_p))

	''' End member solutions '''
	# Find max k2/Q and max K/C assuming all of offset is due to those sources
	end_k2oQ = k2oQ_from_offset(Mp, M, R, a, n, inc, theta0, eta, c, gamma)
	end_KoC = KoC_from_offset(theta0, inc, eta, gamma, de)

	''' Uncertainty '''
	# Uncertainty in k2/Q due to uncertainty in theta, gamma, and c
	# Used for Titan not the Moon b/c the Moon's k2/Q and K/C are known
	end_k2oQ_unc = dk2oQ(Mp, M, R, a, n, inc, theta0, eta, c, gamma, end_k2oQ, dgamma, dtheta, dc)

	print('\nPredicted diss. factors from Cassini plane offset')
	print('Endmembers: k2/Q %.4g, K/C %.4g s^-1'%(end_k2oQ, end_KoC))
	print('Uncertainty in endmember k2/Q: %.4g'%end_k2oQ_unc)

	''' Heat flux for one k2/Q '''
	# Titan's k2/Q is its endmember solution, the Moon's is its known value
	if moon == Titan : k2oQ_test = end_k2oQ
	if moon == Moon : k2oQ_test = k2oQ0

	# Eccentricity and obliquity tidal heating rate [W] and surface flux [W/m^2]
	Edote, Edoto, flux = heat_flux(Mp, R, n, a, e, theta0, k2oQ_test)

	# Eccentricity and inclination damping timescale due to tidal heating [s]
	Edote2, taue = timescale_ecc(Mp, M, R, a, n, e, k2oQ_test)
	Edoto2, taui = timescale_inc(Mp, M, R, a, n, inc, theta0, k2oQ_test)

	# How semimajor axis shrinks b/c of satellite tides (ignores planet tides)
	dadt_tid = deriv_a(Mp, M, a, Edote+Edoto)
	
	print('Tidal heating: k2oQ %.3g, Edote %.3g W, Edoto %.3g W, total %.3g mW/m^2'%(k2oQ_test, Edote, Edoto, flux*1e3))
	print('Tidal damping: taue %.3g yrs, taui %.3g yrs, dadt %.3g cm/yr'%(taue/s_in_yr, taui/s_in_yr, dadt_tid*100*s_in_yr))

	return()

def satellite_offset(moon, graph=1, k2oQ=0, precision=0, legend=1):
	'''
	PURPOSE: Print out predicted spin angles for Jovian satellites; graph angles vs k2/Q
	CHOOSE: moon = Io/Europa/Ganymede/Callisto
	HOW TO USE:
		a) Make new figure -> graph=1
		b) Subplots -> graph = <subplot axis> (called from satellite_subplots())
		- k2oQ is input variable if you want
		- precision (in deg) is if you want to compare to a space mission accuracy/uncertainty
	MANUSCRIPT: Io/Ganymede angles appear in Table 1, info in "Application to Io, Europa, Ganymede, and Callisto"
	'''	

	# Satellite properties
	Mp, M, R, a, e, inc, theta, n, c, k2oQ0, KoC, J2, C22, eta, fc, gamma, name = moon()
	de = 0 # Assume no differential rotation at core-mantle boundary

	# PHASE 1, print one set of angles from one k2oQ
	theta_pred = obliquity(-inc, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ, KoC, de) # [rad] predicted obliquity, "guess=-inc" b/c Cassini state 1 so negative
	phi_pred = np.arcsin( sin_phi(Mp, M, R, a, n, inc, theta_pred, eta, c, J2, C22, k2oQ, KoC, de) ) # [rad] predicted azimuth
	gamma_pred = np.arcsin( np.sin(theta_pred) * sin_phi(Mp, M, R, a, n, inc, theta_pred, eta, c, J2, C22, k2oQ, KoC, de) ) # [rad] predicted Cassini plane offset
	Edote, Edoto, hf = heat_flux(Mp, R, n, a, e, theta_pred, k2oQ) # [W/m^2] heat flux from chosen k2/Q

	print('%s'%name)
	print('k2/Q %.4g'%k2oQ)
	print('Obliquity %.4g deg'%(theta_pred*rd))
	print('Azimuth %.4g deg'%(phi_pred*rd))
	print('Cassini plane offset %.4g deg'%(gamma_pred*rd))
	print('Heat flux %.4g W/m^2'%hf)

	# PHASE 2, make graph of all spin angles, varying k2oQ, no KoC
	k2oQs = np.logspace(-4, -1, 100)

	thetas = np.array([ obliquity(-inc, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ_iter, KoC, de) for k2oQ_iter in k2oQs ])
	sinphis = np.array([ sin_phi(Mp, M, R, a, n, inc, thetas[j], eta, c, J2, C22, k2oQs[j], KoC, de) for j in range(len(k2oQs)) ])
	singammas = np.sin(thetas) * sinphis

	# GRAPHING
	if graph == 1 :
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.set_ylabel('spin axis angles (deg)')
	else :
		ax = graph

	one = 'paleturquoise'
	two = 'darkturquoise'
	three = 'darkcyan'
	four = 'darkslategrey'

	ax.axhline(y=inc*rd, color=one, linestyle=':', label=r'inclination $i$')
	ax.plot(k2oQs, np.abs(thetas)*rd, c=two, linestyle='--', label=r'obliquity $\theta$')
	ax.plot(k2oQs, np.abs(np.arcsin(sinphis))*rd, c=three, linestyle='-.', label=r'azimuth $\phi$')
	ax.plot(k2oQs, np.abs(np.arcsin(singammas))*rd, c=four, linestyle='-', label=r'offset $\gamma$')

	if precision > 0 :
		ax.axhline(y=precision, color='m', label='accuracy')
		ax.scatter(k2oQ, np.abs(theta_pred)*rd, s=60, c='m', marker='X')
		ax.scatter(k2oQ, np.abs(phi_pred)*rd, s=60, c='m', marker='X')
		ax.scatter(k2oQ, np.abs(gamma_pred)*rd, s=60, c='m', marker='X')
		
	ax.set_xlabel(r'$k_2/Q$'); 
	ax.set_xlim([1e-4, 1e-1])
	ax.set_ylim([3e-6, 1])
	ax.set_xscale('log'); ax.set_yscale('log')
	ax.set_title(name)
	
	if legend == 1: ax.legend()
	if graph == 1: plt.show()

def satellite_subplots():
	''' See satellite_offset() description
	4 subplots
		- Io, Europa, Ganymede, Callisto
		- Inclination, obliquity, azimuth, gamma
	Pick test k2/Q for each body
	Pick test precision/uncertainty for min detectable gamma
	MANUSCRIPT: Table 1, Section on "Application to I, E, G, C"
	'''
	fig, ax = plt.subplots(2,2, sharey=True, figsize=(8,7))

	k2oQ_i = 0.015 # [ ] Io's k2/Q, ref: Lainey et al. 2009, from astrometry
	k2oQ_e = 2e-2 # [ ] Europa's test k2/Q, min detectable by Europa Clipper
	k2oQ_g = 1.6e-3 # [ ] Ganymede's test k2/Q, min detectable by JUICE, 1e-6 rad
	k2oQ_c = 1e-3 # [ ] Callisto's test k2/Q, close to the Moon's value

	# Spacecraft mission spin state precision/uncertainty
	prec_i = 0 # (No planned mission to get Io's spin state)
	prec_e = 0.05/60 # [deg] 0.05 arcmin desired for obl to get 0.004 precision in MOI, Europa Clipper Mazarico et al. 2023
	prec_g = 1e-6*rd # [deg] 1 micro radian, JUICE Cappuccio et al. 2020
	prec_c = 5.5e-3*rd # [deg] 5.5 mrad for obliquity, JUICE Cappuccio et al. 2022

	satellite_offset(Io, ax[0,0], k2oQ_i, prec_i, legend=1)
	satellite_offset(Europa, ax[0,1], k2oQ_e, prec_e, legend=0)
	satellite_offset(Ganymede, ax[1,0], k2oQ_g, prec_g, legend=0)
	satellite_offset(Callisto, ax[1,1], k2oQ_c, prec_c, legend=0)
	ax[0,0].set_ylabel('spin axis angles (deg)')
	ax[1,0].set_ylabel('spin axis angles (deg)')

	plt.tight_layout()
	plt.show()

def k2oQ_remainder(KoC_var, moon):
	'''
	INPUT: A K/C value and choice of moon (Moon/Titan)
	Step 1) What offset does that K/C produce
	Step 2) What offset remains
	Step 3) OUTPUT what k2/Q produces the remainder of the offset
	MANUSCRIPT: helper function for Fig 2 (graph_dissipation_partition())
	'''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, fc0, gamma, name = moon()
	de = theta0 - inc # max angle b/w core spin axis and mantle spin axis

	# Offset from variable K/C, k2/Q = 0
	sin_gamma_cmb = np.sin(theta0) * sin_phi(Mp, M, R, a, n, inc, theta0, eta, c, J2, C22, 0, KoC_var, de)

	# Subtract variable offset from total offset
	sin_gamma_tid = np.sin(gamma) - sin_gamma_cmb

	# k2/Q to produce remainder of offset
	k2oQ_rem = k2oQ_from_offset(Mp, M, R, a, n, inc, theta0, eta, c, sin_gamma_tid)

	return( k2oQ_rem )

def graph_dissipation_partition(if_Titan, if_Moon):
	'''
	PURPOSE: Graph solution curves of constant gamma
	INPUT: if_Titan = 1(0) -> please (don't) graph Titan; if_Moon = 1(0) -> please (don't) graph Moon
	Step 1) Vary K/C
	Step 2) Calculate k2/Q to produce remainder of offset not from variable K/C
	Step 3) Plot
	MANUSCRIPT: Makes Figure 2
	'''

	plt.figure(figsize = 3/5*np.array([6.4, 4.8])) # maintain approximate figure size ratios
	
	if if_Titan == 1 :
		print('Titan')
		KoCs_T = np.logspace(-14, -9.9, 500) # Titan
		k2oQs_rem_T = k2oQ_remainder(KoCs_T, Titan) # Titan
		plt.plot(KoCs_T, k2oQs_rem_T, 'k', linestyle=':', label='Titan')
		txtt = plt.text(*(1.3e-14, .06), r'Titan') #:$\gamma$='+'%.4g'%offset_deg0_t+r'$\degree$', c='w')

	if if_Moon == 1 :
		print('Moon')
		KoCs_M = np.logspace(-14.2, -12.2, 500) # Moon
		k2oQs_rem_M = k2oQ_remainder(KoCs_M, Moon) # Moon
		plt.plot(KoCs_M, k2oQs_rem_M, 'k', linestyle='-', label='Moon')

		# LLR
		parameters = Moon()
		k2oQ0_M, KoC0_M = parameters[9], parameters[10]
		plt.plot(KoC0_M, k2oQ0_M, c='k', marker='o', linestyle='none', label='LLR')
		plt.plot([KoC0_M, KoC0_M], k2oQ0_M+np.array([-1, 1])*.6e-4, c='k', linestyle='-') # LLR k2/Q error bars
		plt.plot((1.41+0.34*np.array([-1, 1]))*1e-8/(3600*24), [k2oQ0_M, k2oQ0_M], c='k', linestyle='-') # LLR K/C error bars

		txtm = plt.text(*(1.3e-14, 1.2e-3*1.2), r'Moon') #:$\gamma$='+'%.4g'%offset_deg0_m+r'$\degree$', c=grey)
		txtm = plt.text(*(KoC0_M*1.5, k2oQ0_M), 'LLR', c='k')

	#plt.legend()
	plt.xlim([1e-14, 1e-10])
	plt.ylim([1e-5, 1.5e-01])
	plt.tick_params(axis='y', which='both',direction='in', right=True)
	plt.tick_params(axis='x', which='both',direction='in', top=True)
	plt.xlabel(r'$K/C$ (s$^{-1}$)')
	plt.ylabel(r'$k_2/Q$')
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

	return()

def calculate_Titan_ocean_tides():
	'''
	PURPOSE: Calculate Titan's ocean obliquity tidal heating
	Step 1) Parameters for Titan's ocean and ice shell
	Step 2) Bottom drag coefficient from Reynolds number scaling law (e.g. Sohl et al. 1995, Fan et al. 2019)
	Step 3) Ocean obliquity tides from scaling law (Chen et al. 2014; Hay & Matsuyama 2019)
	MANUSCRIPT: numbers appear in-text, helper function for Figure 3
	'''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, fc0, gamma, name = Titan()

	rho_f = 1000 # [kg/m^3] density of fluid layer where turbulence is, aka water ocean
	eta_water = 1e-3 # [Pa s] viscosity of water, ref: Sohl et al. 1995
	g = 1.35 # [m/s^2] Titan's surface gravity, ref: Chen et al. 2014 
	h = 150e3 # [m] Titan's approximate ocean thickness

	# Coefficients of order unity described in Hay & Matsuyama 2019
	b2 = 1 # [ ] beta_2 = shell pressure forcing, ref: Hay & Matsuyama 2019
	u2 = 1 # [ ] upsilon_2 = perturbation to the forcing tidal potential due to shell pressure forcing, self-gravity, and deformation of the solid regions, ref: HM2019
	d = 100e3 # [m] ice shell thickness 50-200km, ref: Hemingway et al. 2013

	# Calculate bottom drag from Reynolds number scaling law (Fan et al. 2019 scaling law for Earth used here)
	cd = root_scalar(cd_Fan19, args=(R, n, theta0, g, h, eta_water, b2, u2, d, rho_f), x0=2e-3, x1=1e-4).root

	# Calculate obliquity tidal heating from scaling law (Hay & Matsuyama 2019 include effect of overlying ice shell)
	Edotoo = Edot_obl_ocean_HM19(n, R-d, g, h, R, np.abs(theta0), cd, b2, u2, rho_f)
	return(cd, Edotoo)

	
def graph_heat_damping(moon):
	'''
	LEFT: graph tidal heating rate (left axis) and surface heat flux (right axis) as a fcn of k2/Q
	RIGHT: graph eccentricity/inclination damping timescales as a fcn of k2/Q
	BOTH: mark one k2/Q (Moon is observed k2/Q, Titan is upper bound)
	CHOOSE: moon = Moon or moon = Titan
	MANUSCRIPT: Makes Figure 3
	'''

	''' Satellite parameters '''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, fc0, gamma, name = moon()
	sa = 4 * np.pi * R**2 # [m^2] surface area

	# Pick a k2/Q to display as an "X" on graph
	if moon == Titan :
		k2oQ = k2oQ_from_offset(Mp, M, R, a, n, inc, theta0, eta, c, gamma) # [ ] tides endmember
		KoC = 0 # [ ] assume no differential rotation between ice shell and ocean
		# Comparisons (ocean heating, radiogenic heating)
		cd_T, Edoto_oc_T = calculate_Titan_ocean_tides() # [ ] bottom drag coefficient, [W] ocean obliquity tidal heating
		Edot_rad = 0.003*sa # [W/m^2->W] radiogenic surface flux 3 mW/m^2, ref: Kirk & Stevenson 1987; Mitri & Showman 2008
		
	elif moon == Moon :
		# Moon's k2/Q and K/C are known from Lunar Laser Ranging
		k2oQ = k2oQ0 
		KoC = KoC0


	''' GRAPHS '''
	def heat2flux(heat): return( heat*1e9/sa ) # for left and right y-axes
	def flux2heat(flux): return( flux*sa/1e9 )

	# Vary k2/Q
	k2oQs = np.logspace(-4, -0.5, 100)

	# Eccentricity and inclination damping timescales due to tidal heating only
	Edotes, taues = timescale_ecc(Mp, M, R, a, n, e, k2oQs) # [W] eccentricity tidal heating, [s] eccentricity damping timescale
	Edotos, tauis = timescale_inc(Mp, M, R, a, n, inc, theta0, k2oQs) # [W] obliquity tidal heating, [s] inclination damping timescale

	# For specific k2/Q defined above
	Edote, taue = timescale_ecc(Mp, M, R, a, n, e, k2oQ) # [W] eccentricity tidal heating, [s] eccentricity damping timescale
	Edoto, taui = timescale_inc(Mp, M, R, a, n, inc, theta0, k2oQ) # [W] obliquity tidal heating, [s] inclination damping timescale
	
	''' Left a) Graph tidal heating '''
	fig, axes = plt.subplots(1, 2)
	ax = axes[0]
	ax.plot(k2oQs, Edotes/1e9, 'k', label='solid-body eccentricity')
	ax.plot(k2oQs, Edotos/1e9, 'k--', label='solid-body obliquity')
	ax.scatter(k2oQ, Edote/1e9, s=60, c='k', marker='X')
	ax.scatter(k2oQ, Edoto/1e9, s=60, c='k', marker='X')

	secax = ax.secondary_yaxis('right', functions=(heat2flux, flux2heat))
	secax.set_ylabel(r'heat flux (W/$m^{2}$)')

	if moon == Titan :
		ax.axhline(y=Edoto_oc_T/1e9, c='k', linestyle='-.', label='ocean obliquity')
		ax.axhline(y=Edot_rad/1e9, c='k', linestyle=':', label='radiogenic') 

	ax.set_xlabel(r'$k_2/Q$')
	ax.set_ylabel('tidal heating (GW)')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.legend()

	''' Right b) Damping timescale '''
	ax = axes[1]
	ax.plot(k2oQs, tauis/(1e6*s_in_yr), 'k--', label='inclination')
	ax.plot(k2oQs, taues/(1e6*s_in_yr), 'k', label='eccentricity')
	ax.scatter(k2oQ, taue/(1e6*s_in_yr), s=60, c='k', marker='X')
	ax.scatter(k2oQ, taui/(1e6*s_in_yr), s=60, c='k', marker='X')
	ax.tick_params(axis='y', which='both', direction='in', right=True)
	ax.tick_params(axis='x', which='both', direction='in', top=True)
	ax.set_xlabel(r'$k_2/Q$')
	ax.set_ylabel('damping timescale (Myr)')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.legend()
	fig.tight_layout()
	plt.show()

	return()

def graph_Titan_free_ocean_nutation(graph):
	'''
	PURPOSE: Titan's free ocean nutation period vs cmb flattening
	Step 1) Surface flattening from data and ice shell-ocean boundary flattening from models
	Step 2) Print one example of ice shell thickness, ocean flattening, ice shell MoI ratio
	Step 3) Do ocean flattenings from models produce Free Ocean Nutation periods faster or slower than ice shell?
	CHOOSE: graph = 1 (0), please (don't) graph Titan's Free Ocean Nutation period as a function of ocean flattening
	MANUSCRIPT: Figure S1 in the Supplementary Materials, used to conclude that ocean is aligned with ice shell
	'''

	# Titan parameters
	Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name = Titan()
	C = c * M * R**2 # [kg m^2] total body polar MoI 
	rho_m = 1000 # [kg/m^3] density of ice shell, subscript m for mantle        

	# Polar flattenings/oblateness, defined f(or alpha)~J2/c, beta~4C22/c '''
	f_s = 19.2236e-5 # [ ] Surface polar flattening, ref: Baland et al 2014, Eq. 5, from Zebker et al 2009
	f_cmb_low = -2e-4 # [ ] Lower bound on ocean polar flattening, ref: alpha_o Baland+14 Fig. 1 & Coyette+18 Fig. 1
	f_cmb_hig = -4.4e-3 # [ ] Upper bound on ocean polar flattening, ref: alpha_o Baland+14 Fig. 1 & Coyette+18 Fig. 1

	''' Helper Functions '''
	def fcn(f_cmb, n, C, C_m):
		''' Free core nutation
		Ref: Meyer Wisdom 2011 Eq. 1
		Manuscript: Eq. 5 of the Supplemental Materials '''
		return( np.abs(f_cmb) * n * C / C_m )
	
	def suite(d, f_cmb):
		''' 1) radius of top of ocean 2) MoI of ice shell 3) free ocean nutation '''
		R_cmb = R - d
		C_m = moi_mantle(rho_m, R, R_cmb, f_s, f_cmb)
		fcns = fcn(f_cmb, n, C, C_m)
		return(R_cmb, C_m, fcns)
	
	''' Print one case '''
	# CHOOSE an ice shell thickness d0
	d0 = 200e3 # [m] ice shell thickness, 10km - 200km, Baland 2011 Table 2

	# CHOOSE a flattening of the ice shell-ocean boundary, either lower bound f_cmb_low or upper bound f_cmb_hig
	f_cmb0 = f_cmb_low; bound = 'lower'
	#f_cmb0 = f_cmb_hig; bound = 'upper'

	# Calculate MoI of ice shell and the free ocean nutation
	R_cmb0, C_m0, fcn0 = suite(d0, f_cmb0)

	print(name)
	print('Shell: thickness %.3g km, cmb radius %.3g km, density %.3g kg/m^3'%(d0/1e3, R_cmb0/1e3, rho_m))
	print('Polar flattening: surface %.3g, %s end of ocean %.3g'%(f_s, bound, f_cmb0))
	print('MoI: Cm/C %.3g'%(C_m0/C))

	''' GRAPH '''
	# Baland+19
	f_B19 = 0.8*f_s # [ ] ocean polar shape flattening, ref: Baland+19, Table 5, ~15.38e-5
	T_FON_B19 = 323.87*s_in_yr # [s] free ocean nutation period, ref: Baland+19, Table 6

	# Vary shell thickness
	d_thin = 10e3 # [m] thin ice shell limit, ref: Baland+11, Table 2, 10km - 200km
	d_thic = 200e3 # [m] thick ice shell limit, ref: Baland+11, Table 2, 10km - 200km

	# Vary flattening of cmb (ice shell-ocean boundary)
	f_cmbs = np.linspace(1e-6, 5e-3, 100) # [ ] range for graphing purposes

	# Calculate ice shell MoI and Free Ocean Nutation
	R_cmb_thin, C_ms_thin, fcns_thin = suite(d_thin, f_cmbs)
	R_cmb_thic, C_ms_thic, fcns_thic = suite(d_thic, f_cmbs)

	if graph :
		plt.figure(figsize = 3/5*np.array([6.4, 4.8]))

		# Precession period of mantle/ice shell (core period is compared to mantle period)
		plt.axhline(y=-2*np.pi/eta/s_in_yr, c='k', label='mantle') 

		# Precession period of core/ocean+below for wide range of f_cmb
		plt.plot(f_cmbs, 2*np.pi/fcns_thic/s_in_yr, ':', c='k', label='200 km shell')
		plt.plot(f_cmbs, 2*np.pi/fcns_thin/s_in_yr, '--', c='k', label='10 km shell')
		plt.fill_between(f_cmbs, 2*np.pi/fcns_thic/s_in_yr, 2*np.pi/fcns_thin/s_in_yr, color='0.8', label='core')

		# What f_cmbs are in literature and what are their FCN/FON periods? See if FCN/FON is faster or slower than mantleice shell

		# Baland et al. 2019
		plt.plot(f_B19, T_FON_B19/s_in_yr, linestyle='None', marker='^', c='k', label='B19')

		# Baland et al. 2014 & Coyette et al. 2018
		R_cmb_thin, C_m_thin_low, fcn_thin_low = suite(d_thin, f_cmb_low)
		R_cmb_thin, C_m_thin_hig, fcn_thin_hig = suite(d_thin, f_cmb_hig)
		R_cmb_thic, C_m_thic_low, fcn_thic_low = suite(d_thic, f_cmb_low)
		R_cmb_thic, C_m_thic_hig, fcn_thic_hig = suite(d_thic, f_cmb_hig)

		# Shade the areas consistent with the literature
		plt.fill_between(np.abs([f_cmb_low, f_cmb_hig]),
				 2*np.pi/np.array([fcn_thic_low, fcn_thic_hig])/s_in_yr,
				 2*np.pi/np.array([fcn_thin_low, fcn_thin_hig])/s_in_yr,
				 hatch='xx', color='k', facecolor='purple', alpha=0.75, label='B14 & C18')
		
		plt.xlabel(r'$f_{cmb}$')
		plt.ylabel(r'free ocean nutation period (yrs)')
		plt.xscale('log')
		plt.yscale('log')
		plt.legend(loc='lower left')
		plt.title(name)
		plt.show()

	return()



###############################################################
###############################################################
###############################################################
#                                                             #
# Replicate numbers and graphs in Downey & Nimmo (2024) paper #
#                                                             #
###############################################################
###############################################################
###############################################################

''' Figure 1: diagram of the spin geometry (made in ppt not with python code) '''

''' Figure 2: the set of k2/Q and K/C solutions to reproduce the Moon's and Titan's Cassini plane offsets '''
#graph_dissipation_partition(if_Titan = 1, if_Moon = 1)


''' Figure 3: a) tidal heating & heat flux b) eccentricity & inclination damping timescales as fcn of k2/Q '''
#graph_heat_damping(moon=Titan)


''' Figure S1: Titan's free ocean nutation period vs flattening at ocean-ice shell boundary '''
#graph_Titan_free_ocean_nutation(graph = 1)


''' Table 1: rotation states of Moon/Titan/Io/Ganymede '''
#spin_dissipation_calculations(Moon)
#satellite_offset(moon=Io, graph=1, k2oQ=0.015, precision=0, legend=1)
#satellite_offset(moon=Ganymede, graph=1, k2oQ=0.0016, precision=1e-6*rd, legend=1)


''' Table 2: physical and orbital parameters for Moon/Titan/Io/Ganymede '''
#Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, name = Moon()
#print('Indices: 0Mp, 1M, 2R, 3a, 4e, 5inc, 6theta, 7n, 8c, 9k2oQ, 10KoC, 11J2, 12C22, 13eta, 14fc, 15gamma, 16name')
#Titan()
#Io()
#Ganymede()
## Example
#eccentricity_io = Io()[4]


''' Other: print observed angles and predicted angles, dissipative parameters, heating rates, and damping timescales '''
#spin_dissipation_calculations(Moon)
#spin_dissipation_calculations(Titan)


''' Other: print Titan's ocean bottom drag coefficient and obliquity tidal heating  '''
#cD, Edoto = calculate_Titan_ocean_tides(); print('Bottom drag coefficient %.4g, ocean obliquity tidal heating %.4g W'%(cD, Edoto))


''' Other: predict spin states of Jovian satellites '''
#satellite_subplots()

