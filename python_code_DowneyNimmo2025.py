'''
NOTES
This file (python_code_DowneyNimmo.py) is intended to reproduce the graphs and numbers of Downey & Nimmo (202x)
Uses Python
Email bgdowney@ucsc.edu (old) or brynna.downey@swri.org (new)
Last modified 12/23/2024

* Functions to make figures in paper are at the very end of this document *


TABLE OF CONTENTS

* General parameters*
- conversions, G, color blind friendly color scheme, etc.

* Satellite parameters *
- returns list of satellite parameters used
Moon() 
Titan()
Io()
Europa()
Ganymede()
Callisto()

* Spin angle equations *
prec_fig() - term to help calculate precessional torques on the satellite's permanent figure
tid() - term to help calculate tidal torques on satellite
cmb() - term to help calculate CMB torques on satellite    
tan_phi() - equilibrium spin axis: tan(phi), phi is azimuth
sin_phi() - equilibrium spin axis: sin(phi), phi is azimuth
cos_phi() - equilibrium spin axis: cos(phi), phi is azimuth
CS_spin_axis() - equilibrium spin axis: sx, sy, sz

* Obliquity solver *
CSR_T_KoC() - Cassini state relation (nonlinear equation to numerically solve for obliquity)
obliquity() - Solves for the obliquity

* Dissipation parameters from spin state *
k2oQ_from_offset() - solve for k2/Q from Cassini plane offset
k2oQ_from_phi() - solve for k2/Q from azimuth
KoC_from_offset() - solve for K/C from Cassini plane offset
KoC_from_phi() - solve for K/C from azimuth
k2oQ_remainder() - given K/C, what k2/Q produces remainder of azimuth
	
* Uncertainty *
dk2oQ() - uncertainty in Titan's k2/Q

* Moments of inertia and density *
moi_core() - moment of inertia of a core
moi_mantle() - moment of inertia of the mantle/shell

moi_ellipsoid() - MoI of an interior layer
moi_layer() - MoI of one layer of a body
moi_three_layers() - MsoI of each layer of a 3-layer body
density_interior_3layer() - density of interior of a 3-layer body (Titan)
density_surface_2layer() - density of surface layer of a 2-layer body (Moon)
moi_density_FCN_Titan() - 3-layers in Titan: radii, density, MoI + FON

* Free core/ocean nutation *
fcn() - Moon's FCN
fcn_thin_top() - Titan's FON

* Differential rotation friction *
d_omega_viscous() - differential angular velocity, viscous coupling only
d_omega_pressure() - differential angular velocity, pressure coupling only
d_omega_pressure_viscous() - differential angular velocity, pressure and viscous coupling
K_viscous() - Turbulent friction parameter
xi_viscous() - Viscous coupling parameter
K_func() - K equation: (LHS - RHS = 0)
kappa_func() - Skin friction kappa equation: (LHS - RHS = 0)
viscous_funcs() - Solve for K, kappa, xi, and d_omega simultaneously

* Ocean tidal heating *
Edot_obl_ocean_HM19() - ocean obliquity tidal heating
nu_obl_HM19() - turbulent viscous diffusivity from obliquity tides
vel_obl_Chen14() - average velocity of flow from obliquity tides
cd_Fan19() - scaling law to relate bottom drag coefficient to Reynolds number
calculate_Titan_ocean_tides() - obliquity tidal heating in Titan's ocean

* Orbital and tidal equations *
Edot_ecc_solid() - dissipation from solid-body eccentricity tides
Edot_obl_solid() - dissipation from solid-body obliquity tides
Edot_differential_rotation() - dissipation from differential rotation at fluid-solid boundary
deriv_a() - da/dt due to eccentricity and obliquity tides
deriv_ecc() - de/dt due to eccentricity tides
deriv_inc() - di/dt due to obliquity tides	
heat_flux() - surface heat flux due to satellite tidal heating
mean_motion() - satellite's orbital mean motion
timescale_ecc() - eccentricity damping timescale due to satellite tidal heating
timescale_inc() - inclination damping timescale due to satellite tidal heating

* Functions to print information/generate graphs *
1. calculate_dissipation_from_spin()
2. calculate_spin_from_dissipation()
3. solve_KoC() - see notes
4. calculate_heat_damping() - calculate tidal heating
5. graph_dissipation_partition() - Moon and Titan's k2/Q vs K/C_s solutions
6. graph_heat_damping() - Titan's tidal heading and inc/ecc damping timescales as fcn of k2/Q
7. graph_Titan_free_ocean_nutation() - calculate FON as fcn of h and f_o
8. graph_Jovian_spin - inclination, obliquity, azimuth, offset as a fcn of k2/Q
9. subplots_Jovian_spin() - 4 panels for Io, Europa, Ganymede, and Callisto

* Functions to make figures in paper are at the very end of this document *

'''

#################
#               #
# START OF CODE #
#               #
#################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, fsolve
	
''' General parameters '''
# JPL SSD: J2000 = (12h on January 1, 2000) = JD 2451545.0
# All JPL SSD Horizons ephemeris numbers taken from JD 2451545.0
G = 6.674e-11 # [N m^2 / kg^2] Gravitational constant, ref: Wikipedia
Msun = 1.9885e30 # [kg] Mass of the Sun, ref: JPL SSD, Horizons physical properties
s_in_yr = 365*24*3600 # [s] Number of seconds in a year
s_in_Myr = 1e6*s_in_yr # [s] Number of seconds in a million years
rd = 180/np.pi # [deg] Conversion rad -> deg

# IBM Design Library: Color blind friendly colors
# * From davidmathlogic.com/colorblind/
tan = np.array([255, 176, 0])/255
ora = np.array([254, 97, 0])/255
pin = np.array([220, 38, 127])/255
pur = np.array([120, 94, 240])/255
blu = np.array([100, 143, 255])/255

# Bang Wong
# * From davidmathlogic.com/colorblind/
tan = "#E6CB3E"
lbl = "#88CCEE"
tea = "#44AA99"
gre = "#117733"
dbl = "#332288"
cor = "#CC6677"
pin = "#AA4499"
mag = "#882255"

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
	#f_cmb = 2.46e-4 # [ ] oblateness of core-mantle boundary, ref: GRAIL + LLR - Williams+2014, Table 3, (2.46+/-1.4)e-4
	#k2oQ = 6.4e-4 # [ ] tidal dissipation, ref: Williams+2014, page 1558, (6.4+/-0.6)e-4 at monthly tidal frequency
	#KoC = 1.64e-8 / (3600*24) # [1/day -> 1/s] CMB friction, ref: GRAIL + LLR - Williams+2014, Table 3 & 5, (1.64+/-0.17)e-8 days
	#k2 = 0.024059 # [ ] tidal k2 Love number, ref: GRAIL + LLR - Williams+2014, Table 3 & 5, fixed from GRAIL
	#k2 = 0.02416 # [ ] tidal k2 Love number, ref: GRAIL + LLR - Williams+2014, summary averaged GRAIL-determined value

	''' Williams & Boggs 2015 - LLR '''
	k2oQ = 6.4e-4 # [ ] Tidal dissipation factor, ref: LLR - Williams & Boggs 2015, Table 7 (6.4 +/- 0.6 e-4)
	KoC = 1.41e-8 / (3600*24) # [1/day -> 1/s] Core-mantle boundary friction, ref: LLR - Williams & Boggs 2015, Table 4 (1.41 +/- 0.34 e-8/day) 
	k2 = 0.02416 # [ ] tidal k2 Love number, ref: LLR - Williams & Boggs 2015, Table 7

	''' Viswanathan et al. 2019 - LLR INPOP17a '''
	f_cmb = 2.17e-4 # [ ] oblateness of core-mantle boundary, ref: Viswanathan+2019, Fig. 1 (2.2+/-0.6)e-4, Table B3 Rcmb=380 so fc = 2.17e-4
	R_cmb = 380e3 # [m] core radius, ref: Viswanathan+2019, 381 +/- 12 km
	rhoc = 5811.6 # [kg/m^3] core density, ref: Viswanathan+2019, Table B3 pick R_cmb = 380 km
	#KoC = 7.72e-9 / (3600*24) # [1/day->1/s] visous cmb friction ref: Viswanathan+2019, Table B3, pick R_cmb = 380 km
	#c = 0.39322 # [ ] normalized polar MOI, ref: Viswanathan+2019, Table B3, pick Rcmb = 380 km so c=.39322
	
	''' Moon orbital/rotational parameters '''
	a = 3.844e8 # [m] Semimajor axis, ref: JPL SSD, Planetary satellite mean elements
	e = 0.0554 # [ ] Eccentricity, ref: JPL SSD, Planetary satellite mean elements
	#inc = 5.16/rd # [deg -> rad] Inclination, ref: JPL SSD, Planetary satellite mean elements
	inc = 5.145/rd # [deg -> rad] Inclination, ref: Williams+2001, Lunar Laser Ranging
	#theta = 6.67/rd # [deg -> rad] Obliquity, ref: JPL SSD, Horizons geophysical data
	theta = 6.688/rd # [deg -> rad] Obliquity, ref: Williams+2001, Lunar Laser Ranging
	n = 2.662e-6 # [rad/s] Orbital frequency/mean motion, ref: JPL SSD, Horizons geophysical data
	T = 18.6*s_in_yr # [yr -> s] Nodal precession period, ref: JPL SSD, Horizons Planetary satellite mean elements
	eta = -2*np.pi/T # [rad/s] Precession frequency of the longitude of the ascending node, neg b/c clockwise
	gamma = -0.27/3600/rd # [arcseconds -> deg -> rad] Cassini plane offset, ref: LLR - Williams+2014, page 1557, 0.27" or 7.5e-5 deg, neg b/c phase lead
	phi_CS = 0 # [rad] Azimuth of spin axis without dissipation, Moon is in Cassini state 2, so phi_CS = 0
	phi = phi_CS + np.arcsin( np.sin(gamma)/np.sin(theta) )# [rad] Azimuth of spin axis with dissipation, measured from x-axis
	
	name = 'Moon'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_cmb, gamma, phi_CS, phi, name)

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
	#T = 703*s_in_yr # [yr->s] nodal precession period, ref: Vienne & Duriez 1995 via Table 1 of Baland+2019
	eta = -2*np.pi/T # [1/s] nodal precession frequency
	
	''' Different published rotation states (some explicitly state negative) '''
	#theta = -0.323/rd # [deg->rad] obliquity, ref: Stiles et al. 2008, Section 5
	#gamma = 0.091/rd # [deg->rad] Cassini plane offset, ref: Stiles et al. 2008 
	#theta = 0.31/rd # [deg->rad] obliquity, ref: Meriggiola et al 2016
	theta = 0.32/rd # [deg->rad] obliquity, ref: Baland+2011, 0.32 +/- 0.02 deg
	gamma = 0.12/rd # [deg->rad] Cassini plane offset, ref: Baland+2011, 0.12 +/- 0.02 deg, pos b/c phase lag
	phi_CS = np.pi # [rad] Azimuth of spin axis without dissipation, Titan is in Cassini state 1, so phi_CS = pi
	phi = phi_CS + np.arcsin( np.sin(gamma)/np.sin(theta) )# [rad] Azimuth of spin axis with dissipation, measured from x-axis

	''' Not measured '''
	k2oQ = 0 # [ ] tidal dissipation factor
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	f_cmb = 0 # [ ] oblateness of ice shell-ocean interface (aka cmb)
	
	name = 'Titan'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_cmb, gamma, phi_CS, phi, name)

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
	phi_CS = np.pi # [rad] Azimuth of spin axis w/o dissipation, Io is predicted to be in Cassini state 1
	phi = phi_CS # [rad] Azimuth of spin axis w/ dissipation
	f_cmb = 0 # [ ] oblateness of crust-magma ocean boundary

	name = 'Io'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_cmb, gamma, phi_CS, phi, name)

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
	f_cmb = 0 # [ ] oblateness of ice shell-ocean interface
	theta = 0 # [rad] obliquity
	gamma = 0 # [rad] Cassini plane offset
	phi_CS = np.pi # [rad] Azimuth of spin axis w/o dissipation, Io is predicted to be in Cassini state 1
	phi = phi_CS # [rad] Azimuth of spin axis w/ dissipation
	
	name = 'Europa'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_cmb, gamma, phi_CS, phi, name)

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
	phi_CS = np.pi # [rad] Azimuth of spin axis w/o dissipation, Io is predicted to be in Cassini state 1
	phi = phi_CS # [rad] Azimuth of spin axis w/ dissipation
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	k2oQ = 0 # [ ] tidal dissipation factor
	f_cmb = 0 # [ ] oblateness of ice shell-ocean interface
	
	name = 'Ganymede'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_cmb, gamma, phi_CS, phi, name)

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
	phi_CS = np.pi # [rad] Azimuth of spin axis w/o dissipation, Io is predicted to be in Cassini state 1
	phi = phi_CS # [rad] Azimuth of spin axis w/ dissipation
	KoC = 0 # [1/s] ice shell-ocean boundary dissipation factor
	k2oQ = 0 # [ ] tidal dissipation factor
	f_cmb = 0 # [ ] oblateness of ice shell-ocean interface
	
	name = 'Callisto'
	return(Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_cmb, gamma, phi_CS, phi, name)

########################
#                      #
# SPIN ANGLE EQUATIONS #
#                      #
########################

def prec_fig(n, inc, theta, eta, c, J2, C22):
	''' (Helper function)
	- Component of precession of spin axis due to torques on permanent figure '''
	fig = 3/2 * n/c * ((J2+C22)*np.cos(theta)+C22)
	orb = eta*np.cos(inc)
	return( fig + orb )
	
def tid(Mp, M, R, a, n, theta, c, k2oQ):
	''' (Helper function)
	- Component of tidal torque exerted on satellite
	Ref: Gladman et al. 1996, Appendix A, Eq. 49 '''
	T = 3 * k2oQ * n/c * Mp/M * (R/a)**3
	return( T * (1-1/2*np.cos(theta) ) )

def cmb(KoC, delta_epsilon):
	''' (Helper function) 
	- Component of torque due to differential rotation at core-mantle boundary
	Ref: Williams et al. 2001 '''
	return( KoC * np.sin(delta_epsilon) )
    
def tan_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Steady-state spin axis: dsx/dt = 0 -> tan(phi)
	Manuscript: Eq. 16 Main Text '''
	tid_term = tid(Mp, M, R, a, n, theta, c, k2oQ)
	cmb_term = cmb(KoC, de) / np.sin(theta)
	prec = prec_fig(n, inc, theta, eta, c, J2, C22)
	return( (tid_term+cmb_term)*np.cos(theta)/prec )

def sin_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Steady-state spin axis: dsz/dt = 0 -> sin(phi)
	Manuscript: sy in Eqs. 14 '''
	tid_term = tid(Mp, M, R, a, n, theta, c, k2oQ) * np.sin(theta)
	cmb_term = cmb(KoC, de)
	prec = eta * np.sin(inc)
	return( (tid_term+cmb_term)/prec )

def cos_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22):
	''' Steady-state spin axis: cos(phi) = sin(phi)/tan(phi)
	Manuscript: sx in Eqns. 16 '''
	prec_figure = prec_fig(n, inc, theta, eta, c, J2, C22)
	prec_orbit = eta * np.sin(inc)
	return( np.tan(theta)*prec_figure/prec_orbit )

def CS_spin_axis(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de):
	''' Components of the equilibrium spin axis: sx, sy, sz
	Manuscript: Eqns. 16, sy = sin(gamma) is Eq. 1'''
	sx = np.sin(theta) * cos_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22)
	sy = np.sin(theta) * sin_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, de)
	sz = np.cos(theta)
	return(sx, sy, sz)

####################
#                  #
# OBLIQUITY SOLVER #
#                  #
####################

def CSR_T_KoC(o, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ, KoC, de):
	''' Cassini state relation with tidal and CMB dissipation
	Manuscript: Eq. 15 '''
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

##########################################
#                                        #
# DISSIPATION PARAMETERS FROM SPIN STATE #
#                                        #
##########################################

def k2oQ_from_offset(Mp, M, R, a, n, inc, theta, eta, c, gamma):
	''' Solve for k2/Q from sin(gamma) (Eq. 1) '''
	tid_mod = 3 * n/c * Mp/M * (R/a)**3 * (1-1/2*np.cos(theta)) # Same term as tid() except NO k2/Q
	prec_orb = eta * np.sin(inc)
	return( np.sin(gamma) * prec_orb / np.sin(theta)**2 / tid_mod )

def k2oQ_from_phi(Mp, M, R, a, n, inc, theta, eta, c, phi):
	''' Solve for k2/Q from sin(phi) (Eq. 1 or Eq. 14)
	Manuscript: used to find endmember k2/Q
	- Figure 2
	- Eq. 8 in SM + uncertainty analysis '''
	tid_mod = 3 * n/c * Mp/M * (R/a)**3 * (1-1/2*np.cos(theta)) # Same term as tid() except NO k2/Q
	prec_orb = eta * np.sin(inc)
	return( np.sin(phi) * prec_orb / np.sin(theta) / tid_mod )

def KoC_from_offset(theta, inc, eta, gamma, de):
	''' Solve for K/C from sin(gamma) (Eq. 1) '''
	cmb_mod = np.sin(de) # Same term as cmb() except NO K/C
	prec_orb = eta * np.sin(inc)
	return( np.sin(gamma) * prec_orb / np.sin(theta) / cmb_mod )

def KoC_from_phi(inc, eta, phi, de):
	''' K/C from sin(phi) (Eq. 1 or Eq. 14)
	Manuscript: used to find endmember K/C
	- Figure 2 '''
	cmb_mod = np.sin(de) # Same term as cmb() except NO K/C
	prec_orb = eta * np.sin(inc)
	return( np.sin(phi) * prec_orb / cmb_mod )

def k2oQ_remainder(moon, KoC_var, de, prin=0):
	'''
	Step 1) What offset does K/C_var*sin(de) produce
	Step 2) What offset remains
	Step 3) OUTPUT what k2/Q produces the remainder of the offset
	MANUSCRIPT: helper function for Fig 2 (graph_dissipation_partition())
	'''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma, phi_CS, phi, name = moon()

	sin_phi_cmb = sin_phi(Mp, M, R, a, n, inc, theta0, eta, c, J2, C22, k2oQ=0, KoC=KoC_var, de=de)
	sin_phi_tid = np.sin(phi) - sin_phi_cmb
	
	if prin :
		print('sin(phi)_tides/sin(phi) %.4g'%(sin_phi_tid/np.sin(phi)))
		print('sin(phi)_cmb/sin(phi) %.4g'%(sin_phi_cmb/np.sin(phi)))

	k2oQ_rem = k2oQ_from_phi(Mp, M, R, a, n, inc, theta0, eta, c, np.arcsin(sin_phi_tid))
	return(k2oQ_rem)

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
	return(np.sqrt(dk2oQ_2))

#################################
#                               #
# MOMENT OF INERTIA AND DENSITY #
#                               #
#################################

def moi_ellipsoid(R_e, rho_e, f_e):
	''' Polar moment of inertia of a solid ellipsoid [kg m^2], one-layer
	Notes: R_e, rho_e, f_e are the radius, density, and flattening of the ellipsoid '''
	return( 8*np.pi/15 * rho_e * R_e**5 * (1+2/3*f_e) )

def moi_layer(R_top, R_bottom, rho, f_top, f_bottom):
	''' Polar moment of inertia of an isolated layer of a body [kg m^2], one-layer
	Notes: (R_t, f_t), (R_b, f_b) are the radius and flattening of the top and bottom of the layer
	Manuscript: Eq. 11 of Supplemental Materials
	- used in Titan's Free Ocean Nutation
	- Fig. S1 '''
	return( 8*np.pi/15 * rho * (R_top**5 * (1+2/3*f_top) - R_bottom**5 * (1+2/3*f_bottom)) )

def moi_three_layers(R, R_cmb, R_icb, rho_surface, rho_fluid, rho_interior, f_surface, f_cmb, f_icb):
	''' Polar moments of inertia of each layer of a three layer body '''
	C_surface = moi_layer(R, R_cmb, rho_surface, f_surface, f_cmb) # [kg m^2] Polar moment of inertia of the surface solid layer (ice shell or mantle)
	C_fluid = moi_layer(R_cmb, R_icb, rho_fluid, f_cmb, f_icb) # [kg m^2] Polar moment of inertia of the fluid layer (ocean or fluid outer core)
	C_interior = moi_ellipsoid(R_icb, rho_interior, f_icb) # [kg m^2] Polar moment of inertia of solid interior beneath fluid layer
	return(C_surface, C_fluid, C_interior)

def density_interior_3layer(M, R, R_cmb, R_icb, rho_top, rho_middle):
	''' For a body with three layers (top - middle - interior), find density of interior
	Notes: M = M_top + M_middle + M_interior
	Notes: R_cmb, R_icb are radii of top/middle and middle/interior boundaries
	Usage: Used to find Titan's interior density given ice and ocean thicknesses '''
	rho_bulk = M / (4/3*np.pi*R**3)
	one = (rho_bulk - rho_top) * (R/R_icb)**3
	two = (rho_top - rho_middle) * (R_cmb/R_icb)**3
	rho_interior = rho_middle + one + two
	return( rho_interior )

def density_surface_2layer(M, R, R_cmb, rho_interior):
	''' For a body with two layers (surface - interior), find density of surface
	Usage: Used to find average density of Moon's crust and mantle '''
	rho_bulk = M / (4/3*np.pi*R**3)
	one = rho_bulk*R**3 - rho_interior*R_cmb**3
	two = R**3 - R_cmb**3
	rho_surface = one/two
	return( rho_surface )

def moi_density_FCN_Titan(d, h, rho_surface, rho_fluid, f_surface, f_cmb, f_icb):
	''' Returns radius, density, and MOI of different layers and FON frequency
	Usage: in solve_KoC() and graph_Titan_free_ocean_nutation() '''
	Mp, M, R, a, e, inc, theta, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma, phi_CS, phi, name = Titan()

	R_cmb = R - d # [m] Radius of the solid-liquid boundary aka "core-mantle boundary" per Earth/Moon nomenclature
	R_icb = R - d - h # [m] Radius of the liquid-interior boundary aka "inner core boundary" per Earth/Moon nomenclature

	rho_interior = density_interior_3layer(M, R, R_cmb, R_icb, rho_surface, rho_fluid) # [kg m^2] Density of rocky mantle interior to ocean

	C = c*M*R**2 # [kg m^2] Moment of inertia of the whole body
	Cs = moi_three_layers(R, R_cmb, R_icb, rho_surface, rho_fluid, rho_interior, f_surface, f_cmb, f_icb)
	C_surface, C_fluid, C_interior = Cs

	FCN_freq = fcn_thin_top(M, R, R_cmb, R_icb, C, C_surface, C_interior, c, J2, C22, f_cmb, f_icb, e, n) # [1/s] FON frequency, ref: Dumberry 2021, Eq. 38a
	return(R_cmb, R_icb, rho_interior, C_surface, C_fluid, C_interior, FCN_freq)

############################
#                          #
# FREE CORE/OCEAN NUTATION #
#                          #
############################

def fcn(f_cmb, n, C, C_m):
	''' Free core nutation (of fluid core)
	Signs: Positive -> PROGRADE precession
	Ref: Meyer Wisdom 2011 Eq. 1 (for the Moon)
	Manuscript: Eq. 7 of the Supplementary Materials '''
	return( -f_cmb * n * C / C_m )

def fcn_thin_top(M, R, R_cmb, R_icb, C, C_ice, C_int, c, J2, C22, f_cmb, f_icb, e, n):
	''' Free core nutation for body with thin top solid layer over fluid layer
	Sign convention: Positive -> PROGRADE precession
	Ref: Dumberry 2021 Eq. 38a
	Manuscript: Eq. 8 of the Supplementary Materials '''
	e = 0 # Assume a circular orbit for Titan
	
	G210 = (1-e**2)**(-3/2) # [ ] J2 eccentricity term, Eq. 16a
	G201 = 7/2*e - 123/16*e**3 + 489/128*e**5 # [ ] C22 3:2 spin-orbit res eccentricity term, Eq. 16b
	G_synch = 1-5/2*e**2 # [ ] C22 1:1 spin-orbit res eccentricity term, Table I Peale 1969
	f = J2/c # [ ] J2 term, Eq 4
	gamma = 4 * C22/c # [ ] C22 term, Eq 4

	###phi_m = 3/2 * (G210*f + 1/2*G201*gamma) # [ ] Mercury 3:2, gravitational torques on whole body, Eq. 15a
	phi_sat = 3/2 * ( G210*f + G_synch*1/2*gamma ) # [ ] Titan 1:1, gravitational torques on whole body

	C_rest = C_ice + C_int # [kg m^2] Polar moment of intertia of everything but the fluid layer

	e_f = (R_cmb**5*f_cmb - R_icb**5*f_icb)/(R_cmb**5*(1+2/3*f_cmb) - R_icb**5*(1+2/3*f_icb)) # [ ] Dynamical ellipticity of fluid, ref: DW2016 Eq. 15b
	one = - n * (C/C_rest) * (e_f + phi_sat) 
	two = n * e_f * phi_sat / (e_f + phi_sat)
	return( one + two )

##################################
#                                #
# DIFFERENTIAL ROTATION FRICTION #
#                                #
##################################

def d_omega_viscous(n, eta, inc, theta, phi_CS, xi):
	''' Differential rotational velocity at the solid-fluid boundary
	Note: Viscous coupling only
	Reference: Williams et al. (2001) Eq. (53), Daher et al. (2021) Eq. (53)
	Manuscript: Eq. 15 of SM '''
	dFdt = n - eta
	angle = theta*np.cos(phi_CS) - inc # cos(phi) gives you quadrant of theta
	return( dFdt * np.abs(np.sin(angle)) / np.sqrt(1+xi**2) )

def d_omega_pressure(C, C_fluid, n, eta, theta, phi_CS, inc, FCN):
	''' Differential rotational velocity at the solid-fluid boundary
	Note: Pressure coupling only
	Reference: Derived from Meyer and Wisdom 2011 Appendix A
	Manuscript: sinK is Eq. 6 of SM '''
	delta = C_fluid/C
	angle = theta*np.cos(phi_CS) - inc # cos(phi) gives you quadrant of theta
	sinK = -1/(1-delta) * np.abs(eta)/(-FCN-np.abs(eta))*np.sin(angle)
	return( n * np.abs(sinK) )

def d_omega_pressure_viscous(C, C_fluid, n, eta, theta, phi_CS, inc, FCN, xi):
	''' Differential rotational velocity at the solid-fluid boundary
	Note: Pressure & viscous coupling, combination
	Manuscript: Eq. 16 of SM '''
	term = d_omega_pressure(C, C_fluid, n, eta, theta, phi_CS, inc, FCN)
	return( term / np.sqrt(1+xi**2) )

def K_viscous(d_omega, rho_f, R_cmb, kappa):
	''' Viscous dissipation parameter at solid-fluid boundary
	Note: Turbulent equation (as opposed to laminar) depends on skin friction (kappa)
	Ref: Williams et al. (2001) Eq. (55)
	Manuscript: Eq. 12 of SM '''
	return( 3/4 * np.pi**2 * kappa * rho_f * R_cmb**5 * np.abs(d_omega) )

def xi_viscous(K, C_fluid, eta):
	''' Visous coupling parameter at solid-fluid boundary
	Ref: Williams et al. (2001) Sec. 10, Daher et al. (2021), Eq. (49)
	Manuscript: Eq. 14 of SM '''
	return( -K/C_fluid / eta )

def K_func(K, kap, d_omega, rho_f, R_cmb):
	''' Turbulent K, same as K_viscous() but in form lhs - rhs = 0
	Usage: Form of equation for viscous_funcs() '''
	return( K - 3/4 * np.pi**2 * kap * rho_f * R_cmb**5 * np.abs(d_omega) )

def kappa_func(kap, n, eta, theta, phi_CS, inc, R_cmb, nu, xi):
	''' Skin friction parameter in form lhs - rhs = 0
	Ref: Williams et al. (2001) Eq. (58a)
	Usage: Form of equation for viscous_funcs() '''
	dFdt = n - eta
	angle = theta*np.cos(phi_CS) - inc # cos(phi) gives you quadrant of theta
	one = np.log( 0.4 * np.sqrt(kap) * R_cmb**2 * dFdt * np.sin(angle)**2 )
	two = np.log( nu * (1+xi**2) )
	return( np.sqrt(kap)*(one - two) - 0.4 )
	
def viscous_funcs(x, R_cmb, rho_f, n, eta, inc, theta, phi_CS, nu, C, C_fluid, FCN):
	''' Equations to simultaneously solve for K, kappa, xi, and delta_omega
	Ref: Williams et al. (2001), iterative solver was their method
	Choose: 3 forms of delta_omega, with or without pressure and viscous effects
	Usage: Used in solve_KoC()'''
	K, kap = x # "x" are unknowns that solve_KoC() will solve for
	
	xi = xi_viscous(K, C_fluid, eta)

	# * CHOOSE which functional form of delta_omega you want to use
	#d_omega = d_omega_viscous(n, eta, inc, theta, phi_CS, xi)
	#d_omega = d_omega_pressure(C, C_fluid, n, eta, theta, phi_CS, inc, FCN)
	d_omega = d_omega_pressure_viscous(C, C_fluid, n, eta, theta, phi_CS, inc, FCN, xi)

	K_eqn = K_func(K, kap, d_omega, rho_f, R_cmb)
	kap_eqn = kappa_func(kap, n, eta, theta, phi_CS, inc, R_cmb, nu, xi)
	
	return([K_eqn, kap_eqn])

#######################
#                     #
# OCEAN TIDAL HEATING #
#                     #
#######################

def Edot_obl_ocean_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2, rho):
	''' [W] Energy dissipation due to obliquity tides in a subsurface ocean on a synchronous satellite
	Ref: Hay & Matsuyama 2019 - scaling laws
	Notes: nu is the turbulent viscous diffusivity, other parameters described in Ref
	Manuscript: plotted in Fig 3a, used in calculate_Titan_ocean_tides() '''
	nu = nu_obl_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2)
	A = 12*np.pi * rho * h * nu * (Omega*R*theta*u2*(R/rt))**2
	B = 1 + (20*u2*beta2*nu*g*h/(Omega**3*rt**4))**2
	return(A/B)

def nu_obl_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2):
	''' Turbulent viscous diffusivity from obliquity tide flow
	Ref: Hay & Matsuyama 2019 - scaling laws
	Notes: to be used in Edot_obl_ocean_HM19 and vel_obl_Chen14 '''
	C = Omega**3 * rt**4 / (20 * np.sqrt(2) * beta2 * g * h)
	D = 200 / 3 * 0.4 * cd * beta2 * u2 * g * R**2 * theta / (Omega**2 * rt**3)
	return(C * np.sqrt(-1+np.sqrt(1+D**2)))

def vel_obl_Chen14(R, Omega, theta, g, h, cd, beta2, u2, rt):
	''' Average obliquity flow velocity
	Ref: Chen et al. 2014, Table 4 - scaling laws
	Notes: uses turbulent viscous diffusivity from HM19 '''
	u0 = (Omega*R)**3 / (g*h)
	num = np.sqrt(3/2) * Omega * R * theta
	denom = np.sqrt( 1 + 400*nu_obl_HM19(Omega, rt, g, h, R, theta, cd, beta2, u2)**2 / (u0**2 * R**2) )
	return(num/denom)

def cd_Fan19(cd, R, n, theta, g, h, eta, b2, u2, d, rho):
	''' Ocean bottom drag coefficient cD using empirical scaling laws from Earth oceans
	Ref: Fan et al. 2019
	Notes:
	- cD is a fcn of Reynolds number which is a fcn of flow velocity, which is a fcn of cD (circular dependency)
	- Circular dependency means need to solve numerically
	- Uses Chen+2014 velocity flow equation and Hay & Matsuyama 2019 viscous diffusivity equation
	Manuscript: used in calculate_Titan_ocean_tides(), helper function for Edot_obl_ocean_HM19
	e.g., cd = root_scalar(cd_Fan19, args=(R, n, theta0, g, h, eta_water, b2, u2, d, rho_f), x0=2e-3, x1=1e-4).root '''
	rhs = cd
	vel = np.abs(vel_obl_Chen14(R, n, theta, g, h, cd, b2, u2, R-d))
	lhs = 0.395*(vel*h*rho/eta)**(-0.518) # scaling law for cD: cD as a function of the Reynolds number
	return(rhs - lhs)

def calculate_Titan_ocean_tides():
	''' Titan's ocean obliquity tidal heating
	Manuscript: numbers appear in-text, helper function for Figure 3a
	e.g., cd, Edot_obl_ocean = calculate_Titan_ocean_tides()'''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, fc0, gamma, phi_CS, phi, name = Titan()

	rho_f = 1000 # [kg/m^3] density of fluid layer where turbulence is, aka water ocean
	eta_water = 1e-3 # [Pa s] viscosity of water, ref: Sohl et al. 1995
	g = 1.35 # [m/s^2] Titan's surface gravity, ref: Chen et al. 2014 
	h = 150e3 # [m] Titan's approximate ocean thickness

	# Coefficients of order unity described in Hay & Matsuyama 2019
	b2 = 1 # [ ] beta_2 = shell pressure forcing, ref: Hay & Matsuyama 2019
	u2 = 1 # [ ] upsilon_2 = perturbation to the forcing tidal potential due to shell pressure forcing, self-gravity, and deformation of the solid regions, ref: HM2019
	d = 100e3 # [m] ice shell thickness 50-200km, ref: Hemingway et al. 2013

	# Calculate bottom drag from Reynolds number scaling law (Fan et al. 2019 scaling law for Earth used here)
	# * could also use scaling law in Sohl et al. 1995
	cd = root_scalar(cd_Fan19, args=(R, n, theta0, g, h, eta_water, b2, u2, d, rho_f), x0=2e-3, x1=1e-4).root
	#cd = 4e-3 # cD = 2e-3 for Earth's oceans, cD = 4e-3 for two boundary layers, ref: Hay & Matsuyama 2019
	
	# Calculate obliquity tidal heating from scaling law (Hay & Matsuyama 2019 include effect of overlying ice shell)
	Edotoo = Edot_obl_ocean_HM19(n, R-d, g, h, R, np.abs(theta0), cd, b2, u2, rho_f)
	return(cd, Edotoo)

###############################
#                             #
# ORBITAL AND TIDAL EQUATIONS #
#                             #
###############################

def Edot_ecc_solid(e, R, n, k2oQ):
	''' Tidal dissipation - solid-body eccentricity [W]
	References: Peale & Cassen 1978, e.g., Wisdom 2004
	Manuscript: Eq. 2, used in Fig. 3 '''
	Edot_ecc = 3/2 * k2oQ * (n*R)**5 / G * 7*e**2
	return( Edot_ecc )

def Edot_obl_solid(theta, R, n, k2oQ):
	''' Tidal dissipation - solid-body obliquity [W]
	References: Peale & Cassen 1978, e.g., Wisdom 2004
	Manuscript: Eq. 2, used in Fig. 3 '''
	Edot_obl = 3/2 * k2oQ * (n*R)**5 / G * np.sin(theta)**2
	return( Edot_obl )

def Edot_differential_rotation(K, delta_omega):
	''' Power dissipated because of differential rotation at the solid-fluid boundary [W]
	Reference: Williams et al. 2001 Eq. 81a
	Manuscript: Eq. 3 '''
	Edot_cmb = K * delta_omega**2
	return( Edot_cmb )

def deriv_a(Mp, M, a, Edot):
	''' da/dt due to energy dissipation in the satellite [m/s]
	E.g. reference: Chyba et al. 1989
	Notes: Edot (energy dissipation) is positive number, so negative sign to decrease a
	Notes: Input either tidal dissipation (Edot_tid) or differential rotation dissipation (Edot_cmb)
	Manuscript: Eq. 4 '''
	dadt = - 2 * a**2 * Edot / (G*Mp*M)
	return( dadt )

def deriv_ecc(Mp, M, a, e, Edot_ecc):
	''' de/dt due to eccentricity tidal heating in the satellite [1/s]
	E.g. reference: Chyba et al. 1989
	Notes: Edot (dissipation) is positive number, so negative sign to decrease e
	Manuscript: Eq. 6, used in Fig 3b '''
	dedt = - a/(G*Mp*M) * (1-e**2)/e * Edot_ecc
	return( dedt )

def deriv_inc(Mp, M, a, inc, Edot):
	''' di/dt due to a) obliquity tidal heating b) differential rotation dissipation in the satellite [rad/s]
	E.g. reference: Chyba et al. 1989
	Notes: Edot (dissipation) is positive number, so negative sign to decrease i
	Notes: Input either tidal dissipation (Edot_tid) or differential rotation dissipation (Edot_cmb)
	Manuscript: Eq. 5, used in Fig 3b '''
	didt = - a/(G*Mp*M) / np.tan(inc) * Edot
	return( didt )

def heat_flux(Mp, R, n, a, e, theta, k2oQ, K, delta_omega, prin):
	''' Surface heat flux [W/m^2]: solid-body tidal heating (eccentricity and obliquity) and differential rotation
	Manuscript: Fig 3a right y-axis '''
	Edot_ecc = Edot_ecc_solid(e, R, n, k2oQ)
	Edot_obl = Edot_obl_solid(theta, R, n, k2oQ)
	Edot_cmb = Edot_differential_rotation(K, delta_omega)
	
	Edot = Edot_ecc + Edot_obl + Edot_cmb
	flux = Edot / (4*np.pi*R**2)

	if prin :
		print('Edot-k2oQ / Edot-tot = %.3g'%( (Edot_ecc+Edot_obl)/Edot ))
		print('Edot-KoC / Edot-tot = %.3g'%( Edot_cmb/Edot ))
		print('Edot-k2oQ %.3g, Edot-KoC %.3g'%(Edot_ecc+Edot_obl, Edot_cmb))
		
	return(Edot, flux)
    
def mean_motion(Mp, a):
	''' Orbital mean motion '''
	n2 = G*Mp/a**3
	return( np.sqrt(n2) )

def timescale_ecc(Mp, M, R, a, n, e, k2oQ):
	''' Eccentricity damping timescale due to tidal heating in the satellite [s]
	e / (de/dt)
	Manuscript: Fig 3b '''
	Edot_ecc = Edot_ecc_solid(e, R, n, k2oQ)
	dedt = deriv_ecc(Mp, M, a, e, Edot_ecc) # Negative
	taue = np.abs( e / dedt ) # Abs because de/dt is negative
	return(Edot_ecc, taue)

def timescale_inc(Mp, M, R, a, n, inc, theta, k2oQ, K, delta_omega, prin):
	''' Inclination damping timescale due to energy dissipation in the satellite [s]
	tan(inc) / (di/dt)
	Notes: Use tidal dissipation and/or differential rotation dissipation
	Manuscript: Fig 3b '''
	Edot_obl = Edot_obl_solid(theta, R, n, k2oQ)
	Edot_cmb = Edot_differential_rotation(K, delta_omega)
	Edot = Edot_obl + Edot_cmb

	didt_tid = deriv_inc(Mp, M, a, inc, Edot_obl)
	didt_cmb = deriv_inc(Mp, M, a, inc, Edot_cmb)
	
	didt_tot = didt_tid + didt_cmb # Negative number
	taui = np.abs( np.tan(inc) / didt_tot ) # Absolute value because di/dt is negative

	if prin :
		print('didt-k2oQ / didt-tot = %.3g'%(didt_tid/didt_tot))
		print('didt-KoC / didt-tot = %.3g'%(didt_cmb/didt_tot))
	
	return(Edot, taui)
	
###########################################
#                                         #
# FUNCTIONS TO PRINT INFO/GENERATE GRAPHS #
#                                         #
###########################################

def calculate_dissipation_from_spin(moon, delta_epsilon=0, prin=1):
	''' Endmember k2/Q and K/C if all of offset due to each dissipation mechanism
	CHOOSE: moon = Moon or Titan
	NOTE: You might not have a delta_epsilon in mind, so leave as zero and it will be set to spin-Laplace pole angle 
	e.g., end_k2oQ_M, end_KoC_M = calculate_dissipation_from_spin(Moon)
	e.g., end_k2oQ_T, end_KoC_T = calculate_dissipation_from_spin(Titan) '''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma0, phi_CS, phi0, name = moon()

	# Default: delta_epsilon is angle b/w spin and Laplace poles
	if delta_epsilon == 0 : delta_epsilon = np.abs(theta0*np.cos(phi_CS)-inc)

	end_k2oQ = k2oQ_from_phi(Mp, M, R, a, n, inc, theta0, eta, c, phi0)
	end_KoC = KoC_from_phi(inc, eta, phi0, delta_epsilon)

	if moon == Moon:
		dgamma = 0 # Uncertainty in offset (might exist, but I didn't look for)
		dtheta = 0 # Uncertainty in obliquity (might exist, but I didn't look for)
		dc = 0 # Uncertainty in polar moment of inertia (might exist, but I didn't look for)
	elif moon == Titan:
		dgamma = 0.02/rd # [rad] Baland+2011 Sec 3.2 derive own uncertainties for Cassini plane offset, 0.02 deg
		dtheta = 0.02/rd # [rad] Baland+2011 Sec 3.2 derive own uncertainties for obliquity, 0.02 deg
		dc = 0.03 # This is an estimate: nominal = 0.341, highest = 0.36 (Hemingway+2013), lowest = 0.31 (Baland+2014)
		
	end_k2oQ_unc = dk2oQ(Mp, M, R, a, n, inc, theta0, eta, c, gamma0, end_k2oQ, dgamma, dtheta, dc)

	if prin == 1:
		print('%s'%name)
		print('Endmember: k2/Q %.4g, K/C %.4g [1/s] (delta-epsilon %.4g [deg])'%(end_k2oQ, end_KoC, delta_epsilon*rd))
		print('Uncertainty in k2/Q: %.4g'%end_k2oQ_unc)
		# Would need K or K/C & C_surface if wanted K/C endmember in calculate_heat_damping
		heat_output = calculate_heat_damping(moon, end_k2oQ, K_test=0, delta_epsilon=delta_epsilon, prin=1) 
		
	return(end_k2oQ, end_KoC)

def calculate_spin_from_dissipation(moon, k2oQ_test=0, KoC_test=0, delta_epsilon=0, prin=1):
	''' Spin state with k2/Q and K/C (spherical and Cartesian coordinates)
	CHOOSE: moon = Moon, Titan, Io, Europa, Ganymede, Callisto
	DEFAULT: Moon - use LLR k2/Q and K/C; Titan - use end-member k2/Q
	NOTE: Same as above, if delta_epsilon=0, it will be set to spin-Laplace pole angle
	e.g., theta_pred, phi_pred, gamma_pred, sx_p, sy_p, sz_p = calculate_spin_from_dissipation(Moon)
	e.g., theta_pred, phi_pred, gamma_pred, sx_p, sy_p, sz_p = calculate_spin_from_dissipation(Titan) '''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma0, phi_CS, phi0, name = moon()

	# Default: Use determined k2oQ0 if exists, otherwise endmember k2/Q
	if k2oQ_test == 0 and KoC_test== 0 :
		k2oQ = k2oQ0
		KoC = KoC0
		if k2oQ0 == 0 :
			k2oQ = k2oQ_from_phi(Mp, M, R, a, n, inc, theta0, eta, c, phi0)
	else :
		k2oQ = k2oQ_test
		KoC = KoC_test

	# Default: delta_epsilon is angle b/w spin and Laplace poles
	if delta_epsilon == 0 : delta_epsilon = np.abs(theta0*np.cos(phi_CS)-inc)

	theta_guess = inc
	theta_pred = obliquity(theta_guess, Mp, M, R, a, n, inc, eta, c, J2, C22, k2oQ, KoC, delta_epsilon) # Incorrect for Titan b/c not a purely rigid body

	if theta0 == 0 : theta = theta_pred # Use predicted obliquity only if no determined obliquity
	else : theta = theta0

	sx_p, sy_p, sz_p = CS_spin_axis(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, delta_epsilon) 
	sin_phi_pred = sin_phi(Mp, M, R, a, n, inc, theta, eta, c, J2, C22, k2oQ, KoC, delta_epsilon)

	phi_pred = phi_CS + np.cos(phi_CS)*np.arcsin(sin_phi_pred) # CS1: pi+|phi_pred|, CS2: 0-|phi-pred|
	gamma_pred = np.arcsin(sy_p)*np.cos(phi_CS) # CS1: pos phase lag, CS2: neg phase lead
	
	if prin :
		print('%s'%name)
		print('Observed diss. factors: k2/Q %.3g, K/C %.3g [1/s]'%(k2oQ0, KoC0))
		print('Observed angles: theta %.3g [deg], phi %.3g [deg], gamma %.3g [deg]'%(theta0*rd, phi0*rd, gamma0*rd))

		print('\nTest diss. factors: k2/Q %.3g, K/C %.3g [1/s]'%(k2oQ, KoC))
		print('Calculated angles: theta %.3g [deg], phi %.3g [deg], gamma %.3g [deg]'%(theta_pred*rd, phi_pred*rd, gamma_pred*rd))
		print('Calculated Cartesian: sx %.3g, sy %.4g, sz %.3g'%(sx_p, sy_p, sz_p))
	return(theta_pred, phi_pred, gamma_pred, sx_p, sy_p, sz_p)

def solve_KoC(moon, option='', d0=50e3, h0=200e3, f_option='low', prin=1):
	''' Simultaneously solve for K, kappa, xi, and delta_omega 
	CHOOSE: moon = Moon or Titan (no f_cmb for Jovian satellites)
	CHOOSE: Titan only - thickness of ice shell and ocean, flattenings of cmb and icb
	NOTATION: cmb = core-mantle boundary aka solid-liquid boundary, icb = inner core boundary aka liquid-interior boundary
	e.g., output = solve_KoC(Moon)
	e.g., output = solve_KoC(Titan, '', d0=100e3, h0=100e3, f_option='high', prin=1) '''
	Mp, M, R, a, e, inc, theta, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma, phi_CS, phi, name = moon()
	C = c*M*R**2 # [kg m^2] Polar moment of inertia of the whole body
	nu = 1e-6 # [m^2/s] Kinematic viscosity of liquid

	''' Moon and Titan specific parameters '''
	# * Moon: R_cmb and core density are specified, has its own FCN equation
	if moon == Moon:
		R_cmb = 380e3 # [m] Radius of core-mantle boundary, ref: Viswanathan+2019
		R_icb = 0 # [m] Radius of inner core boundary (can be changed at some point)
		d = R - R_cmb # [m] Thickness of the crust+mantle
		h = R_cmb - R_icb # [m] Thickness of the fluid outer core

		rho_interior = 5812 # [kg/m^3] Density of core, ref: Viswanathan+2019 Table B3, row for R_cmb = 380 km
		rho_fluid = rho_interior # [kg/m^3] Density of liquid outer core (fluid), assume FOC and SIC have same density
		rho_surface = density_surface_2layer(M, R, R_cmb, rho_interior) # [kg/m^3] Density of mantle (surface), solved to be consistent with R_cmb and rho_interior
	
		f_surface, f_cmb, f_icb = 0, f_cmb0, 0 # [ ] Polar flattenings of surface, CMB, and ICB

		Cs = moi_three_layers(R, R_cmb, R_icb, rho_surface, rho_fluid, rho_interior, f_surface, f_cmb, f_icb)
		C_surface, C_fluid, C_interior = Cs # Moments of inertia of mantle (surface), fluid outer core, solid inner core

		FCN_freq = fcn(f_cmb, n, C, C_surface) # [1/s] Free Core Nutation frequency for the Moon, large solid mantle, smaller interior
		FCN_period = 2*np.pi/FCN_freq # [s] FCN period

	# * Titan: Densities are specified, need to input d, h, and flattenings, has its own FCN equation, option to use B+19 parameters
	elif moon == Titan:
		KoC0 = 5e-11 # [1/s] Initial guess for K solver

		f_surface = 19.2236e-5 # [ ] Surface polar flattening, ref: Baland et al 2014, Eq. 5, from Zebker et al 2009
		f_cmb_low = -2e-4 # [ ] Lower bound on ocean polar flattening, ref: alpha_o Baland+14 Fig. 1 & Coyette+18 Fig. 1
		f_cmb_hig = -4.4e-3 # [ ] Upper bound on ocean polar flattening, ref: alpha_o Baland+14 Fig. 1 & Coyette+18 Fig. 1
		f_icb_low = 6e-5 # [ ] Lower bound on inner solid polar flattening, ref: alpha_m Coyette+18 Fig. 1
		f_icb_hig = 1.06e-4 # [ ] Upper bound on inner solid polar flattening, ref: alpha_m Coyette+18 Fig. 1

		if option == 'Baland19':
			''' Use radii, densities, flattenings, FON period that are in Baland et al. (2019)
			Spoiler: these values are not internally consistent b/c they produce a K/C_shell larger than maximum K/C_shell '''
			d = 110e3 # [m] Thickness of the ice shell, Table 5 TM1 Baland+2019
			h = 200e3 # [m] Thickness of the ocean, Table 5 TM1 Baland+2019

			R_cmb = R - d # [m] Radius of the ice shell-ocean boundary aka "core-mantle boundary" per Earth/Moon nomenclature
			R_icb = R - d - h # [m] Radius of the ocean-interior boundary aka "inner core boundary" per Earth/Moon nomenclature

			rho_surface = 1000 # [kg/m^3] Density of the ice shell (surface), ref: Table 5 TM1 Baland+19
			rho_fluid = 1100 # [kg/m^3] Density of the ocean (fluid), ref: Table 5 TM1 Baland+19
			rho_interior = 2250 # [kg/m^3] Density of the silicate interior, ref: Table 5 TM1 Baland+19

			f_cmb_B19 = 0.8*f_surface # [ ] Ice shell-ocean boundary polar flattening, ref: Baland+19 Table 5, TM1 model
			f_icb_B19 = 0.6*f_surface # [ ] Ocean-interior boundary polar flattening, ref: Baland+19 Table 5, TM1 model
			f_cmb, f_icb = f_cmb_B19, f_icb_B19

			Cs = moi_three_layers(R, R_cmb, R_icb, rho_surface, rho_fluid, rho_interior, f_surface, f_cmb, f_icb)
			C_surface, C_fluid, C_interior = Cs

			FCN_period = -324*s_in_yr # [s] FON period (negative b/c retrograde), ref: Baland+19 Table 6, TM1 aka Titan-like model
			FCN_freq = 2*np.pi/FCN_period # [1/s] FON frequency

		else :
			''' Use input ice shell/ocean thicknesses '''
			d, h = d0, h0 # [m] Use INPUT ice shell/ocean thicknesses

			rho_surface, rho_fluid = 1000, 1000 # [kg/m^3] densities of the surface solid layer and the underlying fluid layer

			if f_option == 'low' : f_cmb, f_icb = f_cmb_low, f_icb_low
			elif f_option == 'high' : f_cmb, f_icb = f_cmb_hig, f_icb_hig

			output = moi_density_FCN_Titan(d, h, rho_surface, rho_fluid, f_surface, f_cmb, f_icb)
			R_cmb, R_icb, rho_interior, C_surface, C_fluid, C_interior, FCN_freq = output
			FCN_period = 2*np.pi/FCN_freq # [s] FCN period, neg if retrograde, pos if prograde


	''' Simultaneously solve K (dissipation), kappa (skin friction), xi (viscous coupling), and delta-omega (diff rotation) '''
	# * Differential rotation effected by pressure coupling (need FCN_freq) and viscous coupling (xi)
	K_guess = KoC0*C
	kappa_guess = 7e-4
	
	root = fsolve(viscous_funcs, [K_guess, kappa_guess], args=(R_cmb, rho_fluid, n, eta, inc, theta, phi_CS, nu, C, C_fluid, FCN_freq)) # [K, kappa]
	K_solved, kappa_solved = root
	
	KoC_solved = K_solved / C_surface # [1/s] K/C_mantle (Moon) or K/C_shell (Titan)
	xi = xi_viscous(K_solved, C_fluid, eta) # [ ] xi << 1 means little coupling, x >=1 means a lot of coupling 
	delta_omega = d_omega_pressure_viscous(C, C_fluid, n, eta, theta, phi_CS, inc, FCN_freq, xi) # [rad/s] Differential rotational velocity b/w surface and fluid
	delta_epsilon = np.arcsin( delta_omega/n ) # [rad] Angular separation b/w surface and fluid, set to always be positive

	# Is K/C_solved < K/C_max? I.e., are parameters self-consistent?
	end_KoC = KoC_from_phi(inc, eta, phi, delta_epsilon) # [1/s] End-member K/C from azimuth phi 
	k2oQ_rem = k2oQ_remainder(moon, KoC_solved, delta_epsilon, prin=0) # [ ] KoC_solved contributes to phi/gamma, what k2/Q to contribute the rest?

	if prin == 1 :
		print('%s'%name)
		print('FCN: %.3g [yrs]'%(FCN_period/s_in_yr))
		print('d: %d [km]'%(d/1e3))
		print('h: %d [km]'%(h/1e3))
		print('rho surface: %d [kg/m^3]'%rho_surface)
		print('rho fluid: %d [kg/m^3]'%rho_fluid)
		print('rho interior: %d [kg/m^3]'%rho_interior)
		print('f_cmb: %.1e'%f_cmb)
		print('f_icb: %.1e'%f_icb)
		print('C_surface/C %.3g, C_fluid/C: %.3g, C_interior/C %.3g'%(C_surface/C, C_fluid/C, C_interior/C))

		print('\nSolver: K/C %.3e [1/s], K %.3e [kg m^2 s^-1], kappa %.3e'%(KoC_solved, K_solved, kappa_solved))
		print('xi %.3g'%xi)
		print('delta-epsilon %.4g [deg]'%(delta_epsilon*rd))
		print('End-member: K/C %.3e [1/s]'%end_KoC)

		sin_delta_epsilon = 1/n * d_omega_pressure(C, C_fluid, n, eta, theta, phi_CS, inc, FCN_freq)
		print('\nCore tilt with pressure (no viscous) effects: %.4g [deg]'%(np.arcsin(sin_delta_epsilon)*rd))
		print('Angle between spin axis and Laplace pole: %.4g [deg]'%(np.abs(theta*np.cos(phi_CS)-inc)*rd))

		print('\nIf K/C = %.4g [1/s], then k2/Q = %.4g'%(KoC_solved, k2oQ_rem))
		repeat = k2oQ_remainder(moon, KoC_solved, delta_epsilon, 1) # Repeat b/c want to print out fraction of gamma from tides/CMB
		print('')
		heat_output = calculate_heat_damping(Titan, k2oQ_rem, K_solved, delta_epsilon, prin=1)
		
	return(K_solved, kappa_solved, xi, delta_epsilon, FCN_period, KoC_solved, k2oQ_rem)

def calculate_heat_damping(moon, k2oQ_test=0, K_test=0, delta_epsilon=0, prin=1):
	''' Heat flux, da/dt, inclination and eccentricity damping timescales for one k2/Q and K
	CHOOSE: moon = Moon, Titan, Io, Europa, Ganymede, Callisto, etc.
	DEFAULT: Moon - use LLR k2/Q and K/C; Titan - use end-member k2/Q
	NOTE K: Recommend using solve_KoC() to get K and de (delta_epsilon)
	e.g., output = calculate_heat_damping(Moon)
	e.g., output = calculate_heat_damping(Titan) '''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma0, phi_CS, phi0, name = moon()

	# Default: Use determined k2oQ0 if exists, otherwise endmember k2/Q
	if np.size(k2oQ_test) > 1 :
		k2oQ = k2oQ_test
		K = K_test               
	elif k2oQ_test == 0 and K_test== 0 :
		k2oQ = k2oQ0
		K = KoC0 * c*M*R**2
		if k2oQ0 == 0 :
			k2oQ = k2oQ_from_phi(Mp, M, R, a, n, inc, theta0, eta, c, phi0)
	else :
		k2oQ = k2oQ_test
		K = K_test
	
	# Default: delta_epsilon is angle b/w spin and Laplace poles
	if delta_epsilon == 0 : delta_epsilon = np.abs(theta0*np.cos(phi_CS)-inc)
	dw = n*np.sin(delta_epsilon)

	Edot_tot, surf_flux = heat_flux(Mp, R, n, a, e, theta0, k2oQ, K, dw, prin=0) # [W], [W], [W/m^2]
	dadt = deriv_a(Mp, M, a, Edot_tot) # Dissipation in satellite only (no planet tides)

	Edot_ecc, tau_ecc = timescale_ecc(Mp, M, R, a, n, e, k2oQ)
	Edot_obl_cmb, tau_inc = timescale_inc(Mp, M, R, a, n, inc, theta0, k2oQ, K, dw, prin=0)

	if prin :
		print('For k2/Q = %.3g and K = %.3g [kg m^2 s^-1],'%(k2oQ, K))
		print('da/dt %.3g [cm/yr], surface flux %.3g [mW/m^2]'%(dadt*100*s_in_yr, surf_flux*1e3))
		print('Tau_inc from CMB and tides: %.3g [yr]'%(tau_inc/s_in_yr))
		print('Tau_ecc from tides: %.3g [yr]\n'%(tau_ecc/s_in_yr))
		repeat1, repeat2 = heat_flux(Mp, R, n, a, e, theta0, k2oQ, K, dw, prin=prin)

	return(Edot_ecc, Edot_obl_cmb, Edot_tot, dadt, tau_ecc, tau_inc)

def graph_dissipation_partition(if_Titan, if_Moon, save=0):
	''' Graph solution curves of constant gamma
		1) Vary K/C
		2) Calculate k2/Q to produce remainder of offset not from variable K/C
		3) Plot
	INPUT: if_Titan = 1(0) -> please (don't) graph Titan; if_Moon = 1(0) -> please (don't) graph Moon
	MANUSCRIPT: Makes Figure 2
	e.g., graph_dissipation_partition(1, 1) '''
	# Figure size
	plt.figure(figsize = (3.55, 3.55*.75))
	if save : plt.rcParams.update({'font.size': 8})
	else : plt.rcParams.update({'font.size': 10})

	if if_Titan == 1 :
		print('Titan')

		''' SOLVER EXAMPLES '''
		'''# Copy and paste this part if graphing these examples in other methods #'''
		# Ice shell thickness d & ocean thickness h pairs chosen to bound max/min K/C & k2/Q
		d_min, d_max = 25e3, 200e3 # [m] Ice shell thickness, min (n) and max (x)
		h_min, h_max = 10e3, 400e3 # [m] Ocean thickness, min (n) and max (x)
		f_option1, f_option2 = 'low', 'low'

		# Solve for K/Cs and k2/Q_remainders for min/max d & h
		indices = [0, 3, 5, 6]
		K_T_dnhn, angle_T_dnhn, KoC_T_dnhn, k2oQ_rem_T_dnhn = [solve_KoC(Titan, option='', d0=d_min, h0=h_min, f_option=f_option2, prin=0)[i] for i in indices]
		K_T_dxhn, angle_T_dxhn, KoC_T_dxhn, k2oQ_rem_T_dxhn = [solve_KoC(Titan, option='', d0=d_max, h0=h_min, f_option=f_option2, prin=0)[i] for i in indices]
		K_T_dnhx, angle_T_dnhx, KoC_T_dnhx, k2oQ_rem_T_dnhx = [solve_KoC(Titan, option='', d0=d_min, h0=h_max, f_option=f_option1, prin=0)[i] for i in indices]
		K_T_dxhx, angle_T_dxhx, KoC_T_dxhx, k2oQ_rem_T_dxhx = [solve_KoC(Titan, option='', d0=d_max, h0=h_max, f_option=f_option1, prin=0)[i] for i in indices]

		# Color/fillstyle/marker for thick and thin ocean/ice shell thickness
		c_hx = tea # Color for thick ocean
		c_hn = lbl # Color for thin ocean
		fs_dx = 'full' # Fillstyle for thick ice shell
		fs_dn = 'none' # Fillstyle for thin ice shell
		m_dx = 'o' # Marker for thick ice shell
		m_dn = 'o' # Marker for thin ice shell
		'''# End of copy-paste part #'''

		# Dissipation trade-off
		KoCs_T = np.logspace(-14, -6.3, 500) # Titan curve limits not perfect, might be arcsin() errors
		k2oQs_rem_T_hn = k2oQ_remainder(Titan, KoCs_T, angle_T_dnhn, prin=0) # delta-epsilon ~ constant for a given h, so just use one
		k2oQs_rem_T_hx = k2oQ_remainder(Titan, KoCs_T, angle_T_dnhx, prin=0)

		plt.plot(KoCs_T, k2oQs_rem_T_hx, c=c_hx)#, label='h = %d km'%(h_max/1e3)) # Continous line for thick ocean
		plt.plot(KoCs_T, k2oQs_rem_T_hn, c=c_hn)#, label='h = %d km'%(h_min/1e3)) # Continous line for thin ocean

		txtt = plt.text(*(3e-9, 1.3e-4), r'Titan', c=c_hx)

		# Four examples
		plt.plot(KoC_T_dnhn, k2oQ_rem_T_dnhn, c=c_hn, marker=m_dn, fillstyle=fs_dn, linestyle='none')#, label='d = %d km'%(d_min/1e3)) # Thin ice
		plt.plot(KoC_T_dxhn, k2oQ_rem_T_dxhn, c=c_hn, marker=m_dx, fillstyle=fs_dx, linestyle='none')#, label='d = %d km'%(d_max/1e3)) # Thick ice
		plt.plot(KoC_T_dnhx, k2oQ_rem_T_dnhx, c=c_hx, marker=m_dn, fillstyle=fs_dn, linestyle='none')#, label='d = %d km'%(d_min/1e3)) # Thin ice
		plt.plot(KoC_T_dxhx, k2oQ_rem_T_dxhx, c=c_hx, marker=m_dx, fillstyle=fs_dx, linestyle='none')#, label='d = %d km'%(d_max/1e3)) # Thick ice

		# Print out K/C, k2/Q, angle for four pairs
		print('\nh %d [km], d %d [km], flattening %s'%(h_min/1e3, d_min/1e3, f_option2))
		print('K/C %.4g [1/s], k2/Q %.4g, Delta-epsilon %.4g [deg]'%(KoC_T_dnhn, k2oQ_rem_T_dnhn, angle_T_dnhn*rd))
		print('\nh %d [km], d %d [km], flattening %s'%(h_min/1e3, d_max/1e3, f_option2))
		print('K/C %.4g [1/s], k2/Q %.4g, Delta-epsilon %.4g [deg]'%(KoC_T_dxhn, k2oQ_rem_T_dxhn, angle_T_dxhn*rd))
		print('\nh %d [km], d %d [km], flattening %s'%(h_max/1e3, d_min/1e3, f_option1))
		print('K/C %.4g [1/s], k2/Q %.4g, Delta-epsilon %.4g [deg]'%(KoC_T_dnhx, k2oQ_rem_T_dnhx, angle_T_dnhx*rd))
		print('\nh %d [km], d %d [km], flattening %s'%(h_max/1e3, d_max/1e3, f_option1))
		print('K/C %.4g [1/s], k2/Q %.4g, Delta-epsilon %.4g [deg]\n'%(KoC_T_dxhx, k2oQ_rem_T_dxhx, angle_T_dxhx*rd))

	if if_Moon == 1 :
		print('Moon')

		# Solve for K/C, Delta-epsilon to show that solver works
		indices = [0, 3, 5, 6]
		K_M, angle_M, KoC_M, k2oQ_rem_M = [solve_KoC(Moon, option='', d0=0, h0=0, f_option='', prin=0)[i] for i in indices] # Most arguments don't apply to the Moon

		# Dissipation trade-off
		KoCs_M = np.logspace(-14.2, -12.2, 500) # Moon curve limits
		k2oQs_rem_M = k2oQ_remainder(Moon, KoCs_M, angle_M, 0) # Use Delta-epsilon from solver

		color = tan
		plt.plot(KoCs_M, k2oQs_rem_M, c=color, linestyle='-') # Continuous theoretical line
		plt.plot(KoC_M, k2oQ_rem_M, c=color, marker='s', linestyle='none')#, label='K solved') # K/C_solver solution

		# LLR
		parameters = Moon()
		k2oQ0_M, KoC0_M, inc_M, theta_M = [parameters[i] for i in [9, 10, 5, 6]] # Retrieve the LLR k2/Q and K/C, and i/theta
		k2Q_err = 1.5e-4
		KC_err = 0.39e-13
		plt.plot([KoC0_M, KoC0_M], k2oQ0_M + np.array([-1, 1])*k2Q_err, c='k', linestyle='-') # LLR k2/Q error bars
		plt.plot(KoC0_M + np.array([-1, 1])*KC_err, [k2oQ0_M, k2oQ0_M], c='k', linestyle='-') #, label='LLR') # LLR K/C error bars

		txtm = plt.text(*(2.2e-14, 1.3e-4), r'Moon', c=color)

		# Print out K/C, k2/Q, angle for LLR and the K/C solver
		print('LLR: K/C %.4g [1/s], k2/Q %.4g, Delta-epsilon %.4g [deg]'%(KoC0_M, k2oQ0_M, (theta_M-inc_M)*rd ))
		print('Solver: K/C %.4g [1/s], k2/Q %.4g, Delta-epsilon %.4g [deg]'%(KoC_M, k2oQ_rem_M, angle_M*rd))

	#plt.legend()
	plt.xlim([1e-14, 3e-7])
	plt.ylim([1e-4, 1.6e-01])
	plt.tick_params(axis='y', which='both',direction='in', right=True)
	plt.tick_params(axis='x', which='both',direction='in', top=True)
	plt.xlabel(r'$K/C_s$ (s$^{-1}$)')
	plt.ylabel(r'$k_2/Q$')
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()

	if save: plt.savefig('Fig2_Moon_Titan_k2oQ_KoC.svg', dpi=1200, format='svg')

	plt.show()

def graph_heat_damping(moon, save=0):
	''' Graph heating and damping timescales
	LEFT: tidal heating rate (left axis) and surface heat flux (right axis) as a fcn of k2/Q
	RIGHT: eccentricity/inclination damping timescales as a fcn of k2/Q
	BOTH: mark one k2/Q (Moon is observed k2/Q, Titan is upper bound)
	TITAN: mark 4 examples from graph_dissipation_partition()
	CHOOSE: moon = Moon or moon = Titan
	MANUSCRIPT: Makes Figure 3
	e.g., graph_heat_damping(Titan) '''
	Mp, M, R, a, e, inc, theta0, n, c, k2oQ0, KoC0, J2, C22, eta, f_cmb0, gamma, phi_CS, phi, name = moon()
	sa = 4 * np.pi * R**2 # [m^2] Surface area

	if moon == Titan :
		k2oQ = k2oQ_from_phi(Mp, M, R, a, n, inc, theta0, eta, c, phi) # [ ] Tides end-member
		K, KoC, angle = 0, 0, 0 # In case of no differential rotation between ice shell and ocean
		
		# Comparisons (ocean heating, radiogenic heating)
		cd_T, Edot_o_oc_T = calculate_Titan_ocean_tides() # [ ] Bottom drag coefficient, [W] ocean obliquity tidal heating
		Edot_rad_T = 0.003*sa # [W/m^2->W] Radiogenic surface flux 3 mW/m^2, ref: Kirk & Stevenson 1987; Mitri & Showman 2008

		''' FOUR EXAMPLES '''
		# Ice shell thickness d & ocean thickness h pairs chosen to bound max/min K/C & k2/Q
		d_min, d_max = 25e3, 200e3 # [m] Ice shell thickness, min (n) and max (x)
		h_min, h_max = 10e3, 400e3 # [m] Ocean thickness, min (n) and max (x)
		f_option1, f_option2 = 'low', 'high'

		# Solve for K/Cs and k2/Q_remainders for min/max d & h
		indices = [0, 3, 5, 6] # 0K_solved, 1kappa_solved, 2xi, 3delta_epsilon, 4FCN_period, 5KoC_solved, 6k2oQ_rem
		K_T_dnhn, angle_T_dnhn, KoC_T_dnhn, k2oQ_rem_T_dnhn = [solve_KoC(Titan, option='', d0=d_min, h0=h_min, f_option=f_option2, prin=0)[i] for i in indices]
		K_T_dxhn, angle_T_dxhn, KoC_T_dxhn, k2oQ_rem_T_dxhn = [solve_KoC(Titan, option='', d0=d_max, h0=h_min, f_option=f_option2, prin=0)[i] for i in indices]
		K_T_dnhx, angle_T_dnhx, KoC_T_dnhx, k2oQ_rem_T_dnhx = [solve_KoC(Titan, option='', d0=d_min, h0=h_max, f_option=f_option1, prin=0)[i] for i in indices]
		K_T_dxhx, angle_T_dxhx, KoC_T_dxhx, k2oQ_rem_T_dxhx = [solve_KoC(Titan, option='', d0=d_max, h0=h_max, f_option=f_option1, prin=0)[i] for i in indices]

		# Eccentricity heating [W], CMB + obliquity heating [W], eccentricity damping timescale [s], inclination damping timescale [s]
		Edot_e_dnhn, Edot_i_dnhn, Edot_tot, dadt, tau_e_dnhn, tau_i_dnhn = calculate_heat_damping(Titan, k2oQ_rem_T_dnhn, K_T_dnhn, angle_T_dnhn, prin=1)
		Edot_e_dxhn, Edot_i_dxhn, Edot_tot, dadt, tau_e_dxhn, tau_i_dxhn = calculate_heat_damping(Titan, k2oQ_rem_T_dxhn, K_T_dxhn, angle_T_dxhn, prin=1)
		Edot_e_dnhx, Edot_i_dnhx, Edot_tot, dadt, tau_e_dnhx, tau_i_dnhx = calculate_heat_damping(Titan, k2oQ_rem_T_dnhx, K_T_dnhx, angle_T_dnhx, prin=1)
		Edot_e_dxhx, Edot_i_dxhx, Edot_tot, dadt, tau_e_dxhx, tau_i_dxhx = calculate_heat_damping(Titan, k2oQ_rem_T_dxhx, K_T_dxhx, angle_T_dxhx, prin=1)
		
	elif moon == Moon :
		k2oQ = k2oQ0 
		KoC = KoC0
		K = KoC * c*M*R**2
		angle = np.abs(theta0-inc)


	''' GRAPHS '''
	def heat2flux(heat): return( heat*1e9/sa ) # for left and right y-axes
	def flux2heat(flux): return( flux*sa/1e9 )

	# Eccentricity and inclination damping timescales due to tidal heating only
	k2oQs = np.logspace(-4, -0.5, 100)
	Edot_es, Edot_os, Edot_tot, dadt, tau_es, tau_is = calculate_heat_damping(moon, k2oQs, 0, 0, prin=0)

	# For specific k2/Q defined above
	Edot_e, Edot_o_cmb, Edot_tot, dadt, tau_e, tau_i = calculate_heat_damping(moon, k2oQ, K, angle, prin=0)

	''' Left a) Graph tidal heating '''
	fig, axes = plt.subplots(1, 2, figsize = (7.25, 3.55*.75))

	if save : plt.rcParams.update({'font.size': 8})
	else : plt.rcParams.update({'font.size': 10})
	
	ax = axes[0]
	ax.plot(k2oQs, Edot_es/1e9, 'k', label='solid-body eccentricity')
	ax.plot(k2oQs, Edot_os/1e9, 'k--', label='solid-body obliquity')
	ax.scatter([k2oQ, k2oQ], [Edot_e/1e9, Edot_o_cmb/1e9], s=60, c='k', marker='X')

	if moon == Titan :
		ax.axhline(y=Edot_o_oc_T/1e9, c='k', linestyle='-.', label='ocean obliquity')
		ax.axhline(y=Edot_rad_T/1e9, c='k', linestyle=':', label='radiogenic') 

		''' FOUR EXAMPLES '''
		# Color/fillstyle/marker for thick and thin ocean/ice shell thickness
		c_hx = tea # Color for thick ocean
		c_hn = lbl # Color for thin ocean
		fs_dx = 'full' # Fillstyle for thick ice shell
		fs_dn = 'none' # Fillstyle for thin ice shell
		m_dx = 'o' # Marker for thick ice shell
		m_dn = 'o' # Marker for thin ice shell

		ax.plot([k2oQ_rem_T_dnhn, k2oQ_rem_T_dnhn], [Edot_i_dnhn/1e9, Edot_e_dnhn/1e9],
			c=c_hn, marker=m_dn, fillstyle=fs_dn, ls='none', label='d = %d km'%(d_min/1e3)) # Thin ice
		ax.plot([k2oQ_rem_T_dxhn, k2oQ_rem_T_dxhn], [Edot_i_dxhn/1e9, Edot_e_dxhn/1e9],
			c=c_hn, marker=m_dx, fillstyle=fs_dx, ls='none', label='d = %d km'%(d_max/1e3)) # Thick ice
		ax.plot([k2oQ_rem_T_dnhx, k2oQ_rem_T_dnhx], [Edot_i_dnhx/1e9, Edot_e_dnhx/1e9],
			c=c_hx, marker=m_dn, fillstyle=fs_dn, ls='none', label='d = %d km'%(d_min/1e3)) # Thin ice
		ax.plot([k2oQ_rem_T_dxhx, k2oQ_rem_T_dxhx], [Edot_i_dxhx/1e9, Edot_e_dxhx/1e9],
			c=c_hx, marker=m_dx, fillstyle=fs_dx, ls='none', label='d = %d km'%(d_max/1e3)) # Thick ice

	secax = ax.secondary_yaxis('right', functions=(heat2flux, flux2heat))
	secax.set_ylabel(r'heat flux (W/$m^{2}$)')
	ax.set_xlabel(r'$k_2/Q$')
	ax.set_ylabel('heating (GW)')
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.legend()

	''' Right b) Damping timescale '''
	ax = axes[1]
	ax.plot(k2oQs, tau_is/s_in_Myr, 'k--', label='inclination')
	ax.plot(k2oQs, tau_es/s_in_Myr, 'k', label='eccentricity')
	ax.scatter(k2oQ, tau_e/s_in_Myr, s=60, c='k', marker='X')
	ax.scatter(k2oQ, tau_i/s_in_Myr, s=60, c='k', marker='X')

	if moon == Titan :
		ax.plot([k2oQ_rem_T_dnhn, k2oQ_rem_T_dnhn], [tau_i_dnhn/s_in_Myr, tau_e_dnhn/s_in_Myr],
			c=c_hn, marker=m_dn, fillstyle=fs_dn, ls='none', label='d = %d km'%(d_min/1e3)) # Thin ice
		ax.plot([k2oQ_rem_T_dxhn, k2oQ_rem_T_dxhn], [tau_i_dxhn/s_in_Myr, tau_e_dxhn/s_in_Myr],
			c=c_hn, marker=m_dx, fillstyle=fs_dx, ls='none', label='d = %d km'%(d_max/1e3)) # Thick ice
		ax.plot([k2oQ_rem_T_dnhx, k2oQ_rem_T_dnhx], [tau_i_dnhx/s_in_Myr, tau_e_dnhx/s_in_Myr],
			c=c_hx, marker=m_dn, fillstyle=fs_dn, ls='none', label='d = %d km'%(d_min/1e3)) # Thin ice
		ax.plot([k2oQ_rem_T_dxhx, k2oQ_rem_T_dxhx], [tau_i_dxhx/s_in_Myr, tau_e_dxhx/s_in_Myr],
			c=c_hx, marker=m_dx, fillstyle=fs_dx, ls='none', label='d = %d km'%(d_max/1e3)) # Thick ice

	ax.tick_params(axis='y', which='both', direction='in', right=True)
	ax.tick_params(axis='x', which='both', direction='in', top=True)
	ax.set_xlabel(r'$k_2/Q$')
	ax.set_ylabel('damping timescale (Myr)')
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.legend()
	fig.tight_layout()

	if save: plt.savefig('Fig3_Titan_heating_timescales.svg', dpi=1200, format='svg')

	plt.show()

def graph_Titan_free_ocean_nutation(bounds, save=0):
	''' Titan's free ocean nutation period vs cmb flattening
		1) Ice shell-ocean boundary flattening from models (surface f from data)
		2) Print one Free Ocean Nutation example
		3) Free Ocean Nutation periods for different h/d/f_cmb
	BOUNDS: if bounds=1, print and graph FON period range for low/high f_cmb and low/high ocean thickness
	MANUSCRIPT: Figure S1 in the SM
	e.g., graph_Titan_free_ocean_nutation(1) '''
	Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, fc, gamma, phi_CS, phi, name = Titan()
	C = c * M * R**2 # [kg m^2] Total body polar MoI 
	rho_shell = 1000 # [kg/m^3] Density of ice shell
	rho_ocean = 1000 # [kg/m^3] Density of the subsurface ocean

	# Polar flattenings/oblateness, defined f(or alpha)~J2/c, beta~4C22/c '''
	f_s = 19.2236e-5 # [ ] Surface polar flattening, ref: Baland et al 2014, Eq. 5, from Zebker et al 2009
	f_cmb_low = -2e-4 # [ ] Lower bound on ocean polar flattening, ref: alpha_o Baland+14 Fig. 1 & Coyette+18 Fig. 1
	f_cmb_hig = -4.4e-3 # [ ] Upper bound on ocean polar flattening, ref: alpha_o Baland+14 Fig. 1 & Coyette+18 Fig. 1
	f_icb_low = 6e-5 # [ ] Lower bound on inner solid polar flattening, ref: alpha_m Coyette+18 Fig. 1
	f_icb_hig = 1.06e-4 # [ ] Upper bound on inner solid polar flattening, ref: alpha_m Coyette+18 Fig. 1

	''' Helper Functions '''
	def suite(d, h, f_cmb, f_icb, prin):
		output = moi_density_FCN_Titan(d, h, rho_shell, rho_ocean, f_s, f_cmb, f_icb)
		R_cmb, R_icb, rho_int, C_shell, C_ocean, C_int, FCN_freq = output
		FCN_period = 2*np.pi/FCN_freq # [s] FCN period, neg if retrograde, pos if prograde
		if prin :
			print('d %d km, h %d km, rho_interior %.1f kg/m^3'%(d/1e3, h/1e3, rho_int))
			print('f_cmb %.3g, f_icb %.3g'%(f_cmb, f_icb))
			print('MoI: C_shell/C %.3g'%(C_shell/C))
			print('FON aka FCN (pos prograde, neg retrograde): %.3g yrs\n'%(FCN_period/s_in_yr))
		return(FCN_period)

	''' FONs '''
	d_thin = 25e3 # [m] thin ice shell limit, ref: Baland+11, Table 2, 10km - 200km
	d_thic = 200e3 # [m] thick ice shell limit, ref: Baland+11, Table 2, 10km - 200km
	
	h_thin = 10e3 # [m] thin ocean limit, ref: Baland 2011 Table 2, 5km - 570km
	h_medi = 200e3 # [m] medium ocean
	h_thic = 400e3 # [m] thick ocean limit, ref: Baland 2011 Table 2, 5km - 570km

	d0 = 50e3 # [m] example ice shell thickness
	h0 = 200e3 # [m] example ocean thickness
	f_icb0 = f_icb_low # [ ] example ocean-interior boundary polar flattening

	# Print one case
	print('EXAMPLE')
	FON_period_test = suite(d=25e3, h=10e3, f_cmb=f_cmb_hig, f_icb=f_icb_hig, prin=1)

	# Vary flattening of cmb (ice shell-ocean boundary), negative b/c ice shell thickness variations
	f_cmbs = -np.logspace(-5, -2, 100)

	'''#
	# Constant h, different d, constant f_icb
	fons_thin = suite(d_thin, h0, f_cmbs, f_icb0, prin=0); label_1 = '%d km shell'%(d_thin/1e3) # Thin ice shell
	fons_thic = suite(d_thic, h0, f_cmbs, f_icb0, prin=0); label_2 = '%d km shell'%(d_thic/1e3) # Thick ice shell
	#'''

	'''#
	# Constant h, constant d, different f_icb
	fons_thin = suite(d0, h0, f_cmbs, f_icb_low, prin=0); label_1 = '%.0e icb flat'%(f_icb_low) # Lower bound on f_icb
	fons_thic = suite(d0, h0, f_cmbs, f_icb_hig, prin=0); label_2 = '%.0e icb flat'%(f_icb_hig) # Upper bound on f_icb
	#'''

	'''#'''
	# Different h, constant d, constant f_icb
	fons_thin = suite(d=d0, h=h_thin, f_cmb=f_cmbs, f_icb=f_icb_low, prin=0); label_1 = '%d km ocean'%(h_thin/1e3) # Thin ocean
	fons_thic = suite(d=d0, h=h_thic, f_cmb=f_cmbs, f_icb=f_icb_low, prin=0); label_2 = '%d km ocean'%(h_thic/1e3) # Medium ocean
	fons_medi = suite(d=d0, h=h_medi, f_cmb=f_cmbs, f_icb=f_icb_low, prin=0); label_3 = '%d km ocean'%(h_medi/1e3) # Thick ocean
	#'''
	
	# What f_cmbs are in literature and what are their FON periods? See if FON is faster or slower than ice shell
	f_cmb_B19 = 0.8*f_s # [ ] Ice shell-ocean boundary polar flattening, ref: Baland+19 Table 5, TM1 model
	f_icb_B19 = 0.6*f_s # [ ] Ocean-interior boundary polar flattening, ref: Baland+19 Table 5, TM1 model
	FON_period_B19 = -324*s_in_yr # [s] FON ref: Baland+19 Table 6, TM1 model (assumed retrograde FON mode if f_cmb_B19 is positive)
	
	''' GRAPH '''
	plt.figure(figsize = (3.55, 3.55*.8))
	if save : plt.rcParams.update({'font.size': 8})
	else : plt.rcParams.update({'font.size': 10})
	plt.grid()

	# Precession period of ice shell/orbit (ocean period is compared to ice shell period)
	plt.axhline(y=2*np.pi/eta/s_in_yr, c='k', label='orbit precession')

	# Precession period of core/ocean+below for wide range of f_cmb
	plt.plot(np.abs(f_cmbs), fons_thic/s_in_yr, ':', c='k', label=label_2)
	plt.plot(np.abs(f_cmbs), fons_medi/s_in_yr, '-.', c='k', label=label_3)
	plt.plot(np.abs(f_cmbs), fons_thin/s_in_yr, '--', c='k', label=label_1)

	# Baland et al. 2019
	plt.plot(f_cmb_B19, FON_period_B19/s_in_yr, linestyle='None', marker='^', c='k', label='B19')

	# Shade the areas consistent with the literature
	inds = np.where((f_cmbs < f_cmb_low) & (f_cmbs > f_cmb_hig)) # flipped < and > because negative
	plt.fill_between(np.abs(f_cmbs[inds]), -1e3, 1e3, color='0.8', label='B14 & C18')
	
	# Baland et al. 2014 & Coyette et al. 2018
	if bounds :
		print('BOUNDS ON FON PERIOD from B+14 and C+18 f_cmb and h thickness')
		fcn_thin_low = suite(d0, h_thin, f_cmb_low, f_icb0, prin=1)
		fcn_thin_hig = suite(d0, h_thin, f_cmb_hig, f_icb0, prin=1)
		fcn_thic_low = suite(d0, h_thic, f_cmb_low, f_icb0, prin=1)
		fcn_thic_hig = suite(d0, h_thic, f_cmb_hig, f_icb0, prin=1)
		plt.plot(np.abs([f_cmb_low, f_cmb_hig]), [fcn_thin_low/s_in_yr, fcn_thin_hig/s_in_yr], marker='o', ls='none', c=pin)
		plt.plot(np.abs([f_cmb_low, f_cmb_hig]), [fcn_thic_low/s_in_yr, fcn_thic_hig/s_in_yr], marker='o', ls='none', c=pin)
		plt.legend(loc='lower right')

	plt.xlabel(r'$|f_o|$')
	plt.ylabel(r'free ocean nutation period (yrs)')
	plt.xscale('log')
	plt.yscale('symlog')
	plt.ylim([-1e3, 1e3])
	#plt.title(name)
	plt.tight_layout()

	if save: plt.savefig(name+'_FON_period.svg', dpi=1200, format='svg')

	plt.show()

def graph_Jovian_spin(moon, graph=1, k2oQ_test=0, precision=0, legend=1):
	''' Spin angles for Jovian satellites; graph angles vs k2/Q
	CHOOSE: moon = Io/Europa/Ganymede/Callisto
	HOW TO USE:
		a) Make new figure -> graph=1
		b) Subplots -> graph = <subplot axis> (called from satellite_subplots())
		- k2oQ is input variable if you want
		- precision (in deg) is if you want to compare to a space mission accuracy/uncertainty
	MANUSCRIPT: Io/Europa/Ganymede angles appear in Table 1, info in "Application to Io, Europa, Ganymede, and Callisto"
	e.g., graph_Jovian_offset(Io, 1, 1e-3) '''	

	Mp, M, R, a, e, inc, theta, n, c, k2oQ0, KoC, J2, C22, eta, fc, gamma, phi_CS, phi, name = moon()
	de = 0 # Assume no differential rotation at core-mantle boundary
	dw = n*np.sin(de) # Just a formality

	# PHASE 1, print one set of angles from one k2oQ
	theta_pred, phi_pred, gamma_pred, sx_p, sy_p, sz_p = calculate_spin_from_dissipation(moon, k2oQ_test, KoC, de, prin=0)
	phi_pred = phi_pred - phi_CS
	Edot, hf = heat_flux(Mp, R, n, a, e, theta_pred, k2oQ_test, KoC, dw, prin=0) # [W/m^2] heat flux from chosen k2/Q

	print('%s'%name)
	print('k2/Q %.4g'%k2oQ_test)
	print('Obliquity %.4g deg'%(theta_pred*rd))
	print('Azimuth from Cassini plane %.4g deg'%(phi_pred*rd))
	print('Cassini plane offset %.4g deg'%(gamma_pred*rd))
	print('Heat flux %.4g W/m^2 \n'%hf)

	# PHASE 2, make graph of all spin angles, varying k2oQ, no KoC
	k2oQs = np.logspace(-4, -0.5, 100)
	output = np.array([ calculate_spin_from_dissipation(moon, k2oQ_iter, KoC, de, prin=0) for k2oQ_iter in k2oQs ])
	thetas, phis, gammas = output[:, 0], output[:, 1]-phi_CS, output[:, 2]
	
	# GRAPHING
	if graph == 1 :
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.set_ylabel('spin axis angles (deg)')
	else :
		ax = graph

	one, two, three, four = 'paleturquoise', 'darkturquoise', 'darkcyan', 'darkslategrey'

	ax.axhline(y=inc*rd, color=one, linestyle=':', label=r'inclination $i$')
	ax.plot(k2oQs, np.abs(thetas)*rd, c=two, linestyle='--', label=r'obliquity $\theta$')
	ax.plot(k2oQs, np.abs(phis)*rd, c=three, linestyle='-.', label=r'azimuth $\phi-\phi_{cs}$')
	ax.plot(k2oQs, np.abs(gammas)*rd, c=four, linestyle='-', label=r'offset $\gamma$')

	if precision > 0 :
		ax.axhline(y=precision, color='m', label='accuracy')
		ax.scatter(k2oQ_test, np.abs(theta_pred)*rd, s=60, color=pin, marker='X')
		ax.scatter(k2oQ_test, np.abs(phi_pred)*rd, s=60, color=pin, marker='X')
		ax.scatter(k2oQ_test, np.abs(gamma_pred)*rd, s=60, color=pin, marker='X')
		
	ax.set_xlabel(r'$k_2/Q$'); 
	ax.set_xlim([1e-4, 2e-1])
	ax.set_ylim([3e-6, 1])
	ax.set_xscale('log'); ax.set_yscale('log')
	ax.set_title(name)
	
	if legend == 1: ax.legend()
	if graph == 1: plt.show()

def subplots_Jovian_spin():
	''' See graph_Jovian_spin() description
	4 subplots
		- Io, Europa, Ganymede, Callisto
		- Inclination, obliquity, azimuth, gamma
	Pick test k2/Q for each body
	Pick test precision/uncertainty for min detectable gamma
	MANUSCRIPT: Table 1, Section on "Application to I, E, G, C"
	e.g., subplots_Jovian_offset() '''
	fig, ax = plt.subplots(2,2, sharey=True, figsize=(8,7))

	k2oQ_i = 0.0109 # [ ] Io's k2/Q, ref: Park et al. 2024, astrometry 0.015 -> Lainey et al. 2009
	k2oQ_e = 2e-2 # [ ] Europa's test k2/Q, min detectable by Europa Clipper
	k2oQ_g = 1.6e-3 # [ ] Ganymede's test k2/Q, min detectable by JUICE, 1e-6 rad
	k2oQ_c = 1e-3 # [ ] Callisto's test k2/Q, close to the Moon's value

	# Spacecraft mission spin state precision/uncertainty
	prec_i = 1e-5/rd # (No planned mission to get Io's spin state)
	prec_e = 0.05/60 # [deg] 0.05 arcmin desired for obl to get 0.004 precision in MOI, Europa Clipper Mazarico et al. 2023
	prec_g = 1e-6*rd # [deg] 1 micro radian, JUICE Cappuccio et al. 2020
	prec_c = 5.5e-3*rd # [deg] 5.5 mrad for obliquity, JUICE Cappuccio et al. 2022

	graph_Jovian_spin(Io, ax[0,0], k2oQ_i, prec_i, legend=1)
	graph_Jovian_spin(Europa, ax[0,1], k2oQ_e, prec_e, legend=0)
	graph_Jovian_spin(Ganymede, ax[1,0], k2oQ_g, prec_g, legend=0)
	graph_Jovian_spin(Callisto, ax[1,1], k2oQ_c, prec_c, legend=0)
	ax[0,0].set_ylabel('spin axis angles (deg)')
	ax[1,0].set_ylabel('spin axis angles (deg)')

	plt.tight_layout()
	plt.show()

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
#graph_dissipation_partition(if_Titan = 1, if_Moon = 1, save=0)

''' Figure 3: a) tidal heating & heat flux b) eccentricity & inclination damping timescales as fcn of k2/Q '''
#graph_heat_damping(moon=Titan, save=0)

''' Figure S1: Titan's free ocean nutation period vs flattening at ocean-ice shell boundary '''
#graph_Titan_free_ocean_nutation(bounds=1, save=0) # legend and marked points
#graph_Titan_free_ocean_nutation(bounds=0, save=1) # no legend or marked points

''' Table 1: rotation states of Moon/Titan/Io/Ganymede '''
#end_k2oQ_M, end_KoC_M = calculate_dissipation_from_spin(Moon)
#graph_Jovian_spin(moon=Io, graph=1, k2oQ_test=0.015, precision=0, legend=1)
#graph_Jovian_spin(moon=Europa, graph=1, k2oQ_test=0.02, precision=0.05/60, legend=1)
#graph_Jovian_spin(moon=Ganymede, graph=1, k2oQ_test=0.0016, precision=1e-6*rd, legend=1)

''' Table 2: physical and orbital parameters for Moon/Titan/Io/Ganymede '''
#Mp, M, R, a, e, inc, theta, n, c, k2oQ, KoC, J2, C22, eta, f_c, gamma, phi_CS, phi, name = Moon()
#print('Indices: 0Mp, 1M, 2R, 3a, 4e, 5inc, 6theta, 7n, 8c, 9k2oQ, 10KoC, 11J2, 12C22, 13eta, 14fc, 15gamma, 16name')
#Titan()
#Io()
#Ganymede()
## Example
#eccentricity_io = Io()[4]

''' Other: print observed angles and predicted angles, dissipative parameters, heating rates, and damping timescales '''
#end_k2oQ_M, end_KoC_M = calculate_dissipation_from_spin(Moon)
#end_k2oQ_T, end_KoC_T = calculate_dissipation_from_spin(Titan)

''' Other: check that dissipation parameters produce angles of Moon and Titan '''
#output_M = calculate_spin_from_dissipation(Moon)
#output_T = calculate_spin_from_dissipation(Titan)

''' Other: solve for K/C given physical parameters of Moon and Titan and print important information'''
#output_M = solve_KoC(Moon)
#output_T = solve_KoC(Titan, '', d0=100e3, h0=100e3, f_option='high', prin=1)

''' Other: print Titan's ocean bottom drag coefficient and obliquity tidal heating  '''
#cD, Edoto = calculate_Titan_ocean_tides(); print('Bottom drag coefficient %.4g, ocean obliquity tidal heating %.4g W'%(cD, Edoto))

''' Other: predict spin states of Jovian satellites '''
#subplots_Jovian_spin()

