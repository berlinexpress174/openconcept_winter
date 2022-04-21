# DATA FOR eVTOL_test01
# Collected from various sources
# including SOCATA pilot manual
from __future__ import division
import numpy as np

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero['CLmax_TO']   = {'value' : 2.5}
#aero['CLmax_TO']   = {'value' : 2}
aero['CD0_body'] = {'value' : 0.7} # cylinder shape  

polar = dict()
polar['e']              = {'value' : 0.78} # estimated 
polar['CD0_TO']         = {'value' : 0.03} # estimated 
polar['CD0_cruise']     = {'value' : 0.0205}
polar['CD0'] = {'value' : 0.06}   # reference area = wing

aero['polar'] = polar
ac['aero'] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()

wing['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
wing['S_ref']           = {'value': 12.0, 'units': 'm**2'} # span 12.2m = 40 ft
wing['AR']              = {'value': 10.167} # Assumed chord = 1.2
wing['c4sweep']         = {'value': 1.0, 'units': 'deg'}
wing['taper']           = {'value': 1.0} # assumed rectangle wing
wing['toverc']          = {'value': 0.19} # assumed, small turbojet KingAirC90GT
geom['wing'] = wing

body = dict()
#body['S_ref'] = {'value': np.pi * 0.3**2, 'units': 'm**2'}   # body reference area
body['S_ref'] = {'value': 467.929, 'units': 'ft**2'} # lenght(12) * width(5.8) + hstab_S_ref(47.5) + Wing_S_ref(193.75) + rotor area(dia:5 ft) * num_rotor(8)
geom['body'] = body

fuselage = dict()
#fuselage['S_wet']       = {'value': 780.154, 'units': 'ft**2'} # 10**0.8635 * MTOW**0.5632 | MTOW = 4000 lb  
fuselage['width']       = {'value': 5, 'units': 'ft'} # estimated
fuselage['length']      = {'value': 42, 'units': 'ft'} # given data
fuselage['height']      = {'value': 6, 'units': 'ft'} # estimated
geom['fuselage'] = fuselage

hstab = dict() 
hstab['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
hstab['S_ref']          = {'value': 0.001, 'units': 'm**2'} # 1.2 * 5m
hstab['c4_to_wing_c4']  = {'value': 0.001, 'units': 'ft'}
geom['hstab'] = hstab

vstab = dict() # V-tail
vstab['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
vstab['S_ref']          = {'value': 34.4445, 'units': 'ft**2'} # estimated 2m * 0.8m * 2 = 3.2 m**2
geom['vstab'] = vstab

nosegear = dict()
nosegear['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
nosegear['length'] = {'value': 2.0, 'units': 'ft'} # estimated
geom['nosegear'] = nosegear

maingear = dict()
maingear['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
maingear['length'] = {'value': 3.2, 'units': 'ft'} # estimated
geom['maingear'] = maingear

ac['geom'] = geom

# ==WEIGHTS========================
weights = dict()
weights['MTOW']         = {'value': 4000, 'units': 'lb'}
weights['W_battery'] = {'value': 400, 'units': 'lb'} # estimated
ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
#engine['rating']        = {'value': 100 * 8, 'units': '1000*W'}
#engine['rating']        = {'value': 850, 'units': 'hp'}
propulsion['engine']    = engine

propeller = dict()
propeller['diameter']     = {'value': 1.2, 'units': 'm'}
propeller['num_rotors']   = {'value': 12}
propeller['proprpm']      = {'value': 4500, 'units': 'rpm'}
propeller['coaxialproplayout'] = {'value': 1} 
propulsion['propeller'] = propeller


motor = dict()
motor['rating'] = {'value': 150, 'units': 'kW'}  # estimated
#motor['P_TO'] = {'value': 850, 'units': 'hp'}
motor['num_motors'] = ac['num_motors'] = {'value': 12}
propulsion['motor'] = motor

nacelle = dict()
nacelle['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
propulsion['nacelle'] = nacelle

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 2}
ac['payload'] = {'value': 480, 'units':'lb'}
ac['q_cruise'] = {'value': 135.4, 'units': 'lb*ft**-2'} # we dont need this 

data['ac'] = ac