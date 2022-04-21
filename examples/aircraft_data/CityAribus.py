# DATA FOR eVTOL_test01
# Collected from various sources
# including SOCATA pilot manual
from __future__ import division
import numpy as np

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero['CD0_body'] = {'value' : 0.7} # cylinder shape  

polar = dict()
polar['e']              = {'value' : 0.78} # estimated 
polar['CD0_TO']         = {'value' : 0.03} # estimated 
polar['CD0_cruise']     = {'value' : 0.0205}
polar['CD0'] = {'value' : 0.06}   # reference area = wing
## Thought: Different CD_TO from vertical takeoff and cruise

aero['polar'] = polar
ac['aero'] = aero

# ==GEOMETRY==============================
geom = dict()

wing = dict()
wing['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
wing['S_ref']           = {'value': 0.0001, 'units': 'm**2'}
wing['AR']              = {'value': 0.0001}
geom['wing'] = wing

body = dict()
body['S_ref'] = {'value': 467.929, 'units': 'ft**2'} # Default value
geom['body'] = body

fuselage = dict()
fuselage['length']      = {'value': 8, 'units': 'm'}
fuselage['width']       = {'value': 1.8, 'units': 'm'} 
fuselage['height']      = {'value': 1.8, 'units': 'm'} 
geom['fuselage'] = fuselage
## Thought: 

hstab = dict()
hstab['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
hstab['S_ref']          = {'value': 0.0001, 'units': 'ft**2'} # set to zero is still needed for body['S_ref] calculation
hstab['c4_to_wing_c4']  = {'value': 0.0001, 'units': 'ft'}
geom['hstab'] = hstab

vstab = dict()
vstab['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
vstab['S_ref']          = {'value': 0.0001, 'units': 'ft**2'}
geom['vstab'] = vstab

nosegear = dict()
nosegear['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
nosegear['length'] = {'value': 0.0001, 'units': 'ft'} # not related to weight_cal
geom['nosegear'] = nosegear

maingear = dict()
maingear['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
maingear['length'] = {'value': 0.0001, 'units': 'ft'} # not related to weight_cal
geom['maingear'] = maingear

ac['geom'] = geom

# ==WEIGHTS========================
weights = dict()
weights['MTOW']         = {'value': 4850, 'units': 'lb'}
weights['W_battery']    = {'value': 800, 'units': 'lb'}
weights['Payload']      = {'value': 960, 'units': 'lb'}
ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()

propeller = dict()
propeller['diameter']     = {'value': 2.80416, 'units': 'm'}
propeller['num_rotors']   = {'value': 8}
propeller['proprpm']      = {'value': 2000, 'units': 'rpm'} # guess
propeller['coaxialprop'] = {'value': 1} # 0 for non-coaxial, 1 for coaxial layout -> Calculate body_S_ref & Drag
propeller['FM']           = {'value': 0.8, 'units': None} #Figure of Merit
propeller['num_blades']   = {'value': 3, 'units': None} # Number of blades, only for rotor calculation
propeller['blade_chord']   = {'value': 2.165, 'units': 'ft'} # # Blade chord length, only for rotor calculation
propeller['propeller_type']   = {'value': 1} # Propeller or rotor, 0 for rotor, 1 for propeller
propulsion['propeller'] = propeller

motor = dict()
motor['rating'] = {'value': 200, 'units': 'kW'}   # max power of motor
motor['num_motors'] = ac['num_motors'] = {'value': 8}
propulsion['motor'] = motor


nacelle = dict()
nacelle['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
propulsion['nacelle'] = nacelle

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 4}
ac['payload'] = {'value': 960, 'units':'lb'}
ac['q_cruise'] = {'value': 135.4, 'units': 'lb*ft**-2'} # how to obtain this?

data['ac'] = ac