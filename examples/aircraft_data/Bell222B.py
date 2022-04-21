# DATA FOR Bell 222 B
# Collected from various sources
# Helicopter Performance, Stability, and Control by Prouty R.W. P691
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

aero['polar'] = polar
ac['aero'] = aero

# ==GEOMETRY==============================
geom = dict()

wing = dict()
wing['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
wing['S_ref']           = {'value': 0.0001, 'units': 'm**2'}
wing['AR']              = {'value': 0.0001}
wing['c4sweep']         = {'value': 1.0, 'units': 'deg'}
wing['taper']           = {'value': 1.0} # assumed rectangle wing
wing['toverc']          = {'value': 0.19} # assumed, small turbojet KingAirC90GT
geom['wing'] = wing

body = dict()
body['S_ref'] = {'value': 467.929, 'units': 'ft**2'} # Default value
geom['body'] = body

fuselage = dict()
fuselage['length']      = {'value': 11.14, 'units': 'm'}
fuselage['width']       = {'value': 3.12, 'units': 'm'} 
fuselage['height']      = {'value': 3.43, 'units': 'm'} 
geom['fuselage'] = fuselage
## Thought: 

hstab = dict()
hstab['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
hstab['S_ref']          = {'value': 1.10, 'units': 'm**2'} # set to zero is still needed for body['S_ref] calculation
hstab['c4_to_wing_c4']  = {'value': 0.0001, 'units': 'm'}
geom['hstab'] = hstab

vstab = dict()
vstab['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
vstab['S_ref']          = {'value': 0.9, 'units': 'm**2'}
geom['vstab'] = vstab

nosegear = dict()
nosegear['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
nosegear['length'] = {'value': 0.3, 'units': 'm'} # not related to weight_cal
geom['nosegear'] = nosegear

maingear = dict()
maingear['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
maingear['length'] = {'value': 0.3, 'units': 'm'} # not related to weight_cal
geom['maingear'] = maingear

ac['geom'] = geom

# ==WEIGHTS========================
weights = dict()
weights['MTOW']         = {'value': 8250, 'units': 'lb'}
weights['W_battery']    = {'value': 1275, 'units': 'lb'} # Electrified 
weights['Payload']      = {'value': 2046, 'units': 'lb'}
ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()
# equals to rotor in rotorcraft
propeller = dict()
propeller['diameter']     = {'value': 12.8016, 'units': 'm'}
propeller['num_rotors']   = {'value': 1}
propeller['proprpm']      = {'value': 347.867, 'units': 'rpm'} # V_tip = 765 (ft/s)
propeller['coaxialprop'] = {'value': 0} # 0 for non-coaxial, 1 for coaxial layout -> Calculate body_S_ref & Drag
propeller['FM']           = {'value': 0.8, 'units': None} #Figure of Merit
propeller['num_blades']   = {'value': 2, 'units': None} # Number of blades
propeller['blade_chord']   = {'value': 2.165, 'units': 'ft'} # # Blade chord length, only for rotor calculation
propeller['propeller_type']   = {'value': 0} # Propeller or rotor, 0 for rotor, 1 for propeller
propulsion['propeller'] = propeller

motor = dict()
motor['rating'] = {'value': 680, 'units': 'kW'}   # max power of motor
motor['num_motors'] = ac['num_motors'] = {'value': 1}
propulsion['motor'] = motor


nacelle = dict()
nacelle['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
propulsion['nacelle'] = nacelle

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 4}
ac['payload'] = {'value': 2046, 'units':'lb'}

data['ac'] = ac