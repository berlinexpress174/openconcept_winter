# DATA FOR eVTOL_test01
# Collected from various sources
# including SOCATA pilot manual
from __future__ import division
import numpy as np

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero['CLmax_TO']   = {'value' : 1.7}
#aero['CLmax_TO']   = {'value' : 2}
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
wing['S_ref']           = {'value': 0.0001, 'units': 'm**2'} # 193.75 ft^2
wing['AR']              = {'value': 0.0001}
wing['c4sweep']         = {'value': 0.0001, 'units': 'deg'}
wing['taper']           = {'value': 0.0001}
wing['toverc']          = {'value': 0.0001}
geom['wing'] = wing
## Thought: May have no wings

body = dict()
#body['S_ref'] = {'value': np.pi * 0.3**2, 'units': 'm**2'}   # body reference area
body['S_ref'] = {'value': 467.929, 'units': 'ft**2'} # lenght(12) * width(5.8) + hstab_S_ref(47.5) + Wing_S_ref(193.75) + rotor area(dia:5 ft) * num_rotor(8)
geom['body'] = body

fuselage = dict()
#fuselage['S_wet']       = {'value': 780.154, 'units': 'ft**2'} # 10**0.8635 * MTOW**0.5632 | MTOW = 4000 lb  
fuselage['length']      = {'value': 2.074, 'units': 'm'} 
fuselage['width']       = {'value': 1.018, 'units': 'm'} 
fuselage['height']      = {'value': 1.44, 'units': 'm'} 
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
weights['MTOW']         = {'value': 795.0, 'units': 'lb'}
weights['W_battery']    = {'value': 92, 'units': 'kg'}
weights['Payload']      = {'value': 220, 'units': 'lb'}
ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()

propeller = dict()
propeller['diameter']     = {'value': 1.52093, 'units': 'm'}
propeller['num_rotors']   = {'value': 8}
propeller['proprpm']      = {'value': 2000, 'units': 'rpm'} # guess
propeller['coaxialprop'] = {'value': 1} # 0 for non-coaxial, 1 for coaxial layout -> Calculate body_S_ref & Drag
propeller['FM']           = {'value': 0.8, 'units': None} #Figure of Merit
propeller['propeller_type']   = {'value': 0} # Propeller or rotor, 0 for propeller, 1 for rotor
propulsion['propeller'] = propeller

motor = dict()
motor['rating'] = {'value': 200, 'units': 'kW'}   # max power of motor
motor['num_motors'] = ac['num_motors'] = {'value': 8}
propulsion['motor'] = motor


nacelle = dict()
nacelle['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
propulsion['nacelle'] = nacelle

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 1}
ac['q_cruise'] = {'value': 135.4, 'units': 'lb*ft**-2'} # how to obtain this?

data['ac'] = ac