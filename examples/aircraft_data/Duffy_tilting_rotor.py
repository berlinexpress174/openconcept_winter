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
aero['CD0_body'] = {'value' : 0.3} 

polar = dict()
polar['e']              = {'value' : 0.78}
polar['CD0_TO']         = {'value' : 0.03}
polar['CD0_cruise']     = {'value' : 0.0205}
polar['CD0'] = {'value' : 0.06}   # reference area = wing
## Thought: Different CD_TO from vertical takeoff and cruise

aero['polar'] = polar
ac['aero'] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()

wing['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
wing['S_ref']           = {'value': 257.2, 'units': 'ft**2'}
wing['AR']              = {'value': 9.72} 
wing['c4sweep']         = {'value': 0.001, 'units': 'deg'} 
wing['taper']           = {'value': 1.0} # Rectabgular
wing['toverc']          = {'value': 0.16} 
geom['wing'] = wing
## Thought: May have no wings

body = dict()
body['S_ref'] = {'value': 467.93, 'units': 'ft**2'} # default 467.93 ft**2
geom['body'] = body

fuselage = dict()
#fuselage['S_wet']       = {'value': 780.154, 'units': 'ft**2'} # 10**0.8635 * MTOW**0.5632 | MTOW = 4000 lb  
fuselage['width']       = {'value': 5.8, 'units': 'ft'} 
fuselage['length']      = {'value': 50, 'units': 'ft'} 
fuselage['height']      = {'value': 5.8, 'units': 'ft'} 
geom['fuselage'] = fuselage
## Thought: 

hstab = dict()
hstab['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
hstab['S_ref']          = {'value': 40, 'units': 'ft**2'}
hstab['c4_to_wing_c4']  = {'value': 17.9, 'units': 'ft'}
geom['hstab'] = hstab
## Thought: May have no hstab

vstab = dict()
vstab['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
vstab['S_ref']          = {'value': 25, 'units': 'ft**2'}
geom['vstab'] = vstab
## Thought: May have no vstab or two vstab

nosegear = dict()
nosegear['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
nosegear['length'] = {'value': 3, 'units': 'ft'}
geom['nosegear'] = nosegear
## Thought: May have no vstab

maingear = dict()
maingear['equip']          = {'value': 1} # 1: eqiuped, 0 no eqiuped
maingear['length'] = {'value': 4, 'units': 'ft'}
geom['maingear'] = maingear
## Thought: May have no vstab

ac['geom'] = geom

# ==WEIGHTS========================
weights = dict()
weights['MTOW']         = {'value': 4000, 'units': 'lb'}
weights['W_battery'] = {'value': 462, 'units': 'lb'}
ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
#engine['rating']        = {'value': 100 * 8, 'units': '1000*W'}
#engine['rating']        = {'value': 850, 'units': 'hp'}
propulsion['engine']    = engine

propeller = dict()
propeller['diameter']     = {'value': 6.096, 'units': 'm'}
propeller['num_rotors']   = {'value': 2}
propeller['proprpm']      = {'value': 2000, 'units': 'rpm'}
propeller['coaxialprop'] = {'value': 0} # 0 for non-coaxial, 1 for coaxial layout -> Calculate body_S_ref & Drag.
propeller['FM']           = {'value': 0.8, 'units': None} #Figure of Merit
propeller['propeller_type']   = {'value': 0} # Propeller or rotor, 0 for rotor, 1 for propeller
propulsion['propeller'] = propeller

motor = dict()
motor['rating'] = {'value': 319.942, 'units': 'kW'}   # max power of motor
#motor['P_TO'] = {'value': 850, 'units': 'hp'}
motor['num_motors'] = ac['num_motors'] = {'value': 2}
propulsion['motor'] = motor

nacelle = dict()
nacelle['equip']          = {'value': 0} # 1: eqiuped, 0 no eqiuped
propulsion['nacelle'] = nacelle

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 4}
ac['payload'] = {'value': 960, 'units':'lb'}
ac['q_cruise'] = {'value': 135.4, 'units': 'lb*ft**-2'} # how to obtain this?

ac['eVTOL_type_tiltrotor'] = {'value': 1}
ac['eVTOL_type_multirotor'] = {'value': 0}

data['ac'] = ac