from __future__ import division
import sys
import os
from unittest import skip
from matplotlib.pyplot import connect
import numpy as np
sys.path.insert(0, os.getcwd())
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
import math
from openconcept.utilities.Comp_weight_aux import SeaLevelTakeoffPower, SeaLevelTakeoffPower2, CompBody_S_wet, ComputeDiskLoad, HoverPower, RotorInducedVelocity, AddVerticalPower
from openconcept.utilities.Comp_weight_aux import VerticalPower,VelocityRatio
from openconcept.utilities.eVTOL_Aero_model import BodyDrag
from examples.aircraft_data.JobyS4 import data as acdata
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.dvlabel import DVLabel

##TODO: add fuel system weight back in (depends on Wf, which depends on MTOW and We, and We depends on fuel system weight)

class WingWeight_eVTOL(ExplicitComponent):
    """
    Inputs: MTOW, ac|geom|wing|S_ref, ac|geom|wing|AR
    Outputs: W_wing
    Metadata: n_ult (ult load factor)

    # MTOW <-- interchangable --> Take-off weight (Raymer's book)
    # ac|geom|wing|S_ref <-- interchangable --> S (Raymer's book)
    """
    def initialize(self):
        #self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        #define configuration parameters
        #self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')
        #Cessna methods:
        self.options.declare('n_ult', default=3, desc='Ultimate load factor (dimensionless)')

    def setup(self):
        #nn = self.options['num_nodes']
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='Reference wing area in sq ft')
        self.add_input('ac|geom|wing|AR', desc='Wing aspect ratio')
        self.add_input('ac|geom|wing|equip', desc='equip wing or not')

        #self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('W_wing', units='lb', desc='Wing weight')

        self.declare_partials(['W_wing'], ['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        #USAF method, Roskam PVC5pg68eq5.4
        #W_wing_USAF = 96.948*((inputs['ac|weights|MTOW']*n_ult/1e5)**0.65 * (inputs['ac|geom|wing|AR']/math.cos(inputs['ac|geom|wing|c4sweep']))**0.57 * (inputs['ac|geom|wing|S_ref']/100)**0.61 * ((1+inputs['ac|geom|wing|taper'])/2/inputs['ac|geom|wing|toverc'])**0.36 * (1+inputs['V_H']/500)**0.5)**0.993
        #Torenbeek, Roskam PVC5p68eq5.5
        #b = math.sqrt(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])
        #root_chord = 2*inputs['ac|geom|wing|S_ref']/b/(1+inputs['ac|geom|wing|taper'])
        #tr = root_chord * inputs['ac|geom|wing|toverc']
        #c2sweep_wing = inputs['ac|geom|wing|c4sweep'] # a hack for now
        #W_wing_Torenbeek = 0.00125*inputs['ac|weights|MTOW'] * (b/math.cos(c2sweep_wing))**0.75 * (1+ (6.3*math.cos(c2sweep_wing)/b)**0.5) * n_ult**0.55 * (b*inputs['ac|geom|wing|S_ref']/tr/inputs['ac|weights|MTOW']/math.cos(c2sweep_wing))**0.30

        #Cessna method, Roskam PVC5pg67eq5.2
        W_wing_Cessna = (0.0467 * inputs['ac|weights|MTOW']**0.347 * inputs['ac|geom|wing|S_ref']**0.36 * n_ult**0.397 * inputs['ac|geom|wing|AR']**1.712) * inputs['ac|geom|wing|equip']

        outputs['W_wing'] = W_wing_Cessna

    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']
        J['W_wing','ac|weights|MTOW'] = 0.347*0.0467 * inputs['ac|weights|MTOW']**(0.347-1) * inputs['ac|geom|wing|S_ref']**0.36 * n_ult**0.397 * inputs['ac|geom|wing|AR']**1.712 * inputs['ac|geom|wing|equip']
        J['W_wing','ac|geom|wing|S_ref'] = 0.36*0.0467 * inputs['ac|weights|MTOW']**0.347 * inputs['ac|geom|wing|S_ref']**(0.36-1) * n_ult**0.397 * inputs['ac|geom|wing|AR']**1.712 * inputs['ac|geom|wing|equip']
        J['W_wing','ac|geom|wing|AR'] = 1.712*0.0467 * inputs['ac|weights|MTOW']**0.347 * inputs['ac|geom|wing|S_ref']**0.36 * n_ult**0.397 * inputs['ac|geom|wing|AR']**(1.712-1) * inputs['ac|geom|wing|equip']
        J['W_wing','ac|geom|wing|equip'] = (0.0467 * inputs['ac|weights|MTOW']**0.347 * inputs['ac|geom|wing|S_ref']**0.36 * n_ult**0.397 * inputs['ac|geom|wing|AR']**1.712)
        


class EmpennageWeight_eVTOL(ExplicitComponent):
    """
    Inputs: ac|geom|hstab|S_ref, ac|geom|vstab|S_ref
    Outputs: Wemp
    Metadata: n_ult (ult load factor)

    """
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')


    def setup(self):
        self.add_input('ac|geom|hstab|S_ref', units='ft**2', desc='Projected horiz stab area in sq ft')
        self.add_input('ac|geom|vstab|S_ref', units='ft**2', desc='Projected vert stab area in sq ft')
        self.add_input('ac|geom|hstab|equip', desc=' equip h stablilzer or not')
        self.add_input('ac|geom|vstab|equip', desc=' equip v stablilzer or not')
        #self.add_input('ac|geom|hstab|c4_to_wing_c4', units='ft', desc='Distance from wing c/4 to horiz stab c/4 (tail arm distance)')
        # self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        # self.add_input('AR_h', desc='Horiz stab aspect ratio')
        # self.add_input('AR_v', units='rad', desc='Vert stab aspect ratio')
        # self.add_input('troot_h', units='ft', desc='Horiz stab root thickness (ft)')
        # self.add_input('troot_v', units='ft', desc='Vert stab root thickness (ft)')
        #self.add_input('ac|q_cruise', units='lb*ft**-2', desc='Cruise dynamic pressure')

        self.add_output('W_empennage', units='lb', desc='Empennage weight')
        self.declare_partials(['W_empennage'], ['ac|geom|hstab|S_ref','ac|geom|vstab|S_ref'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        #USAF method, Roskam PVC5pg72eq5.14/15
        # bh = math.sqrt(inputs['ac|geom|hstab|S_ref']*inputs['AR_h'])
        # bv = math.sqrt(inputs['ac|geom|vstab|S_ref']*inputs['AR_v'])
        # # Wh = 127 * ((inputs['ac|weights|MTOW']*n_ult/1e5)**0.87 * (inputs['ac|geom|hstab|S_ref']/100)**1.2 * 0.289*(inputs['ac|geom|hstab|c4_to_wing_c4']/10)**0.483 * (bh/inputs['troot_h'])**0.5)**0.458
        # # #Wh_raymer = 0.016 * (n_ult*inputs['ac|weights|MTOW'])**0.414 * inputs['ac|q_cruise']**0.168 * inputs['ac|geom|hstab|S_ref']**0.896 * (100 * 0.18)**-0.12 * (inputs['AR_h'])**0.043 * 0.7**-0.02
        # # Wv = 98.5 * ((inputs['ac|weights|MTOW']*n_ult/1e5)**0.87 * (inputs['ac|geom|vstab|S_ref']/100)**1.2 * 0.289 * (bv/inputs['troot_v'])**0.5)**0.458

        # # Wemp_USAF = Wh + Wv

        #Torenbeek, Roskam PVC5p73eq5.16
        if inputs['ac|geom|hstab|equip'] == 1 and inputs['ac|geom|vstab|equip'] == 1:
            Wemp_Torenbeek = 0.04 * (n_ult * ((inputs['ac|geom|vstab|S_ref']) + (inputs['ac|geom|hstab|S_ref']))**2)**0.75
            outputs['W_empennage'] = Wemp_Torenbeek
        elif inputs['ac|geom|hstab|equip'] == 0 and inputs['ac|geom|vstab|equip'] == 1:
            Wemp_Torenbeek = 0.04 * (n_ult * (inputs['ac|geom|vstab|S_ref'])**2)**0.75
            outputs['W_empennage'] = Wemp_Torenbeek
        elif inputs['ac|geom|hstab|equip'] == 1 and inputs['ac|geom|vstab|equip'] == 0:
            Wemp_Torenbeek = 0.04 * (n_ult * (inputs['ac|geom|hstab|S_ref'])**2)**0.75
            outputs['W_empennage'] = Wemp_Torenbeek
        elif inputs['ac|geom|hstab|equip'] == 0 and inputs['ac|geom|vstab|equip'] == 0:        
            outputs['W_empennage'] = 0
    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']
        if inputs['ac|geom|hstab|equip'] == 1 and inputs['ac|geom|vstab|equip'] == 1:
            J['W_empennage','ac|geom|hstab|S_ref'] = 0.75* 0.04 * (n_ult * ((inputs['ac|geom|vstab|S_ref'] ) + (inputs['ac|geom|hstab|S_ref'] ))**2)**(0.75-1)*(n_ult * 2* ((inputs['ac|geom|vstab|S_ref'] ) + (inputs['ac|geom|hstab|S_ref']))) 
            J['W_empennage','ac|geom|vstab|S_ref'] = 0.75* 0.04 * (n_ult * ((inputs['ac|geom|vstab|S_ref'] ) + (inputs['ac|geom|hstab|S_ref'] ))**2)**(0.75-1)*(n_ult * 2* ((inputs['ac|geom|vstab|S_ref'] ) + (inputs['ac|geom|hstab|S_ref']))) 
        elif inputs['ac|geom|hstab|equip'] == 0 and inputs['ac|geom|vstab|equip'] == 1:
            #J['W_empennage','ac|geom|hstab|S_ref'] = None
            J['W_empennage','ac|geom|vstab|S_ref'] = 0.75*0.04*(n_ult * ((inputs['ac|geom|vstab|S_ref']))**2)**(0.75-1)*(n_ult*2*(inputs['ac|geom|vstab|S_ref']))
        elif inputs['ac|geom|hstab|equip'] == 1 and inputs['ac|geom|vstab|equip'] == 0:
            J['W_empennage','ac|geom|hstab|S_ref'] = 0.75*0.04*(n_ult * ((inputs['ac|geom|hstab|S_ref']))**2)**(0.75-1)*(n_ult*2*(inputs['ac|geom|hstab|S_ref']))
            #J['W_empennage','ac|geom|vstab|S_ref'] = None


class FuselageWeight_eVTOL(ExplicitComponent):
    """
    Inputs: ac|geom|weights|MTOW, ac|geom|fuselage|length, ac|geom|fuselage|height, ac|geom|fuselage|width
    Outputs: W_fuselage
    Metadata: n_ult (ult load factor)

    """
    def initialize(self):
        #self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        #define configuration parameters
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')


    def setup(self):
        #nn = self.options['num_nodes']
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|geom|fuselage|length', units='ft', desc='Fuselage length (not counting nacelle')
        self.add_input('Body_S_wet', units='ft**2', desc='Body reference area')

        #self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('W_fuselage', units='lb', desc='Fuselage weight')
        self.declare_partials(['W_fuselage'], ['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        #USAF method, Roskam PVC5pg76eq5.25
        # W_fuselage_USAF = 200*((inputs['ac|weights|MTOW']*n_ult/1e5)**0.286 * (inputs['ac|geom|fuselage|length']/10)**0.857 * (inputs['ac|geom|fuselage|width']+inputs['ac|geom|fuselage|height'])/10 * (inputs['V_C']/100)**0.338)**1.1
        # print(W_fuselage_USAF)

        #W_fuselage_Torenbeek = 0.021 * 1.08 * ((inputs['V_MO']*inputs['ac|geom|hstab|c4_to_wing_c4']/(inputs['ac|geom|fuselage|width']+inputs['ac|geom|fuselage|height']))**0.5 * inputs['ac|geom|fuselage|S_wet']**1.2)
        
        # Prouty's method, Prouty Helicoptor performance, Stability, and Control: pg664
        # W_fuselage_Prouty
        # Wetted area: Roskam PVC1pg122eqtable3.5, Twin Engine Propeller driven
        #S_wet = 10**(0.8635) * inputs['ac|weights|MTOW']**(0.5632) # MTOW = 3500 (lb) -> S_wet = 723.634
        W_fuselage_Prouty = 6.9 * (inputs['ac|weights|MTOW']/1000)**0.49 * inputs['ac|geom|fuselage|length']**0.61 * inputs['Body_S_wet']**0.25 # MTOW 3500 : 536.4909
        outputs['W_fuselage'] = W_fuselage_Prouty
        # W_fuselage_Prouty = (6.9/(1000**0.49)*10**0.8635) * MTOW**1.0532
    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']
        J['W_fuselage','ac|weights|MTOW'] =  0.114563209190664*inputs['ac|geom|fuselage|length']**0.61*inputs['Body_S_wet']**0.25/inputs['ac|weights|MTOW']**0.51 # MTOW 3500 : 536.4909
        J['W_fuselage','ac|geom|fuselage|length'] = 0.14261950531899*inputs['ac|weights|MTOW']**0.49*inputs['Body_S_wet']**0.25/inputs['ac|geom|fuselage|length']**0.39
        J['W_fuselage','Body_S_wet'] = 0.0584506169340124*inputs['ac|weights|MTOW']**0.49*inputs['ac|geom|fuselage|length']**0.61/inputs['Body_S_wet']**0.75

class NacelleWeight_Multi_eVTOL(ExplicitComponent):
    """
    Inputs: MTOW, ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|c4sweep, ac|geom|wing|taper, ac|geom|wing|toverc, V_H (max SL speed)
    Outputs: W_nacelle
    Metadata: n_ult (ult load factor)

    """
    def initialize(self):
        #self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        #define configuration parameters
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')


    def setup(self):
        #nn = self.options['num_nodes']
        self.add_input('ac|propulsion|motor|rating', units='hp', desc='Takeoff power')
        self.add_input('ac|propulsion|motor|num_motors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|nacelle|equip', desc='equip_nacelle_or_not')

        #self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('W_nacelle', units='lb', desc='Nacelle weight')
        self.declare_partials(['W_nacelle'], ['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        #Torenbeek method, Roskam PVC5pg78eq5.33
        W_nacelle = 0.14*inputs['ac|propulsion|motor|rating'] * inputs['ac|propulsion|motor|num_motors'] * inputs['ac|propulsion|nacelle|equip']
        outputs['W_nacelle'] = W_nacelle 

    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']
        #Torenbeek method, Roskam PVC5pg78eq5.30
        J['W_nacelle','ac|propulsion|motor|rating'] =  0.14 * inputs['ac|propulsion|motor|num_motors'] * inputs['ac|propulsion|nacelle|equip']
        J['W_nacelle','ac|propulsion|motor|num_motors'] = 0.14*inputs['ac|propulsion|motor|rating'] * inputs['ac|propulsion|nacelle|equip']
        J['W_nacelle','ac|propulsion|nacelle|equip'] = 0.14*inputs['ac|propulsion|motor|rating'] * inputs['ac|propulsion|motor|num_motors'] 

class LandingGearWeight_eVTOL(ExplicitComponent):
    """Inputs: MTOW, ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|c4sweep, ac|geom|wing|taper, ac|geom|wing|toverc, V_H (max SL speed)
    Outputs: W_gear
    Metadata: n_ult (ult load factor)

    """
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')

    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb',desc='Max takeoff weight')
        self.add_input('ac|geom|nosegear|equip',desc='equip_landing_gear_or_not')
        self.add_output('W_gear', units='lb', desc='Landing gear weight')
        self.declare_partials(['W_gear'], ['*'])
    
    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
    # W_landinggear_Prouty: 3: number of legs
    # Prouty's method, Prouty Helicoptor performance, Stability, and Control: pg664
        W_landinggear_Prouty = ((40/(1000**0.67)) * inputs['ac|weights|MTOW'] ** 0.67 * 3**0.54) * inputs['ac|geom|nosegear|equip']
        #print(W_landinggear_Prouty)
        outputs['W_gear'] = W_landinggear_Prouty
    

    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']
        J['W_gear','ac|weights|MTOW'] = (0.67 * (40/(1000**0.67)) * inputs['ac|weights|MTOW'] ** (0.67-1) * 3**0.54) * inputs['ac|geom|nosegear|equip']
        J['W_gear','ac|geom|nosegear|equip'] = ((40/(1000**0.67)) * inputs['ac|weights|MTOW'] ** 0.67 * 3**0.54)

class EquipmentWeight_eVTOL(ExplicitComponent):
    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb',desc='Max takeoff weight')
        self.add_input('ac|num_passengers_max',desc='Number of passengers')
        self.add_input('ac|geom|fuselage|length', units='ft', desc='fuselage width')
        self.add_input('ac|geom|wing|AR', desc='Wing aspect ratio')
        self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='Wing reference area')
        self.add_output('W_equipment', units='lb',desc='Equipment weight')
        self.declare_partials(['W_equipment'], ['*'])
    def compute(self, inputs, outputs):
        b = math.sqrt(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])

        #Flight control system (unpowered)
        #Roskam PVC7p98eq7.2
        #Wfc_Cessna = 1.066*inputs['ac|weights|MTOW']**0.626 
        Wfc_Torenbeek = 0.23*inputs['ac|weights|MTOW']**0.666
        #Hydraulic system weight included in flight controls and LG weight
        Whydraulics = 0.2673*1*(inputs['ac|geom|fuselage|length']*b)**0.937

        #Guesstimate of avionics weight
        #This is a guess for a single turboprop class airplane (such as TBM, Pilatus, etc)
        #Wavionics = 2.117*(np.array([110]))**0.933
        #Electrical system weight (NOT including elec propulsion)
        #Welec = 12.57*(inputs['W_fuelsystem']+Wavionics)**0.51

        #pressurization and air conditioning from Roskam
        #Wapi = 0.265*inputs['ac|weights|MTOW']**0.52 * inputs['ac|num_passengers_max']**0.68 * Wavionics**0.17 * 0.95
        #Woxygen = 30 + 1.2*inputs['ac|num_passengers_max']
        
        #Prouty's method, Prouty Helicoptor performance, Stability, and Control: pg665
        Wavionics = 50

        #Prouty's method, Prouty Helicoptor performance, Stability, and Control: pg665
        Waircondition = 8*(inputs['ac|weights|MTOW']/1000)

        #Prouty's method, Prouty Helicoptor performance, Stability, and Control: pg665
        Winstruments = 3.5*(inputs['ac|weights|MTOW']/1000)**1.3
        
        #furnishings (Cessna method)
        Wfur = 0.412*inputs['ac|num_passengers_max']**1.145 * inputs['ac|weights|MTOW'] ** 0.489
        Wpaint = 0.003 * inputs['ac|weights|MTOW']

        outputs['W_equipment'] = Wfc_Torenbeek + Waircondition + Wfur + Wpaint + Winstruments + Wavionics #+ Whydraulics #+ Wapi

    def compute_partials(self, inputs, J):
        J['W_equipment','ac|weights|MTOW'] =  0.201468 * inputs['ac|weights|MTOW'] ** (-0.511) * inputs['ac|num_passengers_max'] ** (1.145) + 0.15318*inputs['ac|weights|MTOW'] ** (-0.334) + 0.000572811062366346 * inputs['ac|weights|MTOW'] ** 0.3 + 0.011
        J['W_equipment','ac|num_passengers_max'] = 0.47174*inputs['ac|weights|MTOW']**0.489*inputs['ac|num_passengers_max']**0.145
        J['W_equipment','ac|geom|fuselage|length'] = 0
        J['W_equipment','ac|geom|wing|AR'] = 0
        J['W_equipment','ac|geom|wing|S_ref'] = 0
        

class eVTOLPropellerAndHub(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')
    def setup(self):
        # Po-Chih's Propeller regression model, applicable for 3, 4, and 5- blades propeller (both aluminium and composite)
        
        self.add_input('ac|propulsion|propeller|diameter', units='inch', desc='Propeller diameter')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|propeller|num_blades', units= None, desc=' Number of blades')
        self.add_input('ac|propulsion|propeller|blade_chord', units='ft', desc=' Main blade chord length')
        self.add_input('ac|propulsion|propeller|proprpm', units='rpm', desc=' Rotor rpm')
        self.add_input('ac|propulsion|propeller|propeller_type', units=None, desc=' eVTOL type')
        
        self.add_output('W_Prop_hub', units = 'lb')
        self.add_output('W_Main_balde', units = 'lb')

        self.declare_partials(['W_Prop_hub'], ['ac|propulsion|propeller|num_rotors','ac|propulsion|propeller|diameter'])
        self.declare_partials(['W_Main_balde'], ['ac|propulsion|propeller|diameter','ac|propulsion|propeller|num_rotors','ac|propulsion|propeller|num_blades','ac|propulsion|propeller|blade_chord','ac|propulsion|propeller|proprpm'])
        # Single motor thrust. multiply to number of rotor.
    def compute(self, inputs, outputs):
        if inputs['ac|propulsion|propeller|propeller_type'] == 1: # Propeller
            W_Prop_hub =  (0.0003 * (inputs['ac|propulsion|propeller|diameter']) ** 2.8094) * inputs['ac|propulsion|propeller|num_rotors']
            outputs['W_Prop_hub'] = W_Prop_hub
            outputs['W_Main_balde'] = 0
        elif inputs['ac|propulsion|propeller|propeller_type'] == 0: # Rotor
            # 12 for ft to inch 
            #omega = rpm*pi/30
            V_tip = inputs['ac|propulsion|propeller|proprpm'] * (inputs['ac|propulsion|propeller|diameter']/12) * np.pi/60 
            outputs['W_Main_balde'] = 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * inputs['ac|propulsion|propeller|blade_chord'] * (inputs['ac|propulsion|propeller|diameter']/(2*12))**1.3 * V_tip ** 0.67 * inputs['ac|propulsion|propeller|num_rotors']
            outputs['W_Prop_hub'] = outputs['W_Main_balde']
        else:
            pass

    def compute_partials(self, inputs, J):
        if inputs['ac|propulsion|propeller|propeller_type'] == 1:
            J['W_Prop_hub','ac|propulsion|propeller|diameter'] = 0.0003 * 2.8094 * (inputs['ac|propulsion|propeller|diameter']) ** (2.8094-1) * inputs['ac|propulsion|propeller|num_rotors']
            J['W_Prop_hub','ac|propulsion|propeller|num_rotors'] = (0.0003 * (inputs['ac|propulsion|propeller|diameter']) ** 2.8094) 
        elif inputs['ac|propulsion|propeller|propeller_type'] == 0:
            V_tip = inputs['ac|propulsion|propeller|proprpm'] * inputs['ac|propulsion|propeller|diameter']/12 * np.pi/60 
            J['W_Main_balde','ac|propulsion|propeller|diameter'] = 1.97 * 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * inputs['ac|propulsion|propeller|blade_chord'] * ( inputs['ac|propulsion|propeller|proprpm'] * np.pi /30)**0.67 * (inputs['ac|propulsion|propeller|diameter']/(2*12)) ** 0.97 * 1/24 * inputs['ac|propulsion|propeller|num_rotors']
            J['W_Main_balde','ac|propulsion|propeller|num_rotors'] = 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * (inputs['ac|propulsion|propeller|blade_chord']) * (inputs['ac|propulsion|propeller|diameter']/2/12)**1.3 * V_tip ** 0.67 
            J['W_Main_balde','ac|propulsion|propeller|num_blades'] = 0.66 * 0.026 * inputs['ac|propulsion|propeller|num_blades']**(0.66-1) * inputs['ac|propulsion|propeller|blade_chord'] * (inputs['ac|propulsion|propeller|proprpm'] * np.pi/30) ** 0.67 * (inputs['ac|propulsion|propeller|diameter']/(12*2)) ** 1.97
            J['W_Main_balde','ac|propulsion|propeller|blade_chord'] = 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * (inputs['ac|propulsion|propeller|diameter']/2/12)**1.3 * V_tip ** 0.67
            J['W_Main_balde','ac|propulsion|propeller|proprpm'] = 0.67 * 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * inputs['ac|propulsion|propeller|blade_chord'] * ( inputs['ac|propulsion|propeller|proprpm'] * np.pi/30 ) ** (0.67-1) * (inputs['ac|propulsion|propeller|diameter']/(12*2)) ** 1.97 * (np.pi/30)
        else:
            pass
"""

class eVTOLRotorblade(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')
    def setup(self):
        # Helicopter Performance, Stability, and Control by Prouty
        self.add_input('ac|propulsion|propeller|diameter', units='ft', desc=' Main rotor blade diameter')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('ac|propulsion|propeller|num_blades', units= None, desc=' Number of blades')
        self.add_input('ac|propulsion|propeller|blade_chord', units='ft', desc=' Main blade chord length')
        self.add_input('ac|propulsion|propeller|proprpm', units='rpm', desc=' Rotor rpm')
        self.add_output('W_Main_balde', units = 'lb')

        self.declare_partials(['W_Main_balde'], ['ac|propulsion|propeller|diameter','ac|propulsion|propeller|num_rotors','ac|propulsion|propeller|num_blades','ac|propulsion|propeller|blade_chord','ac|propulsion|propeller|proprpm'])
        # Single motor thrust. multiply to number of rotor.
    def compute(self, inputs, outputs):
        #omega = rpm*pi/30
        V_tip = inputs['ac|propulsion|propeller|proprpm'] * inputs['ac|propulsion|propeller|diameter'] * np.pi/60 
        outputs['W_Main_balde'] = 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * (inputs['ac|propulsion|propeller|blade_chord']) * (inputs['ac|propulsion|propeller|diameter']/2)**1.3 * V_tip ** 0.67 * inputs['ac|propulsion|propeller|num_rotors']

    def compute_partials(self, inputs, J):
        V_tip = inputs['ac|propulsion|propeller|proprpm'] * inputs['ac|propulsion|propeller|diameter'] * np.pi/60 
        J['W_Main_balde','ac|propulsion|propeller|diameter'] = 1.97 * 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * inputs['ac|propulsion|propeller|blade_chord'] * ( inputs['ac|propulsion|propeller|proprpm'] * np.pi /30)**0.67 * (inputs['ac|propulsion|propeller|diameter']/2) ** 0.97 * inputs['ac|propulsion|propeller|num_rotors']
        J['W_Main_balde','ac|propulsion|propeller|num_rotors'] = 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * (inputs['ac|propulsion|propeller|blade_chord']) * (inputs['ac|propulsion|propeller|diameter']/2)**1.3 * V_tip ** 0.67 
        J['W_Main_balde','ac|propulsion|propeller|num_blades'] = 0.66 * 0.026 * inputs['ac|propulsion|propeller|num_blades']**(0.66-1) * (inputs['ac|propulsion|propeller|blade_chord']) * (inputs['ac|propulsion|propeller|diameter']/2)**1.3 * V_tip ** 0.67
        J['W_Main_balde','ac|propulsion|propeller|blade_chord'] = 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * (inputs['ac|propulsion|propeller|diameter']/2)**1.3 * V_tip ** 0.67
        J['W_Main_balde','ac|propulsion|propeller|proprpm'] = 0.67 * 0.026 * inputs['ac|propulsion|propeller|num_blades']**0.66 * (inputs['ac|propulsion|propeller|blade_chord']) * (inputs['ac|propulsion|propeller|diameter']/2)**1.3 * V_tip ** 0.67 * np.pi/30
"""

class eVTOLMotor(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')
    
    def setup(self):
        # Compute TOTAL eVTOL motor weight.
        # Akshay R. Kadhiresan and Michael J. Duffy Table2.
        self.add_input('ac|propulsion|motor|rating', units='kW', desc='Takeoff power')
        # number of rotor 
        self.add_input('ac|propulsion|motor|num_motors', desc='Number_of_rotor')
        self.add_output('W_Motor', units = 'kg')
        self.declare_partials(['W_Motor'], ['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        motor_powerdensity = 5 # kW / kg
        W_Motor = 1/motor_powerdensity * inputs['ac|propulsion|motor|rating'] * inputs['ac|propulsion|motor|num_motors']
        
        outputs['W_Motor'] = W_Motor

    def compute_partials(self, inputs, J):
        motor_powerdensity = 5 # kW / kg
        J['W_Motor','ac|propulsion|motor|rating'] = 1/motor_powerdensity * inputs['ac|propulsion|motor|num_motors']
        J['W_Motor','ac|propulsion|motor|num_motors'] = 1/motor_powerdensity * inputs['ac|propulsion|motor|rating']

class eVTOLMotorController(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')
    
    def setup(self):
        # Rotor Torque, Conceptual Design and MIssion Analysis for eVTOL UAM flight vehicle configuration.
        # Akshay R. Kadhiresan and Michael J. Duffy Table2.
        #self.add_input('P_TO', units = 'kW', desc='Motor power')
        self.add_input('ac|propulsion|motor|rating', units='kW', desc='Takeoff power') # Total power needed
        # number of rotor 
        self.add_input('ac|propulsion|motor|num_motors', desc='Number_of_rotor')
        self.add_output('W_Motor_Controller', units = 'lb')
        self.declare_partials(['W_Motor_Controller'], ['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        #W_Motor_Controller = 2.20462*(0.12537 * ( inputs['P_TO'] - 2 ) + 0.1) * inputs['ac|propulsion|motor|num_motors']
        W_Motor_Controller = 2.20462*(0.12537 * ( inputs['ac|propulsion|motor|rating'] - 2 ) + 0.1) * inputs['ac|propulsion|motor|num_motors']
        outputs['W_Motor_Controller'] = W_Motor_Controller

    def compute_partials(self, inputs, J):
        #J['W_Motor_Controller','P_TO'] = 2.20462*(0.12537 * ( 1 - 2 )+0.1) * inputs['ac|propulsion|motor|num_motors']
        J['W_Motor_Controller','ac|propulsion|motor|rating'] = 0.2763932094*inputs['ac|propulsion|motor|num_motors']
        J['W_Motor_Controller','ac|propulsion|motor|num_motors'] = 0.2763932094*inputs['ac|propulsion|motor|rating'] - 0.3323244188

class eVTOLEmptyWeight(Group):


    def setup(self):
        # commented out code are used to compute the weight if you want to run this run scirpt solely. 
        # The order is fixed to derive necessary information from upstreams.

        const = self.add_subsystem('const',IndepVarComp(),promotes_outputs=["*"])
        const.add_output('W_fluids', val=0, units='kg')
        const.add_output('structural_fudge', val=1, units='m/m')
        const.add_output('propeller_fudge', val=1, units='m/m')
        #self.add_subsystem('CompDiskLoad',ComputeDiskLoad(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('wing',WingWeight_eVTOL(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('empennage',EmpennageWeight_eVTOL(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('Comp_S_wet',CompBody_S_wet(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('fuselage',FuselageWeight_eVTOL(),promotes_inputs=["*"],promotes_outputs=["*"])
        ##self.add_subsystem('Comp_P_TO',SeaLevelTakeoffPower(),promotes_inputs=["*"],promotes_outputs=["*"])  # This system computes the power needed for hovering
        #self.add_subsystem('Comp_P_Hover',HoverPower(),promotes_inputs=["*"],promotes_outputs=["*"])
        #self.add_subsystem('Comp_induced_velocuty',RotorInducedVelocity(),promotes_inputs=["*"],promotes_outputs=["*"])
        ##self.add_subsystem('Comp_additional_climb_Power',AddVerticalPower(),promotes_inputs=["*"],promotes_outputs=["*"])
        #self.add_subsystem('Comp_Velocity_ratio',VelocityRatio(),promotes_inputs=["*"],promotes_outputs=["*"])
        #self.add_subsystem('Comp_Vertical_ower',VerticalPower(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('nacelle',NacelleWeight_Multi_eVTOL(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('gear',LandingGearWeight_eVTOL(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('equipment',EquipmentWeight_eVTOL(), promotes_inputs=["*"],promotes_outputs=["*"])
        
        self.add_subsystem('PropellerAndHub',eVTOLPropellerAndHub(), promotes_inputs=["*"],promotes_outputs=["*"])
        #self.add_subsystem('Rotorblade',eVTOLRotorblade(), promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('Motor',eVTOLMotor(), promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('eVTOLMotorController',eVTOLMotorController(), promotes_inputs=["*"],promotes_outputs=["*"])
        #self.add_subsystem('totalPowerVerticalClimb',AddSubtractComp(output_name='P_climb',input_names=['P_Hover','P_addvert'], units='kW'),promotes_outputs=['*'],promotes_inputs=["*"])
        

        self.add_subsystem('structural',AddSubtractComp(output_name='W_structure',input_names=['W_wing','W_fuselage','W_nacelle','W_empennage','W_gear'], units='lb'),promotes_outputs=['*'],promotes_inputs=["*"])
        self.add_subsystem('structural_fudge',ElementMultiplyDivideComp(output_name='W_structure_fudged',input_names=['W_structure','structural_fudge'],input_units=['lb','m/m']),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('prop_fudge',ElementMultiplyDivideComp(output_name='W_prop_fudged',input_names=['W_Prop_hub','propeller_fudge'],input_units=['lb','m/m']),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('rotor_fudge',ElementMultiplyDivideComp(output_name='W_Main_balde_fudged',input_names=['W_Main_balde','propeller_fudge'],input_units=['lb','m/m']),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('totalempty',AddSubtractComp(output_name='OEW',input_names=['W_structure_fudged','W_equipment','W_prop_fudged','W_Main_balde_fudged','W_Motor','W_Motor_Controller'], units='lb'),promotes_outputs=['*'],promotes_inputs=["*"])
        
        


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem
    prob = Problem()
    prob.model = Group()
    
    dv_comp = prob.model.add_subsystem('dvs',DictIndepVarComp(acdata),promotes_outputs=["*"])
    
    #dv_comp = self.add_subsystem('dv_comp', DictIndepVarComp(acdata), promotes_outputs=["*"])

    # add eVTOL parameters
    dv_comp.add_output_from_dict('ac|aero|CLmax_TO')
    dv_comp.add_output_from_dict('ac|aero|CD0_body')
    dv_comp.add_output_from_dict('ac|aero|polar|e')
    dv_comp.add_output_from_dict('ac|aero|polar|CD0_TO')
    dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')
    dv_comp.add_output_from_dict('ac|aero|polar|CD0')

    dv_comp.add_output_from_dict('ac|geom|wing|equip')
    dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
    dv_comp.add_output_from_dict('ac|geom|wing|AR')
    dv_comp.add_output_from_dict('ac|geom|wing|c4sweep')
    dv_comp.add_output_from_dict('ac|geom|wing|taper')
    dv_comp.add_output_from_dict('ac|geom|wing|toverc')
    dv_comp.add_output_from_dict('ac|geom|hstab|equip')
    dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
    dv_comp.add_output_from_dict('ac|geom|hstab|c4_to_wing_c4')
    dv_comp.add_output_from_dict('ac|geom|vstab|equip')
    dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
    
    
    #dv_comp.add_output_from_dict('ac|geom|body|S_ref')
    #dv_comp.add_output_from_dict('ac|geom|fuselage|S_wet')
    dv_comp.add_output_from_dict('ac|geom|fuselage|width')
    dv_comp.add_output_from_dict('ac|geom|fuselage|length')
    dv_comp.add_output_from_dict('ac|geom|fuselage|height')
    dv_comp.add_output_from_dict('ac|geom|nosegear|length')
    dv_comp.add_output_from_dict('ac|geom|nosegear|equip')
    dv_comp.add_output_from_dict('ac|geom|maingear|length')
    dv_comp.add_output_from_dict('ac|geom|maingear|equip')

    dv_comp.add_output_from_dict('ac|weights|MTOW')
    dv_comp.add_output_from_dict('ac|weights|W_battery')

    dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')
    dv_comp.add_output_from_dict('ac|propulsion|propeller|num_rotors')
    #dv_comp.add_output_from_dict('ac|propulsion|propeller|proprpm')
    dv_comp.add_output_from_dict('ac|propulsion|propeller|coaxialprop')

    dv_comp.add_output_from_dict('ac|propulsion|motor|rating')
    dv_comp.add_output_from_dict('ac|propulsion|motor|figure_of_merit')
    dv_comp.add_output_from_dict('ac|propulsion|motor|num_motors')
    dv_comp.add_output_from_dict('ac|propulsion|nacelle|equip')

    #dv_comp.add_output('ac|propulsion|propeller|num_rotors', val = 8, units = None)

    dv_comp.add_output_from_dict('ac|num_passengers_max')
    dv_comp.add_output_from_dict('ac|q_cruise')
    #dv_comp.add_output_from_dict('ac|fltcond|vs')

    #dvs = prob.model.add_subsystem('dvs',IndepVarComp(),promotes_outputs=["*"])
    #dvs.add_output('takeoff.fltcond|vs', np.ones((11,))*500., units='ft/min')
    #dvs.add_output('landing.fltcond|vs', np.ones((11,))*(-500), units='ft/min')
    
    """
    AR = 8.95
    dvs.add_output('ac|weights|MTOW',3800, units='lb')
    dvs.add_output('ac|geom|wing|S_ref',257.2, units='ft**2')
    dvs.add_output('ac|geom|wing|AR',AR)
    dvs.add_output('ac|geom|wing|c4sweep',1.0, units='deg')
    dvs.add_output('ac|geom|wing|taper',1)
    dvs.add_output('ac|geom|wing|toverc',0.16)

    dvs.add_output('ac|geom|hstab|S_ref',40, units='ft**2')
    dvs.add_output('ac|geom|vstab|S_ref',25, units='ft**2')
    dvs.add_output('ac|geom|hstab|c4_to_wing_c4',17.9, units='ft')

    dvs.add_output('ac|geom|fuselage|length',50, units='ft')
    dvs.add_output('ac|geom|fuselage|height',5.8, units='ft')
    dvs.add_output('ac|geom|fuselage|width',5.8, units='ft')
    dvs.add_output('ac|geom|fuselage|S_wet',467.929, units='ft**2')

    #dvs.add_output('P_TO',200, units='hp')
    #TODO Takeoff power 

    dvs.add_output('ac|num_passengers_max', 4)
    dvs.add_output('ac|q_cruise', 135.4, units='lb*ft**-2')
    #dvs.add_output('ac|weights|MLW', 7000, units='lb')
    dvs.add_output('ac|geom|nosegear|length', 3, units='ft')
    dvs.add_output('ac|geom|maingear|length', 4, units='ft')
    
    #TODO Takeoff Thrust
    dvs.add_output('ac|propulsion|motor|num_motors', 2)
    dvs.add_output('ac|propulsion|propeller|num_rotors', 8)
    dvs.add_output('ac|propulsion|propeller|diameter', 4, units = 'ft')
    dvs.add_output('ac|propulsion|propeller|disk_load', 12, units='lbf/ft**2')
    """
    

    prob.model.add_subsystem('OEW',eVTOLEmptyWeight(),promotes_inputs=["*"])

    prob.setup()
    prob.run_model()
    print('================= O.E.W. ======================')
    print('Computed takeoff power for single motor (kW):')
    print(prob.get_val('OEW.ac|propulsion|motor|rating', units = 'kW'))

    print('Computed Disk loading (N/m**2) :')
    print(prob.get_val('OEW.diskload', units = 'N/m**2'))
    print('Computed Disk loading (lbf/ft**2) :')
    print(prob.get_val('OEW.diskload', units = 'lbf/ft**2'))
    
    print('Wing weight (lb):')
    print(prob.get_val('OEW.W_wing', units='lb'))

    print('Fuselage weight (lb):')
    print(prob.get_val('OEW.W_fuselage', units='lb'))

    print('Wetted area (m**2):')
    print(prob.get_val('OEW.Body_S_wet', units='m**2'))

    print('Empennage weight (lb):')
    print(prob.get_val('OEW.W_empennage', units='lb'))

    print('TOTAL Nacelle weight (lb):')
    print(prob.get_val('OEW.W_nacelle', units='lb'))

    print('Gear weight (lb):')
    print(prob.get_val('OEW.W_gear', units='lb'))
    print('Equipment weight (lb):')
    print(prob.get_val('OEW.W_equipment', units='lb'))
    print('W_tructure_fudged (lb):')
    print(prob.get_val('OEW.W_structure_fudged', units='lb'))
    
    print('TOTAL rotor and hub weight (lb):')
    print(prob.get_val('OEW.W_Prop_hub', units='lb'))

    #print('W_Prop_hub adjusted (lb):')
    #print(prob.get_val('OEW.W_prop_adjusted', units='lb'))

    print('TOTAL Motor weight (lb):')
    print(prob.get_val('OEW.W_Motor', units='lb'))
    print('TOTAL Motor Conroller weight (lb):')
    print(prob.get_val('OEW.W_Motor_Controller', units='lb'))

    print('Propeller diameter (ft):')
    print(prob.get_val('ac|propulsion|propeller|diameter', units = 'ft'))

    print('====================Power related====================')
    print('Hover power for single motor (hp):')
    print(prob.get_val('OEW.P_Hover', units = 'hp'))

    print('Induced velocity for each rotor (ft/s):')
    print(prob.get_val('OEW.V_induced', units = 'ft/s'))

    #print('Additional power required for one rotor to have a climb rate of 500 ft/min (hp):')
    #print(prob.get_val('OEW.P_addvert', units = 'hp'))
    
    #print('Total power required for one rotor to have a climb rate of -1500 ft/min (hp):')
    #print(prob.get_val('OEW.P_climb', units = 'hp'))

    print('Velocity Ratio: (None)')
    print(prob.get_val('OEW.V_ClimbRatio', units = None))

    print('Total power required for one rotor to have a climb rate of -1500 ft/min (hp):/ method2')
    print(prob.get_val('OEW.P_vert', units = 'hp'))


    print('========================================')
    print('MTOW (lb):')
    print(prob.get_val('ac|weights|MTOW', units = 'lb'))
    print('Operating empty weight (lb):')
    print(prob.get_val('OEW.OEW', units='lb'))

    
    #data = prob.check_partials(compact_print=True)