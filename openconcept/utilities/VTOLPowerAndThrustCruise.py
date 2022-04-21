from __future__ import division
import numpy as np
import openmdao.api as om
from openmdao.api import Group, ExplicitComponent, IndepVarComp, BalanceComp, ExecComp
from openconcept.analysis.atmospherics.density_comp import DensityComp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.components.battery import SOCBattery
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.dvlabel import DVLabel

class PowerAndThrustCalCruise(Group):
    """This is an example model of a MultiRotor propulsion system. 
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes'] 

        dvlist = [['ac|weights|W_battery','batt_weight',500,'kg'],]
        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])

        #introduce model components
        self.add_subsystem('CompDiskLoad', ComputeDiskLoad(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompHoverPower',HoverPower(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompRotorInducedVelocity', RotorInducedVelocity(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompVelocityRatio', VelocityRatio(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompVerticalPower',VerticalPower(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompTotalVerticalPowerandThrust',TotalPowerAndThrustVert(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        # rotor efficiency is Figure of merit

        self.add_subsystem('batt1', SOCBattery(num_nodes=nn, efficiency=0.97),promotes_inputs=["duration","specific_energy"])
        self.connect('power','batt1.elec_load')
        self.connect('batt_weight','batt1.battery_weight')
        """
        addpower = AddSubtractComp(output_name='motors_elec_load',input_names=['motor1_elec_load','motor2_elec_load','motor3_elec_load','motor4_elec_load'], units='kW',vec_size=nn)
        addpower.add_equation(output_name='thrust',input_names=['prop1_thrust','prop2_thrust','prop3_thrust','prop4_thrust'], units='N',vec_size=nn)
        self.add_subsystem('add_power',subsys=addpower,promotes_outputs=['*'])
        self.connect('motor1.elec_load','add_power.motor1_elec_load')
        self.connect('prop1.thrust','add_power.prop1_thrust')

        self.connect('motor1.shaft_power_out','prop1.shaft_power_in')
        self.connect('motor2.shaft_power_out','prop2.shaft_power_in')
        self.connect('motor3.shaft_power_out','prop3.shaft_power_in')
        """

class TotalPowerAndThrustVert(om.ExplicitComponent):
    """ 
    Compute the thrust needed for the vertical climb and descent
    Inputs
    ------
    P_vert : float
        Power needed for vertical climb and descent (vector, h.p.)
    FM : float
        Figure of Merit (scalar, dimensionless)
    fltcond|Ueas : float 
        Vertical speed (vector, ft/s)
    ac|propulsion|propeller|num_rotors
        Number of rotors (scalar, dimensionless)

    Output
    ------
    thrust_total : vector
        Total thrust generated from all rotor (vector, N)
    power : vector 
        Power needed from all motor (vector, hp)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('P_vert', shape = (nn,), units='hp', desc='Power needed for vertical climb and descent')
        self.add_input('FM',val = 0.8 , units=None, desc='Figure of Merit')
        self.add_input('fltcond|Ueas', shape = (nn,), units='ft/s', desc='Vertical speed')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        #self.add_input('prop_eta', shape = (nn,), units = None, desc='propelelr efficiency')

        self.add_output('thrust', shape = (nn,), units='lbf',desc = 'Total thrust generated from all rotor')
        self.add_output('power', shape = (nn,), units='hp',desc = 'Power needed from all motor')

        self.declare_partials(['thrust'], ['P_vert'], rows=arange, cols=arange)
        self.declare_partials(['thrust'], ['fltcond|Ueas'], rows=arange, cols=arange)
        self.declare_partials(['thrust'], ['FM'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['thrust'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        #self.declare_partials(['thrust_total'], ['prop_eta'], rows=arange, cols=arange)

        self.declare_partials(['power'], ['P_vert'], rows=arange, cols=arange)
        self.declare_partials(['power'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['thrust'] = inputs['ac|propulsion|propeller|num_rotors'] * ( 550 * inputs['P_vert'] * inputs['FM'] / inputs['fltcond|Ueas']) #* inputs['prop_eta']
        outputs['power'] = inputs['ac|propulsion|propeller|num_rotors'] * inputs['P_vert']

    def compute_partials(self, inputs, partials):
        partials['thrust', 'P_vert'] = 550 * inputs['ac|propulsion|propeller|num_rotors'] * inputs['FM'] / inputs['fltcond|Ueas']
        partials['thrust', 'fltcond|Ueas'] = 550 * inputs['ac|propulsion|propeller|num_rotors'] * (- inputs['P_vert'] * inputs['FM'] * inputs['fltcond|Ueas']**(-2))
        partials['thrust', 'FM'] = 550 * inputs['ac|propulsion|propeller|num_rotors'] * (inputs['P_vert'] * inputs['fltcond|Ueas'])
        partials['thrust', 'ac|propulsion|propeller|num_rotors'] = 550 * inputs['P_vert'] * inputs['FM'] / inputs['fltcond|Ueas']
        #partials['thrust', 'prop_eta'] = inputs['ac|propulsion|propeller|num_rotors'] * (inputs['P_vert'] * inputs['FM'] / inputs['fltcond|Ueas'])
        partials['power', 'P_vert'] = inputs['ac|propulsion|propeller|num_rotors']
        partials['power', 'ac|propulsion|propeller|num_rotors'] = inputs['P_vert']

class ComputeDiskLoad(om.ExplicitComponent):
    """ 
    Compute the disk loading for single rotor
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW (scalar, lb)
    ac|propulsion|propeller|num_rotors : int
        Number of rotors (scalar, dimensionless)
    ac|propulsion|propeller|diameter
        Rotor diameter (scalar, ft)
    FM : float
        Figure of merit for calculating required hover power
    Output
    ------
    diskload : float
        Disk load for single rotor (scalar, lbf/ft**2)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        #self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('thrust', shape = (nn,), units='lbf', desc='motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|propeller|diameter', units='ft', desc='Rotor diameter')
        self.add_output('diskload', shape = (nn,), units='lbf/ft**2',desc = 'Disk load per per propeller')
        self.declare_partials(['diskload'], ['thrust'], rows=arange, cols=arange)
        self.declare_partials(['diskload'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['diskload'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
    
        
    def compute(self, inputs, outputs):
        #outputs['diskload'] = (inputs['ac|weights|MTOW']) / (((inputs['ac|propulsion|propeller|diameter']/2) ** 2) * np.pi * inputs['ac|propulsion|propeller|num_rotors'])
        outputs['diskload'] = (inputs['thrust']) / (((inputs['ac|propulsion|propeller|diameter']/2) ** 2) * np.pi * inputs['ac|propulsion|propeller|num_rotors'])

    def compute_partials(self, inputs, partials):
        #partials['diskload', 'ac|weights|MTOW'] = 1.27323954473516 / (inputs['ac|propulsion|propeller|diameter'] ** 2 * inputs['ac|propulsion|propeller|num_rotors'])
        #partials['diskload', 'ac|propulsion|propeller|diameter'] = -2.54647908947033 * inputs['ac|weights|MTOW'] / (inputs['ac|propulsion|propeller|diameter'] ** 3 * inputs['ac|propulsion|propeller|num_rotors'])
        #partials['diskload', 'ac|propulsion|propeller|num_rotors'] = -1.27323954473516 * inputs['ac|weights|MTOW'] / (inputs['ac|propulsion|propeller|diameter'] ** 2 * inputs['ac|propulsion|propeller|num_rotors'] ** 2)
        partials['diskload', 'thrust'] = 1.27323954473516 / (inputs['ac|propulsion|propeller|diameter'] ** 2 * inputs['ac|propulsion|propeller|num_rotors'])
        partials['diskload', 'ac|propulsion|propeller|diameter'] = -2.54647908947033 * inputs['thrust'] / (inputs['ac|propulsion|propeller|diameter'] ** 3 * inputs['ac|propulsion|propeller|num_rotors'])
        partials['diskload', 'ac|propulsion|propeller|num_rotors'] = -1.27323954473516 * inputs['thrust'] / (inputs['ac|propulsion|propeller|diameter'] ** 2 * inputs['ac|propulsion|propeller|num_rotors'] ** 2)
        
class HoverPower(om.ExplicitComponent):
    """
    Calculates the minimum power required for single rotor to produce thrust.
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW (scalar, lb)
    ac|propulsion|propeller|num_rotors : int
        Number of rotors (scalar, dimensionless)
    diskload : float
        Disk load for one rotor (scalar, lbf/ft**2)
    FM : float
        Figure of Merit (scalar, dimensionless)
    Output
    ------
    P_Hover : float
        Ideal power, P_Ideal (h.p.) = thrust * sqrt(diskload) / 38, (vector, h.p.)
        P_Hover = P_Ideal / FM
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    FM : float
        Figure of merit for calculating required hover power (default 0.8)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        #self.options.declare('FM', default = 0.8) 
        #self.options.declare('FM', default = 0.5) # 0.5 for AH-1, 7400lb # Cuz 2 props?
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        #self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('thrust', shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('diskload', shape = (nn,), units='lbf/ft**2', desc = 'Disk load per rotor')
        self.add_input('FM', val = 0.8, units=None, desc = 'Figure of merit')
        
        self.add_output('P_Hover', shape = (nn,), units='hp',desc = 'Ideal hover power')
        self.declare_partials(['P_Hover'], ['thrust'], rows=arange, cols=arange)
        self.declare_partials(['P_Hover'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_Hover'], ['diskload'], rows=arange, cols=arange)
        self.declare_partials(['P_Hover'], ['FM'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        #Thrust = (inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors'])
        Thrust = (inputs['thrust']/inputs['ac|propulsion|propeller|num_rotors'])
        P_act = Thrust * np.sqrt(inputs['diskload']) / (38*inputs['FM'])
        outputs['P_Hover'] = P_act
        
    def compute_partials(self, inputs, partials):
        #partials['P_Hover', 'ac|weights|MTOW'] = (inputs['diskload']**0.5)/(38*inputs['ac|propulsion|propeller|num_rotors']**inputs['FM'])
        #partials['P_Hover', 'ac|propulsion|propeller|num_rotors'] = -inputs['ac|weights|MTOW']*inputs['diskload']**0.5/(38*inputs['ac|propulsion|propeller|num_rotors']**2*inputs['FM'])
        #partials['P_Hover', 'diskload'] = 0.0131578947368421*inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['diskload']**0.5*inputs['FM'])
        #partials['P_Hover', 'FM'] = -inputs['ac|weights|MTOW']*inputs['diskload']**0.5/(38*inputs['ac|propulsion|propeller|num_rotors']*inputs['FM']**2)
        partials['P_Hover', 'thrust'] = (inputs['diskload']**0.5)/(38*inputs['ac|propulsion|propeller|num_rotors']**inputs['FM'])
        partials['P_Hover', 'ac|propulsion|propeller|num_rotors'] = -inputs['thrust']*inputs['diskload']**0.5/(38*inputs['ac|propulsion|propeller|num_rotors']**2*inputs['FM'])
        partials['P_Hover', 'diskload'] = 0.0131578947368421*inputs['thrust']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['diskload']**0.5*inputs['FM'])
        partials['P_Hover', 'FM'] = -inputs['thrust']*inputs['diskload']**0.5/(38*inputs['ac|propulsion|propeller|num_rotors']*inputs['FM']**2)

class RotorInducedVelocity(om.ExplicitComponent):
    """
    Computes the rotor induced speed
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW (scalar, lb)
    ac|propulsion|propeller|num_rotors
        Number of rotors (scalar, dimensionless)
    ac|propulsion|propeller|diameter
        Rotor diameter (scalar, ft)
    fltcond|rho : float 
        Density (vector, slug/ft**3)
    Output
    ------
    V_induced : float
        Rotor induced velocity, should be a negative value since flowing downward (vector, ft/s)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        #self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('thrust', shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number of rotor')
        self.add_input('ac|propulsion|propeller|diameter', units = 'ft', desc='Rotor diameter')
        self.add_input('fltcond|rho', val = 0.0023777 ,shape=(nn,), units='slug/ft**3', desc = 'air density')
        
        self.add_output('V_induced', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')
        #self.declare_partials(['V_induced'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['V_induced'], ['thrust'], rows=arange, cols=arange)
        self.declare_partials(['V_induced'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['V_induced'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['V_induced'], ['fltcond|rho'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print(inputs['fltcond|rho'])
        A = (inputs['ac|propulsion|propeller|diameter'] / 2) ** 2 * np.pi
        #outputs['V_induced'] = np.sqrt((inputs['ac|weights|MTOW']/ inputs['ac|propulsion|propeller|num_rotors'])/(2 * inputs['fltcond|rho'] * A))
        outputs['V_induced'] = np.sqrt((inputs['thrust']/ inputs['ac|propulsion|propeller|num_rotors'])/(2 * inputs['fltcond|rho'] * A))
        
    def compute_partials(self, inputs, partials):
        #partials['V_induced', 'ac|weights|MTOW'] = 0.398942280401433*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|weights|MTOW']
        #partials['V_induced', 'ac|propulsion|propeller|num_rotors'] = -0.398942280401433*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|propulsion|propeller|num_rotors']
        #partials['V_induced', 'ac|propulsion|propeller|diameter'] = -0.797884560802865*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|propulsion|propeller|diameter']
        #partials['V_induced', 'fltcond|rho'] = None
        partials['V_induced', 'thrust'] = 0.398942280401433*(inputs['thrust']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['thrust']
        partials['V_induced', 'ac|propulsion|propeller|num_rotors'] = -0.398942280401433*(inputs['thrust']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|propulsion|propeller|num_rotors']
        partials['V_induced', 'ac|propulsion|propeller|diameter'] = -0.797884560802865*(inputs['thrust']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|propulsion|propeller|diameter']
        partials['V_induced', 'fltcond|rho'] = None
    #TODO Import fltcond in the initiation height, for example, vertical landing phase.


class AddVerticalPower(om.ExplicitComponent):
    """
    Calculates additional power required for vertical clambing or landing for a given vertical climb rate.
    Assuming no rotor wake in the down stream.
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW (scalar, lb)
    diskload : float
        Disk load for one rotor (scalar, lbf/ft**2)
    ac|propulsion|propeller|num_rotors : int 
        Number of rotors (scalar, dimensionless)
    fltcond|Ueas : float 
        Vertical speed (vector, ft/s)
    V_induced : float 
        Rotor induced velocity, should be a negative value since flowing downward (vector, ft/s)
    Output
    ------
    P_addvert : float
        Additional power required for climbing in a given vertical climb rate (vector, ft/s)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        #self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('thrust', shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('fltcond|Ueas', val = -25, shape = (nn,), units='ft/s', desc = 'climb rate')
        self.add_input('V_induced', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')

        self.add_output('P_addvert', shape = (nn,), units='hp',desc = 'Additional power required for climbing in a given vertical climb rate')
        self.declare_partials(['P_addvert'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_addvert'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_addvert'], ['fltcond|Ueas'], rows=arange, cols=arange)
        self.declare_partials(['P_addvert'], ['V_induced'], rows=arange, cols=arange)
        self.declare_partials(['P_addvert'], ['thrust'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print('Vs = ',inputs['fltcond|Ueas'])
        nn = self.options['num_nodes']
        if inputs['fltcond|Ueas'] > 0: # vertical climb
            A = (inputs['fltcond|Ueas']/2 + np.sqrt((-inputs['fltcond|Ueas']/2)**2 + inputs['V_induced']**2 ) - inputs['V_induced'] )
        else: # vertical descent
            A = (inputs['fltcond|Ueas']/2 - np.sqrt((inputs['fltcond|Ueas']/2)**2 - inputs['V_induced']**2 ) - inputs['V_induced'] )
        outputs['P_addvert'] = ((inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors'])/550) * A
        
    def compute_partials(self, inputs, partials):
        #partials['P_addvert', 'ac|weights|MTOW'] = (inputs['fltcond|Ueas']/2 - inputs['V_induced'] + (inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        #partials['P_addvert', 'ac|propulsion|propeller|num_rotors'] = -inputs['ac|weights|MTOW']*(inputs['fltcond|Ueas']/2 - inputs['V_induced'] + (inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors']**2)
        #partials['P_addvert', 'fltcond|Ueas'] = inputs['ac|weights|MTOW']*(0.25*inputs['fltcond|Ueas']/(inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5 + 1/2)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        #partials['P_addvert', 'V_induced'] = inputs['ac|weights|MTOW']*(1.0*inputs['V_induced']/(inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5 - 1)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'thrust'] = (inputs['fltcond|Ueas']/2 - inputs['V_induced'] + (inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'ac|propulsion|propeller|num_rotors'] = -inputs['thrust']*(inputs['fltcond|Ueas']/2 - inputs['V_induced'] + (inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors']**2)
        partials['P_addvert', 'fltcond|Ueas'] = inputs['thrust']*(0.25*inputs['fltcond|Ueas']/(inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5 + 1/2)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'V_induced'] = inputs['thrust']*(1.0*inputs['V_induced']/(inputs['fltcond|Ueas']**2/4 + inputs['V_induced']**2)**0.5 - 1)/(550*inputs['ac|propulsion|propeller|num_rotors'])


class VelocityRatio(om.ExplicitComponent):
    """
    Computes the velocity ratio between
    Inputs
    ------
    fltcond|Ueas : float 
        Vertical speed (vector, ft/s)
    V_induced : float 
        Rotor induced velocity, should be a positive since the positive velocity direction is defined downward in momentum theory (vector, ft/s)
    Output
    ------
    V_ClimbRatio : float
        The velocity ratio between climb rate and induced velocity (vector, dimensionless)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('fltcond|Ueas', val = -25, shape = (nn,), units = 'ft/s', desc='Vertical speed')
        self.add_input('V_induced', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')
        
        self.add_output('V_ClimbRatio', shape = (nn,), units=None, desc = 'velocity ratio between climb rate and induced velocity')
        self.declare_partials(['V_ClimbRatio'], ['fltcond|Ueas'], rows=arange, cols=arange)
        self.declare_partials(['V_ClimbRatio'], ['V_induced'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print('fltcond|Ueas = :',inputs['fltcond|Ueas'])
        print('V_induced = :',inputs['V_induced'])
        outputs['V_ClimbRatio'] = inputs['fltcond|Ueas']/inputs['V_induced']
        
    def compute_partials(self, inputs, partials):
        partials['V_ClimbRatio', 'fltcond|Ueas'] = 1/inputs['V_induced']
        partials['V_ClimbRatio', 'V_induced'] = -inputs['fltcond|Ueas']*inputs['V_induced']**(-2)

class VerticalPower(om.ExplicitComponent):
    """
    Computes the velocity ratio between
    Inputs
    ------
    V_ClimbRatio : float 
        The velocity ratio between climb rate and induced velocity (vector, dimensionless)
    P_Hover : float
        Ideal power, P_Ideal (h.p.) = thrust * sqrt(diskload) / 38, (vector, h.p.)
        P_Hover = P_Ideal / FM
    Output
    ------
    P_vert : float
        Power needed for vertical climb and descent (vector, h.p.)
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('V_ClimbRatio', shape = (nn,), units=None, desc = 'velocity ratio between climb rate and induced velocity')
        self.add_input('P_Hover', shape = (nn,), units='hp',desc = 'Ideal hover power')
        
        self.add_output('P_vert', shape = (nn,), units='hp', desc = 'Power needed for vertical climb and descent')
        self.declare_partials(['P_vert'], ['V_ClimbRatio'], rows=arange, cols=arange)
        self.declare_partials(['P_vert'], ['P_Hover'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for ii in range(len(inputs['V_ClimbRatio'])):
            if inputs['V_ClimbRatio'][ii] >= 0:
                outputs['P_vert'] = inputs['P_Hover']*(0.5 * inputs['V_ClimbRatio'] + np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 + 1))
            elif inputs['V_ClimbRatio'][ii] <= -2:
                outputs['P_vert'] = inputs['P_Hover']*(0.5 * inputs['V_ClimbRatio'] - np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 - 1))
            else:  
                outputs['P_vert'] = inputs['P_Hover']*(0.974-0.125*inputs['V_ClimbRatio']-1.372*inputs['V_ClimbRatio']**2-1.718*inputs['V_ClimbRatio']**3-0.665*inputs['V_ClimbRatio']**4)

    def compute_partials(self, inputs, partials):
        for ii in range(len(inputs['V_ClimbRatio'])):
            if inputs['V_ClimbRatio'][ii] > 0:
                partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 + 0.25 * (inputs['V_ClimbRatio']**2 + 4)**(-0.5) * 2 * inputs['V_ClimbRatio'])
                partials['P_vert', 'P_Hover'] = 0.5 * inputs['V_ClimbRatio'] + np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 + 1)
            elif inputs['V_ClimbRatio'][ii] < -2:
                partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 - 0.25 * (inputs['V_ClimbRatio']**2 - 4)**(-0.5) * 2 * inputs['V_ClimbRatio'])
                partials['P_vert', 'P_Hover'] = 0.5 * inputs['V_ClimbRatio'] - np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 - 1)
            else:
                partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(-0.125-2*1.372*inputs['V_ClimbRatio']-3*1.718*inputs['V_ClimbRatio']**2-4*0.665*inputs['V_ClimbRatio']**3)
                partials['P_vert', 'P_Hover'] = 0.974-0.125*inputs['V_ClimbRatio']-1.372*inputs['V_ClimbRatio']**2-1.718*inputs['V_ClimbRatio']**3-0.665*inputs['V_ClimbRatio']**4
