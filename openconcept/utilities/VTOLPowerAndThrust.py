from __future__ import division
from re import A
from matplotlib import units
import numpy as np
import openmdao.api as om
from openmdao.api import Group, ExplicitComponent, IndepVarComp, BalanceComp, ExecComp
from openconcept.analysis.atmospherics.density_comp import DensityComp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.components.battery import SOCBattery
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.dvlabel import DVLabel

class PowerAndThrustCal(Group):
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes'] 

        dvlist = [['ac|weights|W_battery','batt_weight',500,'kg'],
                 ['ac|propulsion|battery|specific_energy','specific_energy',300,'W*h/kg'],]
        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])

        #introduce model components
        self.add_subsystem('CompDiskLoad', ComputeDiskLoad(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompHoverPower',HoverPower(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompRotorInducedVelocity', RotorHoverVelocity(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompVelocityRatio', VelocityRatio(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompVerticalPower',VerticalPower(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompTotalVerticalPowerandThrust',TotalPowerAndThrustVert(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        #self.add_subsystem('CompInflowVelocity',InflowVelocity(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompThrottle',Throttle(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['throttle'])

        # rotor efficiency is Figure of merit
        self.add_subsystem('batt1', SOCBattery(num_nodes=nn, efficiency=0.97),promotes_inputs=["duration","specific_energy"])
        self.connect('power','batt1.elec_load')
        self.connect('batt_weight','batt1.battery_weight')

class PowerAndThrustCruiseMultiRotor(Group):
    """This is an example model of a MultiRotor propulsion system. 
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")
    def setup(self):
        nn = self.options['num_nodes'] 

        dvlist = [['ac|weights|W_battery','batt_weight',500,'kg'],
                 ['ac|propulsion|battery|specific_energy','specific_energy',300,'W*h/kg'],]

        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])

        #introduce model components
        self.add_subsystem('CompDiskLoad', ComputeDiskLoad(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompHoverPower',HoverPower(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompAdvancedRatioNu', AdvancedRatio(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompCT', ThrustCoef(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompHoverInflowRatio', HoverInflowRatio(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])        
        self.add_subsystem('CompInflowRatio', InflowRatio_implicit(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompCruisePower', CruiserPower(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompTotalThrustAndPowerInCruise', TotalPowerAndThrustCruise(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('CompCruiseThrottle', CruiseThrottle(num_nodes=nn),promotes_inputs=['*'], promotes_outputs=['throttle'])
        
        # rotor efficiency is Figure of merit
        self.add_subsystem('batt1', SOCBattery(num_nodes=nn, efficiency=0.97),promotes_inputs=["duration","specific_energy"])
        self.connect('power','batt1.elec_load')
        self.connect('batt_weight','batt1.battery_weight')


class TotalPowerAndThrustVert(om.ExplicitComponent):
    """ 
    Compute the thrust needed for vertical climb and descent
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW, (scaler, lb)
    P_vert : float
        Single rotor power requried in straight level cruise (vector, dimensionless) (vector, h.p.)
    ac|propulsion|propeller|FM : float
        Figure of Merit (scalar, dimensionless)
    fltcond|vs : float 
        Vertical speed (vector, ft/s)
    ac|propulsion|propeller|num_rotors
        Number of rotors (scalar, dimensionless)
    ac|propulsion|propeller|coaxialprop : int
        If the propeller/rotor is coaxial layout or not (scalar, dimensionless)

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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('P_vert', shape = (nn,), units='hp', desc='Power needed for vertical climb and descent')
        self.add_input('ac|propulsion|propeller|FM', units=None, desc='Figure of Merit')
        self.add_input('fltcond|vs', shape = (nn,), units='ft/s', desc='Vertical speed')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|motor|rating', units='hp', desc='Design motor rating')
        self.add_input('ac|propulsion|propeller|coaxialprop', desc='coaxial layout or not')

        self.add_output('thrust', shape = (nn,), units='lbf',desc = 'Total thrust generated from all rotor')
        self.add_output('power', shape = (nn,), units='hp',desc = 'Power needed from all motor')

        self.declare_partials(['thrust'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['power'], ['P_vert'], rows=arange, cols=arange)
        self.declare_partials(['power'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        
    def compute(self, inputs, outputs):
        outputs['thrust'] = inputs['ac|weights|MTOW']
        if inputs['ac|propulsion|propeller|coaxialprop'] == 0:
            outputs['power'] = inputs['ac|propulsion|propeller|num_rotors'] * inputs['P_vert']
        elif inputs['ac|propulsion|propeller|coaxialprop'] == 1:
            outputs['power'] = inputs['ac|propulsion|propeller|num_rotors']/2*inputs['P_vert']*1.281

    def compute_partials(self, inputs, partials):
        partials['thrust', 'ac|weights|MTOW'] = 1
        if inputs['ac|propulsion|propeller|coaxialprop'] == 0:
            partials['power', 'P_vert'] = inputs['ac|propulsion|propeller|num_rotors']
            partials['power', 'ac|propulsion|propeller|num_rotors'] = inputs['P_vert']
        elif inputs['ac|propulsion|propeller|coaxialprop'] == 1:
            partials['power', 'P_vert'] = 1.281 * inputs['ac|propulsion|propeller|num_rotors']/2
            partials['power', 'ac|propulsion|propeller|num_rotors'] = 0.5*inputs['P_vert']*1.281

class TotalPowerAndThrustCruise(om.ExplicitComponent):
    """ 
    Compute the thrust needed for cruise
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW, (scaler, lb)
    P_cruise : float
        Power needed for cruise for one motor (vector, h.p.)
    ac|propulsion|propeller|num_rotors : int
        Number of rotors (scalar, dimensionless)
    ac|propulsion|propeller|coaxialprop : int
        If the propeller/rotor is coaxial layout or not (scalar, dimensionless)

    Output
    ------
    thrust : vector
        Total thrust generated from all rotor in cruise (vector, N)
    power : vector 
        Power needed from all motor in cruise (vector, h.p.)

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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('P_cruise', shape = (nn,), units='hp', desc='Power needed for vertical climb and descent')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|propeller|coaxialprop', desc='coaxial layout or not')
        
        self.add_output('thrust', shape = (nn,), units='lbf',desc = 'Total thrust generated from all rotor in cruise ')
        self.add_output('power', shape = (nn,), units='hp',desc = 'Power needed from all motor in cruise ')

        self.declare_partials(['thrust'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['power'], ['P_cruise'], rows=arange, cols=arange)
        self.declare_partials(['power'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['thrust'] = inputs['ac|weights|MTOW']
        if inputs['ac|propulsion|propeller|coaxialprop'] == 0:
            outputs['power'] = inputs['ac|propulsion|propeller|num_rotors']*inputs['P_cruise']
        elif inputs['ac|propulsion|propeller|coaxialprop'] == 1:
            outputs['power'] = inputs['ac|propulsion|propeller|num_rotors']/2*inputs['P_cruise']*1.281
        

    def compute_partials(self, inputs, partials):
        partials['thrust', 'ac|weights|MTOW'] = 1
        if inputs['ac|propulsion|propeller|coaxialprop'] == 0:
            partials['power', 'P_cruise'] = inputs['ac|propulsion|propeller|num_rotors']
            partials['power', 'ac|propulsion|propeller|num_rotors'] = inputs['P_cruise']
            
        elif inputs['ac|propulsion|propeller|coaxialprop'] == 1:
            partials['power', 'P_cruise'] = 1.281 * inputs['ac|propulsion|propeller|num_rotors']/2
            partials['power', 'ac|propulsion|propeller|num_rotors'] = 0.5*inputs['P_cruise']*1.281
            
        
       
class CruiseThrottle(om.ExplicitComponent):
    """ 
    Compute the throttle needed for cruise
    Inputs
    ------
    P_cruise : float
        Power needed from all motor in cruise (vector, h.p.)
    ac|propulsion|motor|rating
        Design motor rating (vector, hp)
    
    Output
    ------
    throttle : float 
        Power control setting. Should be in between [0, 1]. (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    eta_m : float
        Motor efficiency (default 0.97, dimensionaless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        self.options.declare('efficiency', default=0.97, desc="Motor efficiency")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('P_cruise', shape = (nn,), units=' hp ', desc='Power needed for vertical climb and descent for one motor')
        self.add_input('ac|propulsion|motor|rating', units='hp', desc='Design motor rating')

        self.add_output('throttle', shape = (nn,), units=None,desc = 'Power control setting')

        self.declare_partials(['throttle'], ['P_cruise'], rows=arange, cols=arange)
        self.declare_partials(['throttle'], ['ac|propulsion|motor|rating'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        eta_m = self.options['efficiency']
        outputs['throttle'] = inputs['P_cruise'] / (inputs['ac|propulsion|motor|rating'] * eta_m)

    def compute_partials(self, inputs, partials):
        eta_m = self.options['efficiency']
        partials['throttle', 'P_cruise'] = 1/(inputs['ac|propulsion|motor|rating'] * eta_m)
        partials['throttle', 'ac|propulsion|motor|rating'] =  - (inputs['P_cruise']/(eta_m)) * inputs['ac|propulsion|motor|rating'] ** (-2)

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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|propeller|diameter', units='ft', desc='Rotor diameter')
        self.add_output('diskload', shape = (nn,), units='lbf/ft**2',desc = 'Disk load per per propeller')
        self.declare_partials(['diskload'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['diskload'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['diskload'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
    
        
    def compute(self, inputs, outputs):
        outputs['diskload'] = (inputs['ac|weights|MTOW']) / (((inputs['ac|propulsion|propeller|diameter']/2) ** 2) * np.pi * inputs['ac|propulsion|propeller|num_rotors'])
        

    def compute_partials(self, inputs, partials):
        partials['diskload', 'ac|weights|MTOW'] = 1.27323954473516 / (inputs['ac|propulsion|propeller|diameter'] ** 2 * inputs['ac|propulsion|propeller|num_rotors'])
        partials['diskload', 'ac|propulsion|propeller|diameter'] = -2.54647908947033 * inputs['ac|weights|MTOW'] / (inputs['ac|propulsion|propeller|diameter'] ** 3 * inputs['ac|propulsion|propeller|num_rotors'])
        partials['diskload', 'ac|propulsion|propeller|num_rotors'] = -1.27323954473516 * inputs['ac|weights|MTOW'] / (inputs['ac|propulsion|propeller|diameter'] ** 2 * inputs['ac|propulsion|propeller|num_rotors'] ** 2)
        
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
    ac|propulsion|propeller|FM : float
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
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('diskload', shape = (nn,), units='lbf/ft**2', desc = 'Disk load per rotor')
        self.add_input('ac|propulsion|propeller|FM', units=None, desc = 'Figure of merit')
        
        self.add_output('P_Hover', shape = (nn,), units='hp',desc = 'Ideal hover power')
        self.declare_partials(['P_Hover'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_Hover'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_Hover'], ['diskload'], rows=arange, cols=arange)
        self.declare_partials(['P_Hover'], ['ac|propulsion|propeller|FM'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        Thrust = (inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors'])
        P_act = Thrust * np.sqrt(inputs['diskload']) / (38*inputs['ac|propulsion|propeller|FM'])
        outputs['P_Hover'] = P_act
        
    def compute_partials(self, inputs, partials):
        partials['P_Hover', 'ac|weights|MTOW'] = (inputs['diskload']**0.5)/(38*inputs['ac|propulsion|propeller|num_rotors']*inputs['ac|propulsion|propeller|FM'])
        partials['P_Hover', 'ac|propulsion|propeller|num_rotors'] = -inputs['ac|weights|MTOW']*inputs['diskload']**0.5/(38*inputs['ac|propulsion|propeller|num_rotors']**2*inputs['ac|propulsion|propeller|FM'])
        partials['P_Hover', 'diskload'] = 0.0131578947368421*inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['diskload']**0.5*inputs['ac|propulsion|propeller|FM'])
        partials['P_Hover', 'ac|propulsion|propeller|FM'] = -(inputs['ac|weights|MTOW']*inputs['diskload']**0.5)/(38*inputs['ac|propulsion|propeller|num_rotors'])*(inputs['ac|propulsion|propeller|FM']**(-2))

class RotorHoverVelocity(om.ExplicitComponent):
    """
    Computes the rotor induced speed
    Inputs
    ------
    thrust : float
        Single propeller thrust (scalar, lb)
    ac|propulsion|propeller|num_rotors
        Number of rotors (scalar, dimensionless)
    ac|propulsion|propeller|diameter
        Rotor diameter (scalar, ft)
    fltcond|rho : float 
        Density (vector, slug/ft**3)
    Output
    ------
    V_hover : float
        (Old)Rotor induced velocity, should be a negative value since flowing downward (vector, ft/s)
        (Old)Rotor induced velocity, squart root, velocity is defined positive downward (vector, ft/s)
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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        #self.add_input('thrust', shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number of rotor')
        self.add_input('ac|propulsion|propeller|diameter', units = 'ft', desc='Rotor diameter')
        self.add_input('fltcond|rho', shape=(nn,), units='slug/ft**3', desc = 'air density')
        # 0.002377 is the default value. Can be replaced by other values.
        self.add_output('V_hover', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')
        self.declare_partials(['V_hover'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        #self.declare_partials(['V_hover'], ['thrust'], rows=arange, cols=arange)
        self.declare_partials(['V_hover'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['V_hover'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['V_hover'], ['fltcond|rho'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print(inputs['fltcond|rho'])
        A = (inputs['ac|propulsion|propeller|diameter'] / 2) ** 2 * np.pi
        #outputs['V_hover'] = np.sqrt((inputs['thrust']/ inputs['ac|propulsion|propeller|num_rotors'])/(2 * inputs['fltcond|rho'] * A))
        outputs['V_hover'] = np.sqrt((inputs['ac|weights|MTOW']/ inputs['ac|propulsion|propeller|num_rotors'])/(2 * inputs['fltcond|rho'] * A))
        
    def compute_partials(self, inputs, partials):
        #partials['V_hover', 'thrust'] = 0.398942280401433*(inputs['thrust']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['thrust']
        partials['V_hover', 'ac|weights|MTOW'] = 0.398942280401433*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|weights|MTOW']
        partials['V_hover', 'ac|propulsion|propeller|num_rotors'] = -0.398942280401433*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|propulsion|propeller|num_rotors']
        partials['V_hover', 'ac|propulsion|propeller|diameter'] = -0.797884560802865*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['ac|propulsion|propeller|diameter']
        #partials['V_hover', 'fltcond|rho'] = 0.5* ( inputs['ac|weights|MTOW']/(2*A*inputs['ac|propulsion|propeller|num_rotors']) ) **(0.5) * (1/inputs['fltcond|rho'])**(-0.5) * (-1/inputs['fltcond|rho']**2)
        partials['V_hover', 'fltcond|rho'] = -0.398942280401433*(inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**2))**0.5/inputs['fltcond|rho']

class InflowVelocity(om.ExplicitComponent):
    """
    Computes the rotor inflow velocity for the thrust estimation.
    Inputs
    ------
    fltcond|vs : float 
        Vertical speed (vector, ft/s)
    V_hover : float
        Rotor induced velocity, squart root, velocity is defined positive downward (vector, ft/s)
    V_ClimbRatio : float 
        The velocity ratio between climb rate and induced velocity (vector, dimensionless)
    
    Output
    ------
    V_inflow : float 
        Net inflow velocity (vector, ft/s)

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
        self.add_input('fltcond|vs', shape = (nn,), units='ft/s', desc='vertical speed')
        self.add_input('V_hover', shape = (nn,), units='ft/s', desc='rotor induced velocity')
        self.add_input('V_ClimbRatio', shape = (nn,), units=None, desc='climb velocity over hover velocity')
        
        self.add_output('V_inflow', shape = (nn,), units='ft/s', desc = 'net inflow velocity')
        
        self.declare_partials(['V_inflow'], ['fltcond|vs'], rows=arange, cols=arange)
        self.declare_partials(['V_inflow'], ['V_hover'], rows=arange, cols=arange)
        self.declare_partials(['V_inflow'], ['V_ClimbRatio'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for ii in range(len(inputs['V_ClimbRatio'])):
            if inputs['V_ClimbRatio'][ii] > 0:
                # Rotorcraft Aeromechanics by Wayne johnson, Cambridge P94. Eqn (4.45)
                outputs['V_inflow'] = inputs['fltcond|vs']/2 + np.sqrt((inputs['fltcond|vs']/2)**2+inputs['V_hover']**2)
            elif inputs['V_ClimbRatio'][ii] < 0 and inputs['V_ClimbRatio'][ii] >= -2 :
                # Selfmade surrogate model model from Model for Vortex Ring State Influence on Rotorcraft Flight Dynamics. Fig. 37, VRS Model, Vx/Vh = 0
                outputs['V_inflow'] = inputs['V_hover'] * (0.914*inputs['V_ClimbRatio']**5 + 3.289*inputs['V_ClimbRatio']**4 + 4.587*inputs['V_ClimbRatio']**3 + 3.518*inputs['V_ClimbRatio']**2 + 1.267*inputs['V_ClimbRatio'] + 1.004 ) 
            else:
                raise RuntimeError('Warning: You are reaching turbulant and Windmill state, two algorithms are under development')
        #print('v_inflow',outputs['V_inflow'])
    def compute_partials(self, inputs, partials):
        for ii in range(len(inputs['V_ClimbRatio'])):
            if inputs['V_ClimbRatio'][ii] > 0:
                #print('case1')
                partials['V_inflow', 'fltcond|vs'] = 1/2 * ( 1 + (inputs['fltcond|vs']/2)**2 + inputs['V_hover']**2 )**(-0.5) * inputs['fltcond|vs'] 
                partials['V_inflow', 'V_hover'] = 1/2 * ((inputs['fltcond|vs']/2)**2 + inputs['V_hover']**2 )**(-0.5) *2*inputs['V_hover']
                #partials['V_inflow', 'V_ClimbRatio'] = None
            elif inputs['V_ClimbRatio'][ii] < 0 and inputs['V_ClimbRatio'][ii] >= -2 :
                #print('case2')
                #partials['V_inflow', 'fltcond|vs'] = None
                partials['V_inflow', 'V_hover'] = (0.914*inputs['V_ClimbRatio']**5 + 3.289*inputs['V_ClimbRatio']**4 + 4.587*inputs['V_ClimbRatio']**3 + 3.518*inputs['V_ClimbRatio']**2 + 1.267*inputs['V_ClimbRatio'] + 1.004 ) 
                partials['V_inflow', 'V_ClimbRatio'] = inputs['V_hover'] * (5*0.914*inputs['V_ClimbRatio']**4 + 4*3.289*inputs['V_ClimbRatio']**3 + 3*4.587*inputs['V_ClimbRatio']**2 + 2*3.518*inputs['V_ClimbRatio']**1 + 1.267)
            else:
                raise RuntimeError('Warning: You are reaching turbulant and Windmill state, two algorithms are under development')   


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
    fltcond|vs : float 
        Vertical speed (vector, ft/s)
    V_hover : float 
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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        #self.add_input('thrust', shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('fltcond|vs', val = -25, shape = (nn,), units='ft/s', desc = 'climb rate')
        self.add_input('V_hover', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')

        self.add_output('P_addvert', shape = (nn,), units='hp',desc = 'Additional power required for climbing in a given vertical climb rate')
        self.declare_partials(['P_addvert'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_addvert'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_addvert'], ['fltcond|vs'], rows=arange, cols=arange)
        self.declare_partials(['P_addvert'], ['V_hover'], rows=arange, cols=arange)
        #self.declare_partials(['P_addvert'], ['thrust'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print('Vs = ',inputs['fltcond|vs'])
        nn = self.options['num_nodes']
        if inputs['fltcond|vs'] > 0: # vertical climb
            A = (inputs['fltcond|vs']/2 + np.sqrt((-inputs['fltcond|vs']/2)**2 + inputs['V_hover']**2 ) - inputs['V_hover'] )
        else: # vertical descent
            A = (inputs['fltcond|vs']/2 - np.sqrt((inputs['fltcond|vs']/2)**2 - inputs['V_hover']**2 ) - inputs['V_hover'] )
        outputs['P_addvert'] = ((inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors'])/550) * A
        
    def compute_partials(self, inputs, partials):
        partials['P_addvert', 'ac|weights|MTOW'] = (inputs['fltcond|vs']/2 - inputs['V_hover'] + (inputs['fltcond|vs']**2/4 + inputs['V_hover']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'ac|propulsion|propeller|num_rotors'] = -inputs['ac|weights|MTOW']*(inputs['fltcond|vs']/2 - inputs['V_hover'] + (inputs['fltcond|vs']**2/4 + inputs['V_hover']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors']**2)
        partials['P_addvert', 'fltcond|vs'] = inputs['ac|weights|MTOW']*(0.25*inputs['fltcond|vs']/(inputs['fltcond|vs']**2/4 + inputs['V_hover']**2)**0.5 + 1/2)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'V_hover'] = inputs['ac|weights|MTOW']*(1.0*inputs['V_hover']/(inputs['fltcond|vs']**2/4 + inputs['V_hover']**2)**0.5 - 1)/(550*inputs['ac|propulsion|propeller|num_rotors'])


class VelocityRatio(om.ExplicitComponent):
    """
    Computes the velocity ratio between
    Inputs
    ------
    fltcond|vs : float 
        Vertical speed (vector, ft/s)
    V_hover : float 
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
        self.add_input('fltcond|vs', val = -25, shape = (nn,), units = 'ft/s', desc='Vertical speed')
        self.add_input('V_hover', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')
        
        self.add_output('V_ClimbRatio', shape = (nn,), units=None, desc = 'velocity ratio between climb rate and induced velocity')
        self.declare_partials(['V_ClimbRatio'], ['fltcond|vs'], rows=arange, cols=arange)
        self.declare_partials(['V_ClimbRatio'], ['V_hover'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print('fltcond|vs = :',inputs['fltcond|vs'])
        #print('V_hover = :',inputs['V_hover'])
        outputs['V_ClimbRatio'] = inputs['fltcond|vs']/inputs['V_hover']
        
    def compute_partials(self, inputs, partials):
        partials['V_ClimbRatio', 'fltcond|vs'] = 1/inputs['V_hover']
        partials['V_ClimbRatio', 'V_hover'] = -inputs['fltcond|vs']*inputs['V_hover']**(-2)

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
                outputs['P_vert'] = inputs['P_Hover']*(0.974-0.125*inputs['V_ClimbRatio']-1.372*inputs['V_ClimbRatio']**2-1.718*inputs['V_ClimbRatio']**3-0.655*inputs['V_ClimbRatio']**4)
        #print('V_ClimbRatio = :',inputs['V_ClimbRatio'])
        #print('P_vert = :',outputs['P_vert'])
    def compute_partials(self, inputs, partials):
        for ii in range(len(inputs['V_ClimbRatio'])):
            if inputs['V_ClimbRatio'][ii] >= 0:
                #partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 + 0.25 * (inputs['V_ClimbRatio']**2 + 4)**(-0.5) * 2 * inputs['V_ClimbRatio'])
                partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 + 0.25 * inputs['V_ClimbRatio'] * (0.25*inputs['V_ClimbRatio']**2 + 1)**(-0.5))
                partials['P_vert', 'P_Hover'] = 0.5 * inputs['V_ClimbRatio'] + np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 + 1)
            elif inputs['V_ClimbRatio'][ii] <= -2:
                #partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 - 0.25 * (inputs['V_ClimbRatio']**2 - 4)**(-0.5) * 2 * inputs['V_ClimbRatio'])
                partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 - 0.25 * inputs['V_ClimbRatio'] * (0.25*inputs['V_ClimbRatio']**2 - 1)**(-0.5))
                partials['P_vert', 'P_Hover'] = 0.5 * inputs['V_ClimbRatio'] - np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 - 1)
            else:
                partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(-0.125 - 2*1.372*inputs['V_ClimbRatio'] - 3*1.718*inputs['V_ClimbRatio']**2 - 4*0.655*inputs['V_ClimbRatio']**3)
                partials['P_vert', 'P_Hover'] = 0.974-0.125*inputs['V_ClimbRatio']-1.372*inputs['V_ClimbRatio']**2-1.718*inputs['V_ClimbRatio']**3-0.655*inputs['V_ClimbRatio']**4

class Throttle(om.ExplicitComponent):
    """ 
    Compute the throttle needed for the vertical climb and descent
    Inputs
    ------
    P_vert : float
        Power needed for vertical climb and descent for one motor (vector, h.p.)
    ac|propulsion|propeller|num_rotors
        Number of rotors (scalar, dimensionless)
    ac|propulsion|motor|rating
        Design motor rating (vector, hp)
    
    Output
    ------
    throttle : float 
        Power control setting. Should be [0, 1]. (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    eta_m : float
        Motor efficiency (default 0.97, dimensionaless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        self.options.declare('efficiency', default=0.97, desc="Motor efficiency")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('P_vert', shape = (nn,), units=' hp ', desc='Power needed for vertical climb and descent for one motor')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|motor|rating', units='hp', desc='Design motor rating')

        self.add_output('throttle', shape = (nn,), units=None,desc = 'Power control setting')

        self.declare_partials(['throttle'], ['P_vert'], rows=arange, cols=arange)
        self.declare_partials(['throttle'], ['ac|propulsion|motor|rating'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        eta_m = self.options['efficiency']
        outputs['throttle'] = inputs['P_vert'] / (inputs['ac|propulsion|motor|rating'] * eta_m)

    def compute_partials(self, inputs, partials):
        eta_m = self.options['efficiency']
        partials['throttle', 'P_vert'] = 1/(inputs['ac|propulsion|motor|rating'] * eta_m)
        partials['throttle', 'ac|propulsion|motor|rating'] =  - (inputs['P_vert']/(eta_m)) * inputs['ac|propulsion|motor|rating'] ** (-2)

class AdvancedRatio(om.ExplicitComponent):
    """ 
    Compute the rotor advanced ratio, usually denoted as mu
    Inputs
    ------
    fltcond|Ueas : float
        Absolute airspeed, (vector, ft/s)
    proprpm
        Rotor rpm (vector, rpm)
    ac|propulsion|propeller|diameter : float 
        Rotor diameter (scalar, ft)
    aircraftAOA : float 
        Aircraft cruise angle of attack (vector, deg)

    Output
    ------
    mu : float 
        Rotor advanced ratio (vector, dimensionless)

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
        self.add_input('fltcond|Ueas', shape = (nn,), units='ft/s', desc='Absolute airspeed')
        self.add_input('proprpm', shape = (nn,), units='rpm', desc='Rotor rpm')
        self.add_input('ac|propulsion|propeller|diameter', desc='Rotor diameter', units='ft')
        self.add_input('aircraftAOA', shape = (nn,), units=' rad ', desc='Aircraft cruise angle of attack')

        self.add_output('mu', shape = (nn,), units=None,desc = 'Rotor advanced ratio')

        self.declare_partials(['mu'], ['fltcond|Ueas'], rows=arange, cols=arange)
        self.declare_partials(['mu'], ['proprpm'], rows=arange, cols=arange)
        self.declare_partials(['mu'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['mu'], ['aircraftAOA'], rows=arange, cols=arange)
        
        
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        V_tip = inputs['proprpm'] * inputs['ac|propulsion|propeller|diameter'] * np.pi /60
        outputs['mu'] = np.cos(inputs['aircraftAOA'])*inputs['fltcond|Ueas'] / (V_tip)

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        partials['mu', 'fltcond|Ueas'] = np.cos(inputs['aircraftAOA']) * 60 / (inputs['proprpm'] * inputs['ac|propulsion|propeller|diameter'] * np.pi)
        partials['mu', 'proprpm'] = - np.cos(inputs['aircraftAOA']) * inputs['fltcond|Ueas'] * 60 / (inputs['ac|propulsion|propeller|diameter'] * np.pi) * (inputs['proprpm'] ** -2)
        partials['mu', 'ac|propulsion|propeller|diameter'] = - np.cos(inputs['aircraftAOA']) * inputs['fltcond|Ueas'] * 60 / (inputs['proprpm'] * np.pi) * (inputs['ac|propulsion|propeller|diameter'] ** -2)
        partials['mu', 'aircraftAOA'] = -np.sin(inputs['aircraftAOA']) * inputs['fltcond|Ueas'] * 60 / (inputs['proprpm'] * inputs['ac|propulsion|propeller|diameter'] * np.pi)


class ThrustCoef(om.ExplicitComponent):
    """ 
    Compute the hover thrust coefficient
    Inputs
    ------
    ac|weights|MTOW : float
        MTOW (scaler, lb)
    fltcond|rho : float 
        Air density (vector, slug/ft**3)
    proprpm : float
        Rotor rpm (vector, rpm)
    ac|propulsion|propeller|diameter : float
        Rotor diameter (scalar, ft)
    ac|propulsion|propeller|num_rotors : int
        Number of rotor (scaler, dimensionless)
    aircraftAOA : float 
        Aircraft cruise angle of attack (vector, deg)

    Output
    ------
    C_T : float 
       Thrust coefficient (vector, dimensionless)

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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('fltcond|rho', shape = (nn,), units='slug/ft**3', desc='Air desnity')
        self.add_input('proprpm', shape = (nn,), units='rpm', desc='Rotor rpm')
        self.add_input('ac|propulsion|propeller|diameter', desc=' Rotor diameter', units='ft')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('aircraftAOA', shape = (nn,), units=' rad ', desc='Aircraft cruise angle of attack')

        self.add_output('C_T', shape = (nn,), units=None,desc = 'Thrust coefficient')

        self.declare_partials(['C_T'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['C_T'], ['fltcond|rho'], rows=arange, cols=arange)
        self.declare_partials(['C_T'], ['proprpm'], rows=arange, cols=arange)
        self.declare_partials(['C_T'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['C_T'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['C_T'], ['aircraftAOA'], rows=arange, cols=arange)
        
        
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        #single_motor_thrust = (inputs['ac|weights|MTOW'] / (np.cos(inputs['aircraftAOA'])) * inputs['ac|propulsion|propeller|num_rotors'])
        #print(  'np.cos(inputs[aircraftAOA])'  , np.cos(inputs['aircraftAOA']))
        #V_tip = inputs['proprpm'] * inputs['ac|propulsion|propeller|diameter'] * np.pi /60
        #disk_area = (inputs['ac|propulsion|propeller|diameter'] /2) ** 2 * np.pi
        #outputs['C_T'] = single_motor_thrust / ( disk_area * inputs['fltcond|rho'] * V_tip ** 2) 
        outputs['C_T'] = inputs['ac|weights|MTOW'] / (inputs['ac|propulsion|propeller|num_rotors']*inputs['fltcond|rho']*inputs['ac|propulsion|propeller|diameter']**3*inputs['proprpm']*15*np.cos(inputs['aircraftAOA']) )

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        disk_area = (inputs['ac|propulsion|propeller|diameter'] /2) ** 2 * np.pi
        """
        partials['C_T', 'ac|weights|MTOW'] = 0.0212206590789194/(inputs['ac|propulsion|propeller|num_rotors']*inputs['ac|propulsion|propeller|diameter']**3*inputs['fltcond|rho']*inputs['proprpm']*np.cos(inputs['aircraftAOA']))
        partials['C_T', 'fltcond|rho'] = -0.0212206590789194*inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['ac|propulsion|propeller|diameter']**3*inputs['fltcond|rho']**2**inputs['proprpm']*np.cos(inputs['aircraftAOA']))
        partials['C_T', 'proprpm'] = -0.0212206590789194*inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['ac|propulsion|propeller|diameter']**3*inputs['fltcond|rho']*inputs['proprpm']**2*np.cos(inputs['aircraftAOA']))
        partials['C_T', 'ac|propulsion|propeller|diameter'] = -0.0636619772367581*inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']*inputs['ac|propulsion|propeller|diameter']**4*inputs['fltcond|rho']*inputs['proprpm']*np.cos(inputs['aircraftAOA']))
        partials['C_T', 'ac|propulsion|propeller|num_rotors'] = -0.0212206590789194*inputs['ac|weights|MTOW']/(inputs['ac|propulsion|propeller|num_rotors']**2*inputs['ac|propulsion|propeller|diameter']**3*inputs['fltcond|rho']*inputs['proprpm']*np.cos(inputs['aircraftAOA']))
        partials['C_T', 'aircraftAOA'] = 0.0212206590789194*inputs['ac|weights|MTOW']*np.sin(inputs['aircraftAOA'])/(inputs['ac|propulsion|propeller|num_rotors']*inputs['ac|propulsion|propeller|diameter']**3*inputs['fltcond|rho']*inputs['proprpm']*np.cos(inputs['aircraftAOA'])**2)
        """
        
        partials['C_T', 'ac|weights|MTOW'] = 1 / ( inputs['ac|propulsion|propeller|num_rotors'] * inputs['fltcond|rho'] * inputs['ac|propulsion|propeller|diameter']**3 
                                            * inputs['proprpm'] * 15 * np.cos(inputs['aircraftAOA']) )
        
        partials['C_T', 'fltcond|rho'] = - (inputs['ac|weights|MTOW']/ (inputs['ac|propulsion|propeller|num_rotors'] * inputs['ac|propulsion|propeller|diameter']**3 
                                            * inputs['proprpm'] * 15 * np.cos(inputs['aircraftAOA']) )) * inputs['fltcond|rho'] ** (-2)
        
        partials['C_T', 'proprpm'] = - (inputs['ac|weights|MTOW']/ (inputs['ac|propulsion|propeller|num_rotors'] * inputs['ac|propulsion|propeller|diameter']**3 
                                            * inputs['fltcond|rho'] * 15 * np.cos(inputs['aircraftAOA']) )) * inputs['proprpm'] ** (-2)
        
        partials['C_T', 'ac|propulsion|propeller|diameter'] = -3 * ( inputs['ac|weights|MTOW'] / ( inputs['ac|propulsion|propeller|num_rotors']  * inputs['fltcond|rho'] * inputs['proprpm'] 
                                            * 15 * np.cos(inputs['aircraftAOA']))) * inputs['ac|propulsion|propeller|diameter'] ** (-4)

        partials['C_T', 'ac|propulsion|propeller|num_rotors'] = - ( inputs['ac|weights|MTOW']/ (inputs['ac|propulsion|propeller|diameter']**3 * inputs['fltcond|rho'] * inputs['proprpm'] 
                                            * 15 * np.cos(inputs['aircraftAOA']) )) * inputs['ac|propulsion|propeller|num_rotors'] ** (-2)

        partials['C_T', 'aircraftAOA'] = ( inputs['ac|weights|MTOW']/ ( inputs['ac|propulsion|propeller|num_rotors'] * inputs['ac|propulsion|propeller|diameter']**3 * inputs['fltcond|rho'] 
                                            * 15 * inputs['proprpm']  ) ) * np.cos(inputs['aircraftAOA']) ** (-2) * np.sin(inputs['aircraftAOA'])
        

class HoverInflowRatio(om.ExplicitComponent):
    """ 
    Compute the hover inflow ratio during forward flight 
    Inputs
    ------
    C_T : float
        Thrust coefficient, (vector, dimensionless)

    Output
    ------
    lambda_h : float 
        Hover inflow ratio (vector, dimensionless)

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

        self.add_input('C_T', shape = (nn,), units= None, desc='Thrust coefficient')
        self.add_output('lambda_h', shape = (nn,), units=None,desc = ' Hover inflow ratio')

        self.declare_partials(['lambda_h'], ['C_T'], rows=arange, cols=arange)
        
    def compute(self, inputs, outputs):
        outputs['lambda_h'] = (inputs['C_T']/2) ** 0.5

    def compute_partials(self, inputs, partials):
        partials['lambda_h', 'C_T'] = 0.5 * (inputs['C_T']/2) ** (-0.5) * 0.5
       

class InflowRatio(om.ExplicitComponent):
    """ 
    Compute the current inflow ratio during forward flight 
    Inputs
    ------
    lambda_h : float
        Hover inflow ratio (vector, dimensionless)
    mu : float 
        Rotor advanced ratio (vector, dimensionless)

    Output
    ------
    lambda : float 
        Current inflow ratio (vector, dimensionless)

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

        self.add_input('lambda_h', shape = (nn,), units= None, desc='Hover inflow ratio ')
        self.add_input('mu', shape = (nn,), units= None, desc='Rotor advanced ratio')

        self.add_output('lambda', shape = (nn,), units=None,desc = 'current inflow ratio')

        self.declare_partials(['lambda'], ['lambda_h'], rows=arange, cols=arange)
        self.declare_partials(['lambda'], ['mu'], rows=arange, cols=arange)
        
        
    def compute(self, inputs, outputs):
        outputs['lambda'] = (0.05 * (inputs['mu']/inputs['lambda_h']) + 0.3) * inputs['lambda_h']

    def compute_partials(self, inputs, partials):
        partials['lambda', 'lambda_h'] = 0.05 * inputs['mu'] + 0.3 
        partials['lambda', 'mu'] = 0.05 * inputs['lambda_h']

class InflowRatio_implicit(om.ImplicitComponent):
    """ 
    Compute the current inflow ratio during forward flight 
    Inputs
    ------

    aircraftAOA : float 
        Aircraft cruise angle of attack (vector, deg)

    mu : float 
        Rotor advanced ratio (vector, dimensionless)

    C_T : float
        Thrust coefficient, (vector, dimensionless)

    Output
    ------
    lambda : float 
        Current inflow ratio (vector, dimensionless)

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
        
        self.add_input('aircraftAOA', shape = (nn,), units=' rad ', desc='Aircraft cruise angle of attack')
        self.add_input('mu', shape = (nn,), units= None, desc='Rotor advanced ratio')
        self.add_input('C_T', shape = (nn,), units= None, desc='Thrust coefficient')
        

        self.add_output('lambda', shape = (nn,), units=None,desc = 'current inflow ratio')

        self.declare_partials(['lambda'], ['aircraftAOA'], rows=arange, cols=arange)
        self.declare_partials(['lambda'], ['mu'], rows=arange, cols=arange)
        self.declare_partials(['lambda'], ['C_T'], rows=arange, cols=arange)
        self.declare_partials(['lambda'], ['lambda'], rows=arange, cols=arange)
        
    def apply_nonlinear(self, inputs, outputs, residuals):
        mu = inputs['mu']
        AOA = inputs['aircraftAOA']
        C_T = inputs['C_T']
        solv_lambda = outputs['lambda']
        residuals['lambda'] = solv_lambda - mu*np.tan(AOA) - C_T/(2 * np.sqrt(mu**2 + solv_lambda**2))
    
    def guess_nonlinear(self, inputs, outputs, resids):
        # Check residuals
        if np.any(np.abs(resids['lambda'])) > 1.0E-2:
            outputs['lambda'] = 1.0
    
    def linearize(self, inputs, outputs, partials):
        mu = inputs['mu']
        AOA = inputs['aircraftAOA']
        C_T = inputs['C_T']
        solv_lambda = outputs['lambda']

        partials['lambda', 'mu'] = -np.tan(AOA) - C_T/2 * (-1/2) * (mu**2 + solv_lambda**2)**(-3/2) * 2 * mu
        partials['lambda', 'aircraftAOA'] = - mu*(np.cos(inputs['aircraftAOA'])**(-2))
        partials['lambda', 'C_T'] = - 0.5*(mu**2+solv_lambda**2)**(-0.5)
        partials['lambda', 'lambda'] = 1 - C_T/2 * (-0.5) * (mu**2 + solv_lambda**2)**(-3/2) * 2 * solv_lambda
    

class CruiserPower(om.ExplicitComponent):
    """ 
    Compute the current inflow ratio during forward flight 
    Inputs
    ------
    lambda_h : float
        Hover inflow ratio (vector, dimensionless)
    lambda : float
        Current inflow ratio (vector, dimensionless)
    P_Hover : float
        Ideal power, P_Ideal (h.p.) = thrust * sqrt(diskload) / 38, (vector, h.p.)
        P_Hover = P_Ideal / FM

    Output
    ------
    P_cruise : float 
        Single rotor power requried in straight level cruise (vector, h.p.)

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

        self.add_input('lambda_h', shape = (nn,), units= None, desc='Hover inflow ratio ')
        self.add_input('lambda', shape = (nn,), units= None, desc='Current inflow ratio')
        self.add_input('P_Hover', shape = (nn,), units = 'hp', desc='Ideal power')
        
        self.add_output('P_cruise', shape = (nn,), units='hp',desc = 'Single rotor power requried in straight level cruise')

        self.declare_partials(['P_cruise'], ['lambda_h'], rows=arange, cols=arange)
        self.declare_partials(['P_cruise'], ['lambda'], rows=arange, cols=arange)
        self.declare_partials(['P_cruise'], ['P_Hover'], rows=arange, cols=arange)
        
    def compute(self, inputs, outputs):
        outputs['P_cruise'] = inputs['P_Hover'] * inputs['lambda']/inputs['lambda_h']

    def compute_partials(self, inputs, partials):
        partials['P_cruise', 'lambda_h'] = - inputs['P_Hover'] * inputs['lambda'] * inputs['lambda_h'] ** (-2)
        partials['P_cruise', 'lambda'] = inputs['P_Hover'] /inputs['lambda_h']
        partials['P_cruise', 'P_Hover'] = inputs['lambda']/inputs['lambda_h']

