import numpy as np
import openmdao.api as om
from openconcept.analysis.atmospherics.density_comp import DensityComp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties

class SeaLevelTakeoffPower(om.ExplicitComponent):
    """
    Helicoptor performance and control. P8
    computes takeoff Power P_TO (h.p.) for the whole aircraft
    P_TO (h.p.) = thrust * sqrt(DL) / rho 
    Thrust(lbf): Takeoff thrust needed for the whole aircraft
    DL: Single Rotor disk load (lb/ft**2)
    rho: Air density
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        self.options.declare('power_factor', default = 1) # Maximum thrust factor for thrust.
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('ac|weights|MTOW', units='kg', desc='MTOW')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('diskload', units='lbf/ft**2', desc = 'Disk load per per propeller')
        self.add_output('P_TO', units='hp',desc = 'IdealHoverPower')
        self.declare_partials(['*'], ['*'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # 1 N = 0.2248089431 lbf
        power_factor = self.options['power_factor']
        outputs['P_TO'] = (((inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors']) * 9.81 * 0.2248089431 ) * np.sqrt(inputs['diskload']) / 38) * power_factor
        
    def compute_partials(self, inputs, partials):
        power_factor = self.options['power_factor']
        partials['P_TO', 'ac|weights|MTOW'] = power_factor * 0.0580362034687105*inputs['diskload']**0.5 / inputs['ac|propulsion|propeller|num_rotors']
        partials['P_TO', 'ac|propulsion|propeller|num_rotors'] = power_factor * -0.0580362034687105*inputs['ac|weights|MTOW']*inputs['diskload']**0.5 / inputs['ac|propulsion|propeller|num_rotors']**2
        partials['P_TO', 'diskload'] = power_factor * 0.0290181017343553*inputs['ac|weights|MTOW']*inputs['diskload']**(-0.5)/inputs['ac|propulsion|propeller|num_rotors']

class SeaLevelTakeoffPower2(om.ExplicitComponent):
    # Helicoptor performance and control. P8
    # computes takeoff Power P_TO (h.p.) for the whole aircraft
    # P_TO (h.p.) = thrust * sqrt(DL) / rho 
    # Thrust(lbf): Takeoff thrust needed for the whole aircraft
    # DL: Single Rotor disk load (lb/ft**2)
    # rho: Air density

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")
        self.options.declare('power_factor', default = 1) # Maximum thrust factor for thrust.
        
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('ac|weights|MTOW', units='kg', desc='MTOW')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('bodyS_ref', units = 'm**2', desc = 'projected vertical area' )
        self.add_input('diskload', units='N/m**2', desc = 'Disk load per per propeller')
        self.add_output('P_TO', units='kW',desc = 'IdealHoverPower per motor')
        self.declare_partials(['*'], ['*'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # Accuate Thrust cal in N
        #Helicoptor performance and control. P8
        # 1N = 0.2248 lbf
        power_factor = self.options['power_factor']
        D_v_in_SI = 0.225 * ( 0.3 * (inputs['diskload'] * inputs['ac|propulsion|propeller|num_rotors'] ) * inputs['bodyS_ref'] ) # units = N
        T_hover_inN = ((inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors']) * 9.81) #+ D_v_in_SI 
        print('----D_v_in_SI-----',D_v_in_SI)
        print('----T_hover_inN-----',T_hover_inN)
        outputs['P_TO'] = power_factor * T_hover_inN * np.sqrt(inputs['diskload']) / 1551.7 #W  

    def compute_partials(self, inputs, partials):
        a = inputs['ac|weights|MTOW']
        b = inputs['bodyS_ref']
        c = inputs['diskload']

        #TODO correct partials 
        partials['P_TO', 'ac|weights|MTOW'] = a * c ** 0.5 * (1 + 0.3 * b * c/ a)/19 - 0.00789473684210526 * b * c ** 1.5
        partials['P_TO', 'bodyS_ref'] = 0.00789473684210526 * 0.00789473684210526 * b * c ** 1.5
        partials['P_TO', 'diskload'] = 0.0131578947368421 * a ** 2 * c ** (-0.5) * (1 + 0.3 * b * c/ a) + 0.00789473684210526 * b * c ** 1.5



class CompBody_S_wet(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('ac|weights|MTOW', units='kg', desc='MTOW')
        self.add_output('Body_S_wet', units='ft**2',desc = 'fusalge wetted area')
        self.declare_partials(['*'], ['*'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['Body_S_wet'] = 10**(0.8635) * inputs['ac|weights|MTOW']**(0.5632)
        
    def compute_partials(self, inputs, partials):
        # Compute from sympy. Power factor 1.2 has been considered
        partials['Body_S_wet', 'ac|weights|MTOW'] = (10**(0.8635) * inputs['ac|weights|MTOW']**(0.5632-1)) * 0.5632

class TotalPowerAndThrustVert(om.ExplicitComponent):
    """ 
    Compute the thrust needed for the vertical climb and descent
    Inputs
    ------
    P_vert : float
        Power needed for vertical climb and descent (vector, h.p.)
    FM : float
        Figure of Merit (scalar, dimensionless)
    fltcond|vs : float 
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
        self.add_input('FM', units=None, desc='Figure of Merit')
        self.add_input('fltcond|vs', shape = (nn,), units='ft/s', desc='Vertical speed')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        #self.add_input('prop_eta', shape = (nn,), units = None, desc='propelelr efficiency')

        self.add_output('thrust_total', shape = (nn,), units='N',desc = 'Total thrust generated from all rotor')
        self.add_output('power', shape = (nn,), units='hp',desc = 'Power needed from all motor')

        self.declare_partials(['thrust_total'], ['P_vert'], rows=arange, cols=arange)
        self.declare_partials(['thrust_total'], ['fltcond|vs'], rows=arange, cols=arange)
        self.declare_partials(['thrust_total'], ['FM'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['thrust_total'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        #self.declare_partials(['thrust_total'], ['prop_eta'], rows=arange, cols=arange)

        self.declare_partials(['power'], ['P_vert'], rows=arange, cols=arange)
        self.declare_partials(['power'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['thrust_total'] = inputs['ac|propulsion|propeller|num_rotors'] * (inputs['P_vert'] * inputs['FM'] / inputs['fltcond|vs']) #* inputs['prop_eta']
        outputs['power'] = inputs['ac|propulsion|propeller|num_rotors'] * inputs['P_vert']

    def compute_partials(self, inputs, partials):
        partials['thrust_total', 'P_vert'] = inputs['ac|propulsion|propeller|num_rotors'] * inputs['FM'] / inputs['fltcond|vs']
        partials['thrust_total', 'fltcond|vs'] = inputs['ac|propulsion|propeller|num_rotors'] * (- inputs['P_vert'] * inputs['FM'] * inputs['fltcond|vs']**(-2))
        partials['thrust_total', 'FM'] = inputs['ac|propulsion|propeller|num_rotors'] * (inputs['P_vert'] * inputs['fltcond|vs'])
        partials['thrust_total', 'ac|propulsion|propeller|num_rotors'] = inputs['P_vert'] * inputs['FM'] / inputs['fltcond|vs']
        partials['thrust_total', 'prop_eta'] = inputs['ac|propulsion|propeller|num_rotors'] * (inputs['P_vert'] * inputs['FM'] / inputs['fltcond|vs'])
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
        self.add_input('thrust', val = 4000, shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', desc='Number_of_rotor')
        self.add_input('ac|propulsion|propeller|diameter', units='ft', desc='Rotor diameter')
        self.add_output('diskload', shape = (nn,), units='lbf/ft**2',desc = 'Disk load per per propeller')
        self.declare_partials(['*'], ['*'], rows=arange, cols=arange)
        
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
        self.add_input('thrust', val = 4000, shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('diskload', shape = (nn,), units='lbf/ft**2', desc = 'Disk load per rotor')
        self.add_input('FM', units=None, desc = 'Figure of merit')
        
        self.add_output('P_Hover', units='hp',desc = 'Ideal hover power')
        self.declare_partials(['*'], ['*'], rows=arange, cols=arange)

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
        self.add_input('ac|weights|MTOW', units='lb', desc='MTOW')
        self.add_input('thrust', val = 4000, shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number of rotor')
        self.add_input('ac|propulsion|propeller|diameter', units = 'ft', desc='Rotor diameter')
        self.add_input('fltcond|rho', val = 0.0023777 ,shape=(nn,), units='slug/ft**3', desc = 'air density')
        
        self.add_output('V_induced', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')
        self.declare_partials(['V_induced'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
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
    fltcond|vs : float 
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
        self.add_input('thrust', val = 4000, shape = (nn,), units='lbf', desc='single motor thrust')
        self.add_input('ac|propulsion|propeller|num_rotors', units = None, desc='Number_of_rotor')
        self.add_input('fltcond|vs', val = -25, shape = (nn,), units='ft/s', desc = 'climb rate')
        self.add_input('V_induced', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')

        self.add_output('P_addvert', shape = (nn,), units='hp',desc = 'Additional power required for climbing in a given vertical climb rate')
        self.declare_partials(['P_addvert'], ['ac|weights|MTOW'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_addvert'], ['ac|propulsion|propeller|num_rotors'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['P_addvert'], ['fltcond|vs'], rows=arange, cols=arange)
        self.declare_partials(['P_addvert'], ['V_induced'], rows=arange, cols=arange)
        self.declare_partials(['P_addvert'], ['thrust'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        #print('Vs = ',inputs['fltcond|vs'])
        nn = self.options['num_nodes']
        if inputs['fltcond|vs'] > 0: # vertical climb
            A = (inputs['fltcond|vs']/2 + np.sqrt((-inputs['fltcond|vs']/2)**2 + inputs['V_induced']**2 ) - inputs['V_induced'] )
        else: # vertical descent
            A = (inputs['fltcond|vs']/2 - np.sqrt((inputs['fltcond|vs']/2)**2 - inputs['V_induced']**2 ) - inputs['V_induced'] )
        outputs['P_addvert'] = ((inputs['ac|weights|MTOW']/inputs['ac|propulsion|propeller|num_rotors'])/550) * A
        
    def compute_partials(self, inputs, partials):
        #partials['P_addvert', 'ac|weights|MTOW'] = (inputs['fltcond|vs']/2 - inputs['V_induced'] + (inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        #partials['P_addvert', 'ac|propulsion|propeller|num_rotors'] = -inputs['ac|weights|MTOW']*(inputs['fltcond|vs']/2 - inputs['V_induced'] + (inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors']**2)
        #partials['P_addvert', 'fltcond|vs'] = inputs['ac|weights|MTOW']*(0.25*inputs['fltcond|vs']/(inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5 + 1/2)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        #partials['P_addvert', 'V_induced'] = inputs['ac|weights|MTOW']*(1.0*inputs['V_induced']/(inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5 - 1)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'thrust'] = (inputs['fltcond|vs']/2 - inputs['V_induced'] + (inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'ac|propulsion|propeller|num_rotors'] = -inputs['thrust']*(inputs['fltcond|vs']/2 - inputs['V_induced'] + (inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5)/(550*inputs['ac|propulsion|propeller|num_rotors']**2)
        partials['P_addvert', 'fltcond|vs'] = inputs['thrust']*(0.25*inputs['fltcond|vs']/(inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5 + 1/2)/(550*inputs['ac|propulsion|propeller|num_rotors'])
        partials['P_addvert', 'V_induced'] = inputs['thrust']*(1.0*inputs['V_induced']/(inputs['fltcond|vs']**2/4 + inputs['V_induced']**2)**0.5 - 1)/(550*inputs['ac|propulsion|propeller|num_rotors'])


class VelocityRatio(om.ExplicitComponent):
    """
    Computes the velocity ratio between
    Inputs
    ------
    fltcond|vs : float 
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
        self.add_input('fltcond|vs', val = -25, shape = (nn,), units = 'ft/s', desc='Vertical speed')
        self.add_input('V_induced', shape = (nn,), units='ft/s', desc = 'Rotor induced speed')
        
        self.add_output('V_ClimbRatio', shape = (nn,), units=None, desc = 'velocity ratio between climb rate and induced velocity')
        self.declare_partials(['V_ClimbRatio'], ['fltcond|vs'], rows=arange, cols=arange)
        self.declare_partials(['V_ClimbRatio'], ['V_induced'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        print(inputs['V_induced'])
        outputs['V_ClimbRatio'] = inputs['fltcond|vs']/inputs['V_induced']
        
    def compute_partials(self, inputs, partials):
        partials['V_ClimbRatio', 'fltcond|vs'] = 1/inputs['V_induced']
        partials['V_ClimbRatio', 'V_induced'] = -inputs['fltcond|vs']*inputs['V_induced']**(-2)

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
        self.add_input('P_Hover', units='hp',desc = 'Ideal hover power')
        
        self.add_output('P_vert', shape = (nn,), units='hp', desc = 'Power needed for vertical climb and descent')
        self.declare_partials(['P_vert'], ['V_ClimbRatio'], rows=arange, cols=arange)
        self.declare_partials(['P_vert'], ['P_Hover'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        print
        if inputs['V_ClimbRatio'] >= 0:
            outputs['P_vert'] = inputs['P_Hover']*(0.5 * inputs['V_ClimbRatio'] + np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 + 1))
        elif inputs['V_ClimbRatio'] <= -2:
            outputs['P_vert'] = inputs['P_Hover']*(0.5 * inputs['V_ClimbRatio'] - np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 - 1))
        else:  
            outputs['P_vert'] = inputs['P_Hover']*(0.974-0.125*inputs['V_ClimbRatio']-1.372*inputs['V_ClimbRatio']**2-1.718*inputs['V_ClimbRatio']**3-0.665*inputs['V_ClimbRatio']**4)

    def compute_partials(self, inputs, partials):
        if inputs['V_ClimbRatio'] > 0:
            partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 + 0.25 * (inputs['V_ClimbRatio']**2 + 4)**(-0.5) * 2 * inputs['V_ClimbRatio'])
            partials['P_vert', 'P_Hover'] = 0.5 * inputs['V_ClimbRatio'] + np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 + 1)
        elif inputs['V_ClimbRatio'] < -2:
            partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(0.5 - 0.25 * (inputs['V_ClimbRatio']**2 - 4)**(-0.5) * 2 * inputs['V_ClimbRatio'])
            partials['P_vert', 'P_Hover'] = 0.5 * inputs['V_ClimbRatio'] - np.sqrt(0.25 * (inputs['V_ClimbRatio'])**2 - 1)
        else:
            partials['P_vert', 'V_ClimbRatio'] = inputs['P_Hover']*(-0.125-2*1.372*inputs['V_ClimbRatio']-3*1.718*inputs['V_ClimbRatio']**2-4*0.665*inputs['V_ClimbRatio']**3)
            partials['P_vert', 'P_Hover'] = 0.974-0.125*inputs['V_ClimbRatio']-1.372*inputs['V_ClimbRatio']**2-1.718*inputs['V_ClimbRatio']**3-0.665*inputs['V_ClimbRatio']**4
