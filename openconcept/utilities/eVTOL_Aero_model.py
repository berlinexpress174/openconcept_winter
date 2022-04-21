import numpy as np
import openmdao.api as om
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties

class BodyDrag(om.ExplicitComponent):
    """
    Calculates drag force based on CD, dynamic pressure, and body reference area
    Inputs
    ------
    ac|aero|CD0_body : float
        body drag coefficient (scaler, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|body|S_ref : float
        Reference area of body (scalar, m**2)
    Outputs
    -------
    drag : float
        drag force (vector, Newtons)
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
        self.add_input('ac|aero|CD0_body')
        self.add_input('fltcond|q', units='N * m**-2', shape=(nn,))
        self.add_input('ac|geom|body|S_ref', units='m **2')

        self.add_output('drag', units='N', shape=(nn,))
        self.declare_partials(['drag'], ['fltcond|q'], rows=arange, cols=arange)
        self.declare_partials(['drag'], ['ac|aero|CD0_body', 'ac|geom|body|S_ref'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['drag'] = inputs['fltcond|q'] * inputs['ac|geom|body|S_ref'] * inputs['ac|aero|CD0_body']

    def compute_partials(self, inputs, J):
        J['drag', 'fltcond|q'] = inputs['ac|geom|body|S_ref'] * inputs['ac|aero|CD0_body']
        J['drag', 'ac|aero|CD0_body'] = inputs['fltcond|q'] * inputs['ac|geom|body|S_ref']
        J['drag', 'ac|geom|body|S_ref'] = inputs['fltcond|q'] * inputs['ac|aero|CD0_body']


class LiftDragGivenAoA(om.Group):
    """
    Calculates lift and drag force

    Inputs
    ------
    fltcond|alpha : float
        Angle of attack (vector, rad)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)
    lift : float
        Lift force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('coeff', ClCdModel(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('force', LiftDrag(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

class ClCdModel(om.ExplicitComponent):
    # computes CL and CD given the angle of attack
    # TODO: replace with Tangler-Ostowari model. See Chauhan 2018/20.
    # TODO: CD0 changes depending on the body attitude

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('fltcond|alpha', units='rad', shape=(nn,))
        self.add_output('fltcond|CL', shape=(nn,))
        self.add_output('fltcond|CD', shape=(nn,))
        self.declare_partials('*', '*', rows=arange, cols=arange)
    
    def compute(self, inputs, outputs):
        alpha = inputs['fltcond|alpha']

        Cd = 1.5 / (np.pi / 2)**2 * alpha**2 + 0.05

        stall_AoA = np.deg2rad(15)
        Cl_max = 1.2

        Cl1 = Cl_max / stall_AoA * alpha   # for AoA less than stall_AoA (where stall occurs)
        Cl2 = Cl_max / (np.pi / 2 - stall_AoA) * (np.pi / 2 - alpha)   # for AoA > stall_AoA
        Cl3 = -Cl_max / (-np.pi / 2 + stall_AoA) * (-np.pi / 2 - alpha)   # for AoA < -stall_AoA

        # idx1 = np.abs(alpha) <= stall_AoA
        idx2 = alpha > stall_AoA
        idx3 = alpha < -stall_AoA
        Cl1[idx2] = Cl2[idx2]
        Cl1[idx3] = Cl3[idx3]

        outputs['fltcond|CL'] = Cl1
        outputs['fltcond|CD'] = Cd

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        alpha = inputs['fltcond|alpha']

        stall_AoA = np.deg2rad(15)
        Cl_max = 1.2

        dCl1_da = Cl_max / stall_AoA * np.ones(nn)
        dCl2_da = -Cl_max / (np.pi / 2 - stall_AoA) * np.ones(nn)
        dCl3_da = -Cl_max / (np.pi / 2 - stall_AoA) * np.ones(nn)

        idx2 = alpha > stall_AoA
        idx3 = alpha < -stall_AoA
        dCl1_da[idx2] = dCl2_da[idx2]
        dCl1_da[idx3] = dCl3_da[idx3]

        partials['fltcond|CD', 'fltcond|alpha'] = 1.5 / (np.pi / 2)**2 * 2 * alpha
        partials['fltcond|CL', 'fltcond|alpha'] = dCl1_da


class LiftDrag(om.ExplicitComponent):
    """
    Calculates lift and drag force

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|CD : float
        Lift coefficient (vector, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)
    lift : float
        Lift force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    # computes the lift and drag given the angle of attack
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('fltcond|CL', shape=(nn,))
        self.add_input('fltcond|CD', shape=(nn,))
        self.add_input('fltcond|q', units='N * m**-2', shape=(nn,))
        self.add_input('ac|geom|wing|S_ref', units='m **2')
        self.add_output('lift', units='N', shape=(nn,))
        self.add_output('drag', units='N', shape=(nn,))
        self.declare_partials('lift', ['fltcond|CL', 'fltcond|q'], rows=arange, cols=arange)
        self.declare_partials('drag', ['fltcond|CD', 'fltcond|q'], rows=arange, cols=arange)
        self.declare_partials(['lift', 'drag'], 'ac|geom|wing|S_ref', rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['lift'] = inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] * inputs['fltcond|CL']
        outputs['drag'] = inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] * inputs['fltcond|CD']

    def compute_partials(self, inputs, partials):
        q = inputs['fltcond|q']
        S = inputs['ac|geom|wing|S_ref']
        Cl = inputs['fltcond|CL']
        Cd = inputs['fltcond|CD']
        
        partials['lift', 'fltcond|CL'] = q * S
        partials['lift', 'fltcond|q'] = S * Cl
        partials['lift', 'ac|geom|wing|S_ref'] = q * Cl
        partials['drag', 'fltcond|CD'] = q * S
        partials['drag', 'fltcond|q'] = S * Cd
        partials['drag', 'ac|geom|wing|S_ref'] = q * Cd


class MaxRPM(om.ExplicitComponent):
    """
    Calculates the maximum propeller rpm, subjected to the Vtip = 0.8 Mach number.
    Inputs
    ------
    ac|propulsion|propeller|diameter : float 
        Propeller diameter (scaler, m)

    ac|propulsion|propeller|proprpm : float 
        Propeller rpm (scaler, rpm)

    nr : float 
        Number of propeller (scaler, dimensionless)
    
    Outputs
    -------
    maxproprpm : float
        Maximum prop rpm (vector, rpm)
    
    proprpm : float
        Propeller rpm for all mission segments (vector, rpm)

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
        # At NTP condition, Temperature is 298.15K , R = 287, specific heat = 1.4 for air
        self.add_input('ac|propulsion|propeller|diameter', units='m')
        self.add_input('ac|propulsion|propeller|proprpm', units='rpm')
        self.add_input('fltcond|T',units='K', shape=(nn,), desc='Temperature')
        self.add_output('maxproprpm', units='rpm', shape=(nn,))
        self.add_output('proprpm', units='rpm', shape=(nn,))
        self.declare_partials(['maxproprpm'], ['ac|propulsion|propeller|diameter'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['maxproprpm'], ['ac|propulsion|propeller|proprpm'], rows=arange, cols=np.zeros(nn))
        self.declare_partials(['maxproprpm'], ['fltcond|T'], rows=arange, cols=arange)
        self.declare_partials(['proprpm'], ['ac|propulsion|propeller|proprpm'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        gamma = 1.4
        R = 287
        M = 0.55
        T = inputs['fltcond|T']
        a = np.sqrt(gamma * R * T) # Speed of sound 
        V_tip = M * a  #( units: m/s)
        Max_rpm = V_tip * 60 / (inputs['ac|propulsion|propeller|diameter'] * np.pi)
        outputs['maxproprpm'] = Max_rpm
        outputs['proprpm'] = inputs['ac|propulsion|propeller|proprpm']

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        gamma = 1.4
        R = 287
        M = 0.55
        T = inputs['fltcond|T']
        a = np.sqrt(gamma * R * T) # Speed of sound 
        V_tip = M * a  #( units: m/s)
        J['maxproprpm', 'ac|propulsion|propeller|diameter'] = - (60 * V_tip) / (inputs['ac|propulsion|propeller|diameter'])**2 * (1 /np.pi)
        J['maxproprpm', 'fltcond|T'] = (M/(inputs['ac|propulsion|propeller|diameter']*np.pi)) * np.sqrt(gamma*R) * (1/2) * inputs['fltcond|T'] ** (-0.5)
        J['proprpm','ac|propulsion|propeller|proprpm'] = 1

class RpmResidual(om.ExplicitComponent):
    """
    Calculates the propeller/ rotor residual
    Inputs
    ------
    maxproprpm : float
        Maximum prop rpm (vector, rpm).

    proprpm : float
        Propeller rpm for all mission segments (vector, rpm).

    nr : float 
        Number of propeller (scaler, dimensionless)
    
    Outputs
    -------
    prop_residual : float
         (scaler, rpm)
    
    proprpm : float
        Propeller rpm for all mission segments.

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
        self.add_input('maxproprpm', units='rpm', shape=(nn,))
        self.add_input('proprpm', units='rpm', shape=(nn,))
        self.add_output('proprpm_residual', units='rpm', shape=(nn,))
        self.declare_partials(['proprpm_residual'], ['maxproprpm'], rows=arange, cols=arange)
        self.declare_partials(['proprpm_residual'], ['proprpm'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        outputs['proprpm_residual'] = inputs['maxproprpm'] - inputs['proprpm']

    def compute_partials(self, inputs, J):
        J['proprpm_residual','maxproprpm'] = 1
        J['proprpm_residual','proprpm'] = -1