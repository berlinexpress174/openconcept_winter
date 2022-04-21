import numpy as np
import openmdao.api as om

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