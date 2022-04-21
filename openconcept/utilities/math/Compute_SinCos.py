import numpy as np
import openmdao.api as om

class ComputeSinCos(om.ExplicitComponent):
    # computes sin and cos of the input angle

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('angle', shape=(nn,),units='deg')
        self.add_output('cos', shape=(nn,),units=None)
        self.add_output('sin', shape=(nn,),units=None)
        self.declare_partials(['*'], 'angle', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['cos'] = np.cos(inputs['angle'])
        outputs['sin'] = np.sin(inputs['angle'])

    def compute_partials(self, inputs, partials):
        partials['cos', 'angle'] = -np.sin(inputs['angle'])
        partials['sin', 'angle'] = np.cos(inputs['angle'])