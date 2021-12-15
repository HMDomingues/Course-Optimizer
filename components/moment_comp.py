from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent


class CoordTransfComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('centre_line', types=np.ndarray)
        self.metadata.declare('angle', types=np.ndarray)
        self.metadata.declare('width', types=np.ndarray)

    def setup(self):
        num_elements = self.metadata['num_elements']

        self.add_input('z', shape=num_elements)
        self.add_output('x', shape=num_elements)
        self.add_output('y', shape=num_elements)

        rows = np.arange(num_elements)
        cols = np.arange(num_elements)
        self.declare_partials('x', 'z', rows=rows, cols=cols)
        self.declare_partials('y', 'z', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        centre_line = self.metadata['centre_line']
        centre_line_x = centre_line[:,0]
        centre_line_y = centre_line[:,1]
        width = self.metadata['width']
        angle = self.metadata['angle']

        outputs['x'] = centre_line_x + inputs['z'] * np.sin(angle) * width
        outputs['y'] = centre_line_y - inputs['z'] * np.cos(angle) * width

    def compute_partials(self, inputs, partials):
        width = self.metadata['width']
        angle = self.metadata['angle']

        partials['x', 'z'] = np.sin(angle) * width
        partials['y', 'z'] = - np.sin(angle) * width
