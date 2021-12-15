from __future__ import division
import numpy as np

from openmdao.api import Group, IndepVarComp

from components.moment_comp import CoordTransfComp
from components.local_stiffness_matrix_comp import LocalStiffnessMatrixComp
from components.global_stiffness_matrix_comp import GlobalStiffnessMatrixComp
from components.states_comp import StatesComp
from components.displacements_comp import DisplacementsComp
from components.compliance_comp import ComplianceComp
from components.volume_comp import VolumeComp


class PathGroup(Group):

    def initialize(self):
        self.metadata.declare('centre_line', types=np.ndarray)
        self.metadata.declare('angle', types=np.ndarray)
        self.metadata.declare('width', types=np.ndarray)
        self.metadata.declare('num_elements', int)

    def setup(self):
        centre_line = self.metadata['centre_line']
        angle = self.metadata['angle']
        width = self.metadata['width']
        num_elements = self.metadata['num_elements']

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = -1.

        inputs_comp = IndepVarComp()
        inputs_comp.add_output('z', shape=num_elements)
        self.add_subsystem('inputs_comp', inputs_comp)

        xy_comp = CoordTransfComp(num_elements=num_elements, centre_line=centre_line, angle=angle, width=width)
        self.add_subsystem('xy_comp', xy_comp)

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = GlobalStiffnessMatrixComp(num_elements=num_elements)
        self.add_subsystem('global_stiffness_matrix_comp', comp)

        comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('states_comp', comp)

        comp = DisplacementsComp(num_elements=num_elements)
        self.add_subsystem('displacements_comp', comp)

        comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('compliance_comp', comp)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp)

        self.connect('inputs_comp.h', 'xy_comp.h')
        self.connect('xy_comp.x', 'local_stiffness_matrix_comp.I')
        self.connect('xy_comp.y', 'local_stiffness_matrix_comp.I')
        self.connect(
            'local_stiffness_matrix_comp.K_local',
            'global_stiffness_matrix_comp.K_local')
        self.connect(
            'global_stiffness_matrix_comp.K',
            'states_comp.K')
        self.connect(
            'states_comp.d',
            'displacements_comp.d')
        self.connect(
            'displacements_comp.displacements',
            'compliance_comp.displacements')
        self.connect(
            'inputs_comp.h',
            'volume_comp.h')

        self.add_design_var('inputs_comp.z', lower=-width[0], upper=width[0])
        self.add_objective('energy_comp.energy')
        self.add_constraint('volume_comp.volume', equals=volume)
