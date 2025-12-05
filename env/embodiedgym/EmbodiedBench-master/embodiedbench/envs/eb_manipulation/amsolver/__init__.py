#Copy From the rlbench: https://github.com/stepjam/RLBench
__version__ = '1.1.0'

import numpy as np
import pyrep

pr_v = np.array(pyrep.__version__.split('.'), dtype=int)
if pr_v.size < 4 or np.any(pr_v < np.array([4, 1, 0, 2])):
    raise ImportError(
        'PyRep version must be greater than 4.1.0.2. Please update PyRep.')


from amsolver.environment import Environment
from amsolver.action_modes import ArmActionMode
from amsolver.observation_config import ObservationConfig
from amsolver.observation_config import CameraConfig
from amsolver.sim2real.domain_randomization import RandomizeEvery
from amsolver.sim2real.domain_randomization import VisualRandomizationConfig
from amsolver.sim2real.domain_randomization_environment import DomainRandomizationEnvironment
