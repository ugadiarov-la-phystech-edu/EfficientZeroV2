# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

from ez.agents.ez_atari import EZAtariAgent
from ez.agents.ez_dmc_image import EZDMCImageAgent
from ez.agents.ez_dmc_state import EZDMCStateAgent
from ez.agents.ez_shapes2d import EZShapes2dAgent

names = {
    'atari_agent': EZAtariAgent,
    'dmc_image_agent': EZDMCImageAgent,
    'dmc_state_agent': EZDMCStateAgent,
    'shapes2d_agent': EZShapes2dAgent,
}