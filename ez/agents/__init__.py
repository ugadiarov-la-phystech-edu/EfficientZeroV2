# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

from ez.agents.ocez_shapes2d import OCEZShapes2dAgent
from ez.agents.ocez_cw import OCEZCWAgent
# from ez.agents.ocez_maniskill import OCEZManiskillAgent
# from ez.agents.ocez_robosuite import OCEZRobosuiteAgent

names = {
    'oc_shapes2d_agent': OCEZShapes2dAgent,
    'oc_cw_agent': OCEZCWAgent,
    #'oc_maniskill_agent': OCEZManiskillAgent,
    #'oc_robosuite_agent': OCEZRobosuiteAgent,
}