# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
from numpy.linalg import inv
import utils
import config


def sys_params():
    g = 9.81
    m = 1.52
    Ixx, Iyy, Izz = 0.0347563, 0.0458929, 0.0977
    I1 = (Iyy - Izz) / Ixx
    I2 = (Izz - Ixx) / Iyy
    I3 = (Ixx - Iyy) / Izz
    Jr = 0.0001
    l = 0.09
    b = 8.54858e-6
    d = 1.6e-2


    params = {}
    params["m"]   = m
    params["g"]    = g
    params["Ixx"]  = Ixx
    params["Iyy"]  = Iyy
    params["Izz"]  = Izz
    params["I1"]   = I1
    params["I2"]   = I2
    params["I3"]   = I3
    params["Jr"] = Jr
    params["l"] = l
    params["b"] = b
    params["d"] = d
    
    
    return params


def init_state(state0):
    
    x0     = state0[0]  # m
    y0     = state0[1]  # m
    z0     = state0[2]  # m
    phi0   = state0[3]  # rad
    theta0 = state0[4]  # rad
    psi0   = state0[5]  # rad
    vx0    = state0[6]  # m/s
    vy0    = state0[7]  # m/s
    vz0    = state0[8]  # m/s
    p      = state0[9]  # rad/s
    q      = state0[10] # rad/s
    r      = state0[11] # rad/s

    quat = utils.YPRToQuat(psi0, theta0, phi0)
    
    if (config.orient == "ENU"):
        z0 = -z0

    s = np.zeros(12)
    s[0]  = x0       # x
    s[1]  = y0       # y
    s[2]  = z0       # z
    s[3]  = 0.       # phi
    s[4]  = 0.       # theta
    s[5]  = 0.       # psi
    s[6]  = 0.       # xdot
    s[7]  = 0.       # ydot
    s[8]  = 0.       # zdot
    s[9] = 0.        # p
    s[10] = 0.       # q
    s[11] = 0.       # r
    
    return s
