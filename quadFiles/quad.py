# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode

from quadFiles.initQuad import sys_params, init_state
from control import Control
import utils
import config

deg2rad = pi/180.0

class Quadcopter:

    def __init__(self, states0, Ti=0):
        
        # Quad Params
        # ---------------------------
        self.params = sys_params()

        # Initial State
        # ---------------------------
        self.state = init_state(states0)

        self.pos     = self.state[0:3]
        self.angle   = self.state[3:6]
        self.vel     = self.state[6:9]
        self.angular = self.state[9:12]
        self.backstepping = Control(quad, traj.yawType)

        self.wind_direct = 0.
        
        # Set Integrator
        # ---------------------------
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)
    

    def state_dot(self, t, state):

        # Import Params
        # ---------------------------    
        m   = self.params["m"]
        g    = self.params["g"]
        Ixx  = self.params["Ixx"]
        Iyy  = self.params["Iyy"]
        Izz  = self.params["Izz"]
        I1   = self.params["I1"]
        I2   = self.params["I2"]
        I3   = self.params["I3"]
        Jr   = self.params["Jr"]
        l  = self.params["l"]
        b  = self.params["b"]
        d  = self.params["d"]

        if (config.usePrecession):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x           = self.state[0]
        y           = self.state[1]
        z           = self.state[2]
        phi         = self.state[3]
        theta       = self.state[4]
        psi         = self.state[5]
        x_dot        = self.state[6]
        y_dot        = self.state[7]
        z_dot        = self.state[8]
        phi_dot     = self.state[9]
        theta_dot   = self.state[10]
        psi_dot     = self.state[11]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
    

        
            
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = np.array([
                [x_dot],
                [y_dot],
                [z_dot],
                [phi_dot],
                [theta_dot],
                [psi_dot],
                [(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*U[0]/m],
                [(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*U[0]/m],
                [-g + cos(phi)*cos(theta)*U[0]/m],
                [theta_dot*psi_dot*I1 - Jr / Ixx * theta_dot * omega  + l/Ixx*U[1]],
                [phi_dot*psi_dot*I2 + Jr / Iyy * phi_dot * omega + l/Iyy*U[2]], 
                [phi_dot*theta_dot*I3 + 1/Izz*U[3]]])
    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = np.zeros([12])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]

        return sdot

    def update(self, t, Ts):
        U = get_control_input(cont, Controllers, self.U_list, current_traj, state)
                               
        self.integrator.set_f_params()
        self.state = self.integrator.integrate(t, t+Ts)
                               

        # Wind Model
        # ---------------------------
        wind_direct = (np.random.randint(1,360)/180)*pi #Deg2Rad
        wind_noisev = wind.wind_shear(self.state, wind_direct)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:]

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts
