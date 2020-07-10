# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

# Position and Velocity Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/PositionControl.cpp
# Desired Thrust to Desired Attitude based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/Utility/ControlMath.cpp
# Attitude Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
# and https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
# Rate Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/mc_att_control_main.cpp

import numpy as np
from numpy import pi, array, diag, dot
from numpy import sin, cos, tan, sqrt
from numpy.linalg import norm
import utils
import config
from quadFiles.initQuad import sys_params

rad2deg = 180.0/pi
deg2rad = pi/180.0

class Control:
    
    def __init__(self, quad, yawType):
        self.sDesCalc = np.zeros(16)
        if (yawType == 0):
            att_P_gain[2] = 0
        self.pos_sp    = np.zeros(3)
        self.vel_sp    = np.zeros(3)
        self.acc_sp    = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp    = np.zeros(3)
        self.pqr_sp    = np.zeros(3)
        self.yawFF     = np.zeros(3)
        self.U_list    = np.zeros(4)
        self.A1, self.A2, self.A3, self.A4, self.A5, self.A6 = None, None, None, None, None, None


    def get_controller_coefs(self, backstepping_type):
        if (backstepping_type == 0): #Backstepping_1
            self.A1, self.A2, self.A3 = 15*diag([1,1]), 10*diag([1,1]), 15*diag([1,1]) 
            self.A4, self.A5, self.A6 = 10*diag([1,1]), 15*diag([1,1]), 10*diag([1,1]) 
        elif (backstepping_type == 1): #Backstepping_2
            self.A1, self.A2, self.A3 = 10*diag([1,1]), 5*diag([1,1]), 10*diag([1,1]) 
            self.A4, self.A5, self.A6 = 5*diag([1,1]), 10*diag([1,1]), 5*diag([1,1])
        elif (backstepping_type == 2): #Backstepping_3
            self.A1, self.A2, self.A3 = 5*diag([1,1]), 3*diag([1,1]), 10*diag([1,1]) 
            self.A4, self.A5, self.A6 = 7*diag([1,1]), 1*diag([1,1]), 1*diag([1,1])  
        elif (backstepping_type == 3): #Backstepping_4
            self.A1, self.A2, self.A3 = 2*diag([1,1]), 5*diag([1,1]), 2*diag([1,1]) 
            self.A4, self.A5, self.A6 = 5*diag([1,1]), 2*diag([1,1]), 5*diag([1,1]) 
        elif (backstepping_type == 4): #Single Shot
            self.A1, self.A2, self.A3 = 1*diag([1,1]), 1*diag([1,1]), 1*diag([1,1]) 
            self.A4, self.A5, self.A6 = 1*diag([1,1]), 1*diag([1,1]), 1*diag([1,1]) 
            
    
    def controller(self, traj, quad, sDes, backstepping_type):

        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp[:]    = traj.sDes[0:3]
        self.vel_sp[:]    = traj.sDes[3:6]
        self.acc_sp[:]    = traj.sDes[6:9]
        self.thrust_sp[:] = traj.sDes[9:12]
        self.eul_sp[:]    = traj.sDes[12:15]
        self.pqr_sp[:]    = traj.sDes[15:18]
        self.yawFF[:]     = traj.sDes[18]
        
        # Select Controller
        # ---------------------------
        
        self.get_controller_coefs(backstepping_type)
        self.backstepping(quad)


        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp

    def backstepping(self, quad):
        self.params = sys_params()

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
        
        U1, U2, U3, U4 = self.U_list
        
        #states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]
        x, y, z = quad.state[0], quad.state[1], quad.state[2]
        phi, theta, psi = quad.state[3], quad.state[4], quad.state[5]
        x_dot, y_dot, z_dot = quad.state[6], quad.state[7], quad.state[8]
        phi_dot, theta_dot, psi_dot = quad.state[9], quad.state[10], quad.state[11]
        
    #     ref_traj = [xd[i], yd[i], zd[i], xd_dot[i], yd_dot[i], zd_dot[i], 
    #                 xd_ddot[i], yd_ddot[i], zd_ddot[i], xd_dddot[i], yd_dddot[i],
    #                 xd_ddddot[i], yd_ddddot[i], psid[i], psid_dot[i], psid_ddot[i]]

        
        xd, yd, zd = self.pos_sp[0], self.pos_sp[1], self.pos_sp[2], 
        xd_dot, yd_dot, zd_dot = self.vel_sp[0], self.vel_sp[1], self.vel_sp[2]
        xd_ddot, yd_ddot, zd_ddot = self.acc_sp[0], self.acc_sp[1], self.acc_sp[2]
        xd_dddot, yd_dddot = 0, 0
        xd_ddddot, yd_ddddot = 0, 0
        psid, psid_dot, psid_ddot = self.eul_sp[2], self.pqr_sp[2], 0
        
        x1, x2, x3 = array([[x], [y]]), array([[x_dot], [y_dot]]), array([[phi], [theta]])
        x4, x5, x6 = array([[phi_dot], [theta_dot]]), array([[psi], [z]]), array([[psi_dot], [z_dot]])
        
        g0 = array([[np.cos(psi), np.sin(psi)],  [np.sin(psi), -np.cos(psi)]])
        g0_inv = array([[np.cos(psi), np.sin(psi)],  [np.sin(psi), -np.cos(psi)]])
        
        g1 = array([[theta_dot*psi_dot*I1],  [phi_dot*psi_dot*I2]])
        g2 = array([[phi_dot*theta_dot*I3],  [-g]])
        
        l0 = array([[np.cos(phi)*np.sin(theta)],  [np.sin(phi)]])*U1/m 
        dl0_dx3 = array([[-np.sin(phi)*np.sin(theta), np.cos(phi)*np.cos(theta)],  [np.cos(phi), 0]])*U1/m 
        dl0_dx3_inv = array([[0, 1/np.cos(phi)],  [1/np.cos(theta)*1/np.cos(phi), 1/np.cos(phi)*np.tan(theta)*np.tan(phi)]])*m/U1 
        dl0_dx3_inv_dot = array([[0, 1/np.cos(phi)*np.tan(phi)*phi_dot], 
                                 [1/np.cos(theta)*1/np.cos(phi)*(np.tan(theta)*theta_dot + np.tan(phi)*phi_dot), 1/np.cos(phi)*((1/np.cos(theta))**2*np.tan(phi)*theta_dot + (-1+2*(1/np.cos(phi))**2)*np.tan(theta)*phi_dot)]])*m/U1 
                    
    #     Omega_square = Omega_coef_inv * abs([U1/b  U2/b  U3/b  U4/d]) 
    #     Omega_param = sqrt(Omega_square) 
    #     omega = Omega_param(2) + Omega_param[3] - Omega_param(1) - Omega_param(3) 

    #     h1 = [-Jr/Ixx*theta_dot*omega  Jr/Iyy*phi_dot*omega] 
        h1 = 0 
        k1 = diag([l/Ixx, l/Iyy]) 
        k1_inv = diag([Ixx/l, Iyy/l]) 
        k2 = diag([1/Izz, np.cos(phi)*np.cos(theta)/m]) 
        k2_inv = diag([Izz, m/(np.cos(phi)*np.cos(theta))]) 
        
        x1d = array([[xd], [yd]])  
        x1d_dot = array([[xd_dot], [yd_dot]]) 
        x1d_ddot = array([[xd_ddot], [yd_ddot]]) 
        x1d_dddot = array([[xd_dddot], [yd_dddot]]) 
        x1d_ddddot = array([[xd_ddddot], [yd_ddddot]]) 
        
        x5d = array([[psid], [zd]])
        x5d_dot = array([[psid_dot], [zd_dot]]) 
        x5d_ddot = array([[psid_ddot], [zd_ddot]]) 
        
        z1 = x1d - x1
        v1 = x1d_dot + dot(self.A1,z1) 
        z2 = v1 - x2 
        z1_dot = -dot(self.A1,z1) + z2 
        v1_dot = x1d_ddot + dot(self.A1,z1_dot) 
        v2 = dot(g0_inv, z1 + v1_dot + dot(self.A2,z2)) 
        z3 = v2 - l0  
        z2_dot = -z1 - dot(self.A2,z2) + dot(g0,z3) 
        z1_ddot = -dot(self.A1,z1_dot) + z2_dot 
        v1_ddot = x1d_dddot + dot(self.A1, z1_ddot) 
        v2_dot = dot(g0_inv, z1_dot + v1_ddot + dot(self.A2,z2_dot)) 
        v3 = dot(dl0_dx3_inv, dot(g0.T,z2) + v2_dot + dot(self.A3, z3)) 
        z4 = v3 - x4 
        z3_dot = -dot(g0.T, z2) - dot(self.A3,z3) + dot(dl0_dx3, z4) 
        z2_ddot = - z1_dot - dot(self.A2, z2_dot) + dot(g0, z3_dot) 
        z1_dddot = -dot(self.A1, z1_ddot) + z2_ddot 
        v1_dddot = x1d_ddddot + dot(self.A1, z1_dddot) 
        v2_ddot = dot(g0_inv, z1_ddot + v1_dddot + dot(self.A2, z2_ddot)) 
        v3_dot = dot(dl0_dx3_inv, dot(g0.T, z2_dot) + v2_ddot + dot(self.A3, z3_dot)) + dot(dl0_dx3_inv_dot, dot(g0.T, z2) + v2_dot + dot(self.A3, z3))
        l1 = dot(k1_inv, dot(dl0_dx3.T, z3) + v3_dot - g1 - h1 + dot(self.A4, z4)).ravel()
        
        z5 = x5d - x5 
        v5 = x5d_dot + dot(self.A5, z5) 
        z6 = v5 - x6 
        z5_dot = - dot(self.A5, z5) + z6 
        v5_dot = x5d_ddot + dot(self.A5, z5_dot) 
        l2 = dot(k2_inv, z5 + v5_dot - g2 + dot(self.A6, z6)).ravel()
        
        U1, U2, U3, U4 = l2[1], l1[0], l1[1], l2[0]
        
        U1 = np.clip(U1, 1.0, 1e2)
        U2 = np.clip(U2, -1e3, 1e3)
        U3 = np.clip(U3, -1e3, 1e3)
        U4 = np.clip(U4, -1e2, 1e2)
        
        U = np.array([U1, U2, U3, U4])

        self.U_list = U
        
        return U
