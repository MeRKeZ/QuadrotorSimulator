# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:38:26 2019

@author: onur
"""

from numpy import dot, cos, sin, tan, arcsin, clip, pi
from casadi import *
from numpy.linalg import inv
from numpy.random import uniform
import numpy as np
from rosc import RosClass
from TrajectoryGenerator import TrajectoryGenerator
import rospy
import csv
import pandas as pd
import os
import time
import operator
from mytraj import MyTraj 
import copy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

#stats_columns = ['x','y','z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'des_x','des_y','des_z', 'des_vx', 'des_vy', 'des_vz', 'des_ax', 'des_ay', 'des_az', 'des_roll', 'des_pitch', 'des_yaw', 'cmd_accel_x', 'cmd_accel_y', 'cmd_accel_z']
stats_columns = ['pos_diffx','pos_diffy','pos_diffz', 'vel_desx', 'vel_desy','vel_desz','acc_desx','acc_desy','acc_desz', 'velx','vely','velz','accx','accy','accz','roll','pitch','yaw', 'Tf', 'Cost', 'controller_ID']
stats_filename = "flight.csv"



#stats_columns = ['x_pso','y_pso','z_pso', 'vx_pso', 'vy_pso', 'vz_pso', 'ax_pso', 'ay_pso', 'az_pso', 'roll_pso', 'pitch_pso', 'yaw_pso', 'roll_rate_pso', 'pitch_rate_pso', 'yaw_rate_pso', 'des_x_pso','des_y_pso','des_z_pso', 'des_vx_pso', 'des_vy_pso', 'des_vz_pso', 'des_ax_pso', 'des_ay_pso', 'des_az_pso', 'des_roll_pso', 'des_pitch_pso', 'des_yaw_pso', 'cmd_accel_x_pso', 'cmd_accel_y_pso', 'cmd_accel_z_pso']
#stats_filename = "flight_pso.csv"
# Simulation parameters
g = 9.81
m = 1.52
Ixx = 0.0347563
Iyy = 0.0458929
Izz = 0.0977
I1 = (Iyy - Izz) / Ixx
I2 = (Izz - Ixx) / Iyy
I3 = (Ixx - Iyy) / Izz
Jr = 0.0001
l = 0.09

b = 8.54858e-06
d = 1.6e-2
# Position coefficients
Kp_x = 6 #3.2708367767553828 #15.18
Kp_y = 6 #1.7037296814015281 #10.6
Kp_z = 6 #4.9#5.71

# Velocity coefficients
Kd_x = 4.7 #5.868897305196371  #5.82
Kd_y = 4.7 #4.333355366979583  #3.95 
Kd_z = 4.7 #1.4693439919200522  #5.37

#Attitude coefficients
Kp_roll = 2
Kp_pitch = 2.3
Kp_yaw = 0.15

#Angular Rate coefficients
Kd_roll = 0.4
Kd_pitch = 0.52
Kd_yaw = 0.18

Kp = np.array([Kp_x, Kp_y, Kp_z])
Kd = np.array([Kd_x, Kd_y, Kd_z])
Kp_ang = np.array([Kp_roll, Kp_pitch, Kp_yaw])
Kd_ang = np.array([Kd_roll, Kd_pitch, Kd_yaw])

yaw_des = np.array([0.0])
yawdot_des = np.array([0.0])


F_min = 0.0001
F_max = 2*m*g

N_Control_MPC = 5

Tf = 10

Controller = ["Backstepping_1", "Backstepping_2", "Backstepping_3", "Backstepping_4"]

single_shot = ["Back"] # just a check flight to test the coefficients


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop_layer = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(18, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc4(x))
        x = self.drop_layer(x)
        x = self.fc5(x)
        return x

def predict(X, model, device):
	#Validation part    
	model.eval()
	inputs = torch.from_numpy(X).to(device)
        
	outputs = model(inputs.float())
	_, pred = torch.max(outputs, 1)

	return pred

def write_stats(stats): #testno,stats_columns
    df_stats = pd.DataFrame([stats], columns=stats_columns)
    df_stats.to_csv(stats_filename, mode='a', index=False,header=not os.path.isfile(stats_filename))


def quad_sim(pos0, vel0, ang0, traj, trajectory, cont=None, cost_dict=None, scaler=None, model=None, device=None, status="TEST"):
	"""
	Calculates the necessary thrust and torques for the quadrotor to
	follow the trajectory described by the sets of coefficients
	x_c, y_c, and z_c.
	"""

	state = ros.reset(pos0, vel0, ang0)
	#Position
	x_pos = state[0]
	y_pos = state[1]
	z_pos = state[2]
	#Linear Velocity
	x_vel = state[3]
	y_vel = state[4]
	z_vel = state[5]
	#Acceleration
	x_acc = state[6]
	y_acc = state[7]
	z_acc = state[8]
	#Euler Angles
	roll = state[9]
	pitch = state[10]
	yaw = state[11]
	#Angular Rates
	roll_rate = state[12]
	pitch_rate = state[13]
	yaw_rate = state[14]


	#Time
	dt = 0.0
	t = 0.0
	current_time = time.time()
	last_time = np.copy(current_time)
	costValue = 0

	count_dict = {"Backstepping_1": 0, "Backstepping_2": 0, "Backstepping_3": 0, "Backstepping_4": 0}

	

	while t <= Tf:
		
		pos_des, vel_des, acc_des = traj.givemepoint(trajectory, t)
		current_states = np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel])


		#Test part
		if status == "TEST":
			X_test = np.array([x_pos-pos_des[0], y_pos-pos_des[1], z_pos-pos_des[2], x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, roll, pitch, vel_des[0], vel_des[1], vel_des[2], acc_des[0], acc_des[1], acc_des[2], Tf]).reshape(1,-1)
			X_test = scaler.transform(X_test)
			pred = predict(X_test, model, device)
			cont = Controller[pred[0]]
			count_dict[cont] += 1
		# print ("Predicted Controller: ", cont)

		if (cont == Controller[0]): #Backstepping_1
			coeff = [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
			M = backstepping(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, coeff)
		elif (cont == Controller[1]): #Backstepping_2
			coeff = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
			M = backstepping(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, coeff)
		elif (cont == Controller[2]): #Backstepping_3
			coeff = [1.0, 3.0, 3.0, 3.0, 1.0, 4.0]
			M = backstepping(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, coeff)
		elif (cont == Controller[3]): #Backstepping_4
			coeff = [3.0, 1.5, 3.0, 1.5, 3.0, 2.0]
			M = backstepping(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, coeff)
		elif (cont == Controller[4]): # PID
			M = PID(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate)
		elif (cont == Controller[5]): # Feedback_Linearization
			M = Feedback_Linearization(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw)
		elif (cont == single_shot[0]):
			coeff = [1.0, 3.0, 3.0, 3.0, 1.0, 4.0]
			M = backstepping(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, coeff)

		# elif (cont == Controller[2]): # MPC
		# 	# control_inputs = MPC(pos_des, current_states)
		# 	# M = np.array([control_inputs[0][0], control_inputs[0][1], control_inputs[0][2], 0])

		# 	if i < (N_Control_MPC-1):
		# 		M = np.array([control_inputs[i][0], control_inputs[i][1], control_inputs[i][2], 0])
		# 		i = i + 1
		# 	else:
		# 		control_inputs = MPC(pos_des, current_states, t)
		# 		M = np.array([control_inputs[0][0], control_inputs[0][1], control_inputs[0][2], 0])
		# 		i = 1

		


 
		# print ("u1: {0}, phi_des: {1}, theta_des: {2}".format(M[0],M[1],M[2]))

		state = ros.update(M)
		current_time = time.time()
		dt= current_time - last_time

		#Position
		x_pos = state[0]
		y_pos = state[1]
		z_pos = state[2]
		#Linear Velocity
		x_vel = state[3]
		y_vel = state[4]
		z_vel = state[5]
		#Acceleration
		x_acc = state[6]
		y_acc = state[7]
		z_acc = state[8]
		#Euler Angles
		roll = state[9]
		pitch = state[10]
		yaw = state[11]
		#Anglular Rates
		roll_rate = state[12]
		pitch_rate = state[13]
		yaw_rate = state[14]


		position_tracking_error = np.power((pos_des[0]-x_pos),2) + np.power((pos_des[1]-y_pos),2) + np.power((pos_des[2]-z_pos),2)
		velocity_tracking_error = np.power((vel_des[0]-x_vel),2) + np.power((vel_des[1]-y_vel),2) + np.power((vel_des[2]-z_vel),2)
		acceleration_tracking_error = np.power((acc_des[0]-x_acc),2) + np.power((acc_des[1]-y_acc),2) + np.power((acc_des[2]-z_acc - g),2)
		costValue += t*(position_tracking_error)



		last_time = np.copy(current_time)
		t = t + dt


	if status == "TEST":
		print ("How many times Backstepping_1 is called?: ", count_dict["Backstepping_1"])
		print ("How many times Backstepping_2 is called?: ", count_dict["Backstepping_2"])
		print ("How many times Backstepping_3 is called?: ", count_dict["Backstepping_3"])
		print ("How many times Backstepping_4 is called?: ", count_dict["Backstepping_4"])
		# print ("How many times PID is called?: ", count_dict["PID"])
		# print ("How many times Feedback_Linearization is called?: ", count_dict["Feedback_Linearization"])

	else:
		print ("")
		print ("Controller: {0}, Cost: {1}".format(cont, costValue))
		print ("Final state, x: {0:.3}, y: {1:.3}, z: {2:.3}, vx: {3:.3}, vy: {4:.3}, vz: {5:.3}".format(x_pos,y_pos,z_pos,x_vel,y_vel,z_vel))


	cost_dict[cont] = costValue

	return cost_dict

def Feedback_Linearization(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw):

	FKpx, FKpy, FKpz = 7, 7, 7
	FKdx, FKdy, FKdz = 4, 4, 4

	xd, yd, zd = pos_des[0], pos_des[1], pos_des[2]
	xd_dot, yd_dot, zd_dot = vel_des[0], vel_des[1], vel_des[2]
	xd_dotdot, yd_dotdot, zd_dotdot = acc_des[0], acc_des[1], acc_des[2]

	u1 = m /(cos(roll)*cos(pitch)) * (zd_dotdot + FKdz*(zd_dot - z_vel) + FKpz*(zd - z_pos) + g)
	phid = arcsin(np.clip(-m/u1*(yd_dotdot + FKdy*(yd_dot - y_vel) + FKpy*(yd - y_pos)), -1, 1))
	thetad = arctan2(xd_dotdot + FKdx*(xd_dot - x_vel) + FKpx*(xd - x_pos), m*(zd_dotdot + FKdz*(zd_dot - z_vel) + FKpz*(zd - z_pos) + g))
	thetad = np.clip(thetad, -pi/2, pi/2)
	M = [u1, phid, thetad, 0]

	return M


def PID(des_pos, des_vel, des_acc, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate):
	pos = np.array([x_pos, y_pos, z_pos])
	vel = np.array([x_vel, y_vel, z_vel])
	acc = np.array([x_acc, y_acc, z_acc])
	rot = np.array([roll, pitch, yaw])
	omega = np.array([roll_rate, pitch_rate, yaw_rate])

	omega_des = [0.0, 0.0, 0]
	

	cmd_accel_x = des_acc[0] + Kd[0]*(des_vel[0] - vel[0]) + Kp[0]*(des_pos[0] - pos[0])
	cmd_accel_y = des_acc[1] + Kd[1]*(des_vel[1] - vel[1]) + Kp[1]*(des_pos[1] - pos[1])
	cmd_accel_z = des_acc[2] + Kd[2]*(des_vel[2] - vel[2]) + Kp[2]*(des_pos[2] - pos[2])

	F = np.clip(m*(g + cmd_accel_z), F_min, F_max)

	phi_des = (m/F)*(cmd_accel_x*np.sin(yaw) - cmd_accel_y*np.cos(yaw))
	theta_des = (m/F)*(cmd_accel_x*np.cos(yaw) + cmd_accel_y*np.sin(yaw))
	M = [F, phi_des,theta_des,0]

	return M

def backstepping(pos_des, vel_des, acc_des, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, coeff):
	a1, a2, a3, a4, a5, a6 = coeff

	A1 = a1*np.eye(2)
	A2 = a2*np.eye(2)
	A3 = a3*np.eye(2)
	A4 = a4*np.eye(2)
	A5 = a5*np.eye(2)
	A6 = a6*np.eye(2)

	x1 = np.array([[x_pos], [y_pos]])
	x2 = np.array([[x_vel], [y_vel]])
	x3 = np.array([[roll], [pitch]])
	x4 = np.array([[roll_rate], [pitch_rate]])
	x5 = np.array([[yaw], [z_pos]])
	x6 = np.array([[yaw_rate], [z_vel]])

	U1, U2, U3, U4 = 1, 0, 0, 0

    
#		g0 = np.array([[sin(yaw), cos(yaw)],[-cos(yaw), sin(yaw)]])
#		g0_inv = np.array([[cos(yaw), sin(yaw)], [sin(yaw), -cos(yaw)]])
	g0 = np.array([[cos(yaw), sin(yaw)],[sin(yaw), -cos(yaw)]])
	g0_inv = np.array([[cos(yaw), sin(yaw)], [sin(yaw), -cos(yaw)]])
#		g1 = np.array([[1, sin(roll)*tan(pitch)], [0, cos(roll)]])
#		g2 = np.array([[1/Ixx, 0], [0, 1/Iyy]])
#		g3 = np.array([[1, 0], [0, cos(roll)/cos(pitch)]])
#		g4 = np.array([[cos(roll)*cos(pitch)/m, 0], [0, 1/Izz]])
	g1 = np.array([[pitch_rate * yaw_rate * I1], [roll_rate * yaw_rate * I2]])
	g2 = np.array([[roll_rate * pitch_rate * I3], [-g]])

	l0 = np.array([[cos(roll)*sin(pitch)], [sin(roll)]])*U1 / m
	dl0_dx3 = np.array([[-sin(roll)*sin(pitch), cos(roll)*cos(pitch)], [cos(roll), 0]])*U1/m;
	dl0_dx3_inv = np.array([[0, 1/cos(roll)], [1/(cos(pitch)*cos(roll)), 1/cos(roll)*tan(pitch)*tan(roll)]])*m/U1
	dl0_dx3_inv_dot = np.array([[0, 1/cos(roll)*tan(roll)*roll_rate],
	[1/(cos(pitch)*cos(roll))*(tan(pitch)*pitch_rate + tan(roll)*roll_rate),
	1/cos(roll)*((1/cos(pitch))**2*tan(roll)*pitch_rate + (-1+2*(1/cos(roll))**2)*tan(pitch)*roll_rate)]])*m/U1;
	        	    
    
#		h1 = np.array([[-Jr/Ixx*theta_dot*omega],[Jr/Iyy*phi_dot*omega]])
	h1 = 0
	k1 = np.array([[l/Ixx, 0], [0, l/Iyy]])
	k1_inv = np.array([[Ixx/l, 0], [0, Iyy/l]])
	k2 = np.array([[1/Izz, 0], [0, cos(roll)*cos(pitch)/m]])
	k2_inv = np.array([[Izz, 0], [0, m/(cos(roll)*cos(pitch))]])

	x1d = np.array([pos_des[0], pos_des[1]]).reshape(-1, 1)
	x1d_dot = np.array([vel_des[0], vel_des[1]]).reshape(-1, 1)
	x1d_ddot = np.array([acc_des[0], acc_des[1]]).reshape(-1, 1)
	x1d_dddot = np.array([0, 0]).reshape(-1, 1)
	x1d_ddddot = np.array([0, 0]).reshape(-1, 1)

	x5d = np.array([0, pos_des[2]]).reshape(-1, 1)
	x5d_dot = np.array([0, vel_des[2]]).reshape(-1, 1)
	x5d_ddot = np.array([0, acc_des[2]]).reshape(-1, 1)

	z1 = x1d-x1
	v1 = x1d_dot + np.dot(A1,z1)
	z2 = v1 - x2
	z1_dot = -np.dot(A1, z1) + z2
	v1_dot = x1d_ddot + np.dot(A1, z1_dot)
	v2 = np.dot(g0_inv, (z1 + v1_dot + np.dot(A2,z2)))
	z3 = v2 - l0
	z2_dot = -z1 - np.dot(A2,z2) + np.dot(g0,z3)
	z1_ddot = -np.dot(A1,z1_dot) + z2_dot
	v1_ddot = x1d_dddot + np.dot(A1, z1_ddot)
	v2_dot = np.dot(g0_inv, (z1_dot + v1_ddot + np.dot(A2,z2_dot)))
	v3 = np.dot(dl0_dx3_inv, np.dot(g0.T, z2) + v2_dot + np.dot(A3, z3))
	z4 = v3 - x4
	z3_dot = -np.dot(g0.T, z2) - np.dot(A3, z3) + np.dot(dl0_dx3, z4)
	z2_ddot = - z1_dot - np.dot(A2, z2_dot) + np.dot(g0, z3_dot)
	z1_dddot = -np.dot(A1, z1_ddot) + z2_ddot
	v1_dddot = x1d_ddddot + np.dot(A1, z1_dddot)
	v2_ddot = np.dot(g0_inv, z1_ddot + v1_dddot + np.dot(A2, z2_ddot))
	v3_dot = np.dot(dl0_dx3_inv, np.dot(g0.T, z2_dot) + v2_ddot + np.dot(A3, z3_dot)) + np.dot(dl0_dx3_inv_dot, np.dot(g0.T, z2) + v2_dot + np.dot(A3,z3))
	l1 = np.dot(k1_inv, np.dot(dl0_dx3.T, z3) + v3_dot - g1 - h1 + np.dot(A4, z4))

	z5 = x5d - x5
	v5 = x5d_dot + np.dot(A5, z5)
	z6 = v5 - x6
	z5_dot = - np.dot(A5, z5) + z6
	v5_dot = x5d_ddot + np.dot(A5, z5_dot)
	l2 = np.dot(k2_inv, z5 + v5_dot - g2 + np.dot(A6, z6))


	U1, U2, U3, U4 = l2[1], l1[0], l1[1], l2[0]

	phi_des = arcsin(clip(m/U1*v2[1],-1, 1))
	theta_des = arcsin(clip(m/(U1*cos(phi_des))*v2[0],-1, 1))
	M= [U1, phi_des,theta_des,yawdot_des]

	return M



def MPC(ref_pos, current_states, T):
	# Formulate the DAE
	x=SX.sym('x',6) #x, y, z, xdot, ydot, zdot
	u=SX.sym('u',3) #u1, phid, thetad

	f = vertcat(x[3], x[4], x[5],
		   cos(u[1])*sin(u[2])*u[0]/m,
		   -sin(u[1])*u[0]/m,
		   cos(u[1])*cos(u[2])*u[0]/m - g)# system r.h.s

	Q = np.diag([1, 1, 1])
	R = np.diag([0.001, 0.01, 0.01])
	h=Q[0][0]*(x[0]-ref_pos[0])**2+Q[1][1]*(x[1]-ref_pos[1])**2 + Q[2][2]*(x[2]-ref_pos[2])**2 + R[0][0]*u[0]**2 + R[1][1]*u[1]**2 + R[2][2]*u[2]**2
	dae=dict(x=x,p=u,ode=f,quad=h)

	# Create solver instance
	# T = 1. # end time 
	N = N_Control_MPC # discretization
	options=dict(t0=0,tf=T/N)
	F=integrator('F','idas',dae,options)

	# Empty NLP
	w=[]; lbw=[]; ubw=[]
	G=[]; J=0

	# Initial conditions
	Xk=MX.sym('X0',6)
	w+=[Xk]
	lbw+=[current_states[0],current_states[1],current_states[2],current_states[3],current_states[4],current_states[5]]
	ubw+=[current_states[0],current_states[1],current_states[2],current_states[3],current_states[4],current_states[5]]

	# print ("len(lbw): ", len(lbw))
	# print ("len(ubw): ", len(ubw))

	for k in range(1,N+1):
		# Local control
		name='U'+str(k-1)
		Uk=MX.sym(name, 3)
		w+=[Uk]
		lbw+=[0, -pi/2, -pi/2]
		ubw+=[30, pi/2, pi/2]


		# Call integrator
		Fk=F(x0=Xk,p=Uk)
		J+=Fk['qf']

		# Local state
		name='X'+str(k)
		Xk=MX.sym(name, 6)
		w+=[Xk]
		lbw+=[-inf, -inf, -inf, -inf, -inf, -inf]
		ubw+=[inf, inf, inf, inf, inf, inf]
		G+=[Fk['xf']-Xk]
	    


	# Create NLP solver
	nlp=dict(f=J,g=vertcat(*G),x=vertcat(*w))
	S=nlpsol('S','ipopt',nlp,dict(ipopt=dict(tol=1e-6)))

	# Solve NLP
	r=S(lbx=lbw,ubx=ubw,x0=0,lbg=0,ubg=0)

	# control_inputs = [r['x'][6], r['x'][7], r['x'][8], 0]

	M = r['x'].shape[0]
	states = []
	control_inputs = []
	loop = True
	i, j = 0, 6
	while(loop):
		states.append(r['x'][i:i+6])
		control_inputs.append(r['x'][j:j+3])
		i = i + 9
		j = j + 9
		if j >= M:
			loop = False

	return control_inputs

def main():
	"""
	Calculates the x, y, z coefficients for the four segments 
	of the trajectory
	"""

	STATUS = "DATA_COLLECTION"
	# STATUS = "TEST"
	# STATUS = "SINGLE_SHOT"

	if STATUS == "TEST":
		with open('dataset.pkl') as f:  # Python 3: open(..., 'rb')
			X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

	    #To normalize data
		scaler = StandardScaler()
		scaler.fit(X_train)

		#Neural network
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model = Net()
		model.load_state_dict(torch.load('best_model.pt'))
		model = model.to(device)
		

	K = 3 # how many different trajectories to be followed

	# pos0 = [0,0,0]
	# vel0 = [0,0,0]
	# acc0 = [0,0,0]
	# ang0 = [0,0,0]
	# posf = [5,5,5]
	# velf = [0.2,0.2,0.2]
	# accf = [0,0,0]
	
	for i in range(K):


		pos0 = [uniform(low=-2.0, high=2.0, size=(1,))[0], uniform(low=-2.0, high=2.0, size=(1,))[0], uniform(low=0.0, high=2.0, size=(1,))[0]]
		vel0 = [uniform(low=-0.4, high=0.4, size=(1,))[0], uniform(low=-0.4, high=0.4, size=(1,))[0], uniform(low=-0.4, high=0.4, size=(1,))[0]]
		acc0 = [uniform(low=-0.15, high=0.15, size=(1,))[0], uniform(low=-0.15, high=0.15, size=(1,))[0], uniform(low=-0.15, high=0.15, size=(1,))[0]]
		ang0 = [uniform(low=-0.25, high=0.25, size=(1,))[0], uniform(low=-0.25, high=0.25, size=(1,))[0], 0]
		posf = [uniform(low=-5.0, high=5.0, size=(1,))[0], uniform(low=-5.0, high=5.0, size=(1,))[0], uniform(low=5.0, high=15.0, size=(1,))[0]]
		velf = [0.,0.,0.]#[uniform(low=0, high=0.5, size=(1,))[0], uniform(low=0, high=0.5, size=(1,))[0], uniform(low=0, high=0.5, size=(1,))[0]]
		accf = [0.,0.,0.]#[uniform(low=0, high=0.15, size=(1,))[0], uniform(low=0, high=0.15, size=(1,))[0], uniform(low=0, high=0.15, size=(1,))[0]]
		Tf = uniform(low=10, high=15, size=(1,))[0]


		traj = MyTraj(gravity = -9.81)
		trajectory = traj.givemetraj(pos0, vel0, acc0, posf, velf, accf, Tf)


		print ("")
		print ("-"*25)
		print ("Init, x: {0:.3}, y: {1:.3}, z: {2:.3}, vx: {3:.3}, vy: {4:.3}, vz: {5:.3}".format(pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2]))
		print ("Goal, x: {0:.3}, y: {1:.3}, z: {2:.3}, vx: {3:.3}, vy: {4:.3}, vz: {5:.3} in {6:.3} s.".format(posf[0], posf[1], posf[2], velf[0], velf[1], velf[2], Tf))

		cost_dict = {"Backstepping_1": 0, "Backstepping_2": 0, "Backstepping_3": 0, "Backstepping_4": 0}

		if STATUS == "DATA_COLLECTION":
			for cont in Controller:
				cost_dict = quad_sim(pos0, vel0, ang0, traj, trajectory, cont=cont, cost_dict=cost_dict, status=STATUS)
			
			min_index = min(cost_dict.iteritems(), key=operator.itemgetter(1))[0]
			print ("Step: {0}/{1},  Best Controller: {2}".format(i+1,K,min_index))
			print ("")
			write_stats([pos0[0]-posf[0], pos0[1]-posf[1], pos0[2]-posf[2], velf[0], velf[1], velf[2], accf[0], accf[1], accf[2], vel0[0], vel0[1], vel0[2], acc0[0], acc0[1], acc0[2], ang0[0], ang0[1], ang0[2], Tf, cost_dict[min_index], min_index])
		elif STATUS == "TEST":
			cost_dict = quad_sim(pos0, vel0, ang0, traj, trajectory, cost_dict=cost_dict, scaler=scaler, model=model, device=device, status=STATUS)
		elif STATUS == "SINGLE_SHOT":
			cont = single_shot[0]
			cost_dict = quad_sim(pos0, vel0, ang0, traj, trajectory, cont=cont, cost_dict=cost_dict, status=STATUS)


if __name__ == "__main__":

	# Getting back the objects:
	phase = 'RESET'
	Input_Type = 'Angle'
	rospy.init_node('RosClass', anonymous=True)
	ros = RosClass(phase, Input_Type)
	main()

