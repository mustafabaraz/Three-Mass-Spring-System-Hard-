# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 22:40:15 2022

@author: MUSTAFA
"""

import matplotlib.pyplot as plt
import numpy as np

# PART A):

# Let the rest length of the spring with k_1 be L_1, k_2 be L_2, k_3 be L_3 and k_4 be L_4.
# Let also the origin for positions be the left wall.
# This means:
    
""" Rest position of the first mass is x_first= L_1, second mass is x_second= L_1 + L_2, and
third mass is x_third= L_1 + L_2 + L_3. Notice that the lengths of the masses are neglected. 
Notice also that L_1 + L_2 + L_3 + L_4= L."""

""" Note: One can also define the rest positions of the masses as their own origins, so that
x_first=0, x_second= 0, and x_third= 0. This is what we eventually do when we compute the displacements
from the rest positions of the masses"""

# PART B):
    
# Define the constants of the motion:
    
m_1= 1 # Mass of the first object in kg.
m_2= 2 # Mass of the second object in kg.
m_3= 3 # Mass of the third object in kg.

k_1= 250 # Spring constant of the spring with k_1 in N/m.
k_2= 300 # Spring constant of the spring with k_2 in N/m.
k_3= 350 # Spring constant of the spring with k_3 in N/m.
k_4= 400 # Spring constant of the spring with k_4 in N/m.

L_1= 0.1 # Rest length of the spring with k_1 in m.
L_2= 0.2 # Rest length of the spring with k_2 in m.
L_3= 0.3 # Rest length of the spring with k_3 in m.
L_4= 0.4 # Rest length of the spring with k_4 in m.

L= 1 # Distance between the walls (in m) is naturally L_1 + L_2 + L_3 + L_4.

# Equations of the motion are given as:
    
# For first mass: m_1*a_1 = -k_1*x_1 + k*2(x_2-x_1)
# For the second mass: m_2*a_2= k_2*(x_1-x_2)+k_3*(x_3-x_2)
# For the third mass: m_3*a_3= k_3*(x_2-x_3)-k_4*x_3


# Define the inital conditions:

x_i1= L_1 + 0.003 # Initial position of the first mass in m (initial disturbance is 0.003 m for the first mass).
x_i2= L_1 + L_2 + 0.004 # Initial position of the second mass in m (initial disturbance is 0.004 m for the second mass).
x_i3= L_1 + L_2 + L_3 +0.005  # Initial position of the third mass in m (initial disturbance is 0.005 m for the third mass).

v_i1= 0 # Initial velocity of the first mass in m/s.
v_i2= 0 # Initial velocity of the second mass in m/s.
v_i3= 0 # Initial velocity of the third mass in m/s.


a_i1= (-k_1*(0.003)+ k_2*(0.004-0.003))/m_1  # Initial acceleration of the first mass in m/s^2.
a_i2= (k_2*(0.003-0.004)+k_3*(0.005-0.004))/m_2 # Initial acceleration of the second mass in m/s^2.
a_i3= (k_3*(0.004-0.005)-k_4*0.005)/m_3 # Initial acceleration of the third mass in m/s^2.

kinetic= 1/2*m_1*v_i1**2+1/2*m_2*v_i2**2+ 1/2*m_3*v_i3**2 # Initial kinetic energy of the system in J.
potential = 1/2*k_1*(0.003)**2+ 1/2*k_2*(0.003-0.004)**2+ 1/2*k_3*(0.004-0.005)**2+ 1/2*k_4*(0.005)**2 # Initial potential energy of the system in J.
total_en= kinetic + potential # Initial total energy of the system in J.

# Define a function for these system of ODE's, namely:
    
#  m_1*a_1 = -k_1*x_1 + k*2*(x_2-x_1)
#  m_2*a_2= k_2*(x_1-x_2)+k_3*(x_3-x_2)
#  m_3*a_3= k_3*(x_2-x_3)-k_4*x_3 

def f(x,t):
    return np.array([x[1],(-k_1*x[0]+k_2*(x[2]-x[0]))/m_1, x[3], (k_3*(x[4]-x[2])+k_2*(x[0]-x[2]))/m_2, x[5], (-k_4*x[4]+k_3*(x[2]-x[4]))/m_3])

# Notice that the array above that is returned outside the function is of the form [v_1, a_1, v_2, a_2, v_3, a_3]
# since it is to be used for the fourth order Runge-Kutta Method.

# Set the limits for time:
    
t_start= 0 # in seconds.
t_end= 10 # in seconds.

# Set the step number:
    
N= 50000

# Set the step size:

h= (t_end-t_start)/N 

# Set the time axis:
    
time= np.arange(t_start, t_end+h, h)

# One must declare a 2D array (matrix) for the numerical position and velocity as (it must be for all of the time elapsed):
    
pos_vel= np.zeros((len(time),6)) # There are len(time) rows and 6 columns, notice that it is of the form [x_1, v_1, x_2, v_2, x_3, v_3].

# Also declare lists with only initial values of the numerical energies:
    
kin_en=[kinetic] # For the numerical kinetic energy.
pot_en=[potential] # For the numerical potential energy.
tot_en=[total_en] # For the numerical total energy.

# There must be non-zero initial conditions for the matrix of the numerical position and velocity:

pos_vel[0,:]=[0.003, v_i1, 0.004, v_i2, 0.005, v_i3] # Note that the initial disturbances are equal to the initial displacements from the rest positions (i.e. static equilibrium).

# Position and velocity matrix must be filled:
    
for i in range (0, len(time)-1): # We set the maximum range to len(time)-2 to match the dimensions of the matrix and time arrays.
    # k1, k2, k3, and k4 are necessary for the fourth order Runge-Kutta Method.
    k1 = h * f( pos_vel[i,:], time[i] )
    k2 = h * f( pos_vel[i,:] + 0.5 * k1, time[i] + 0.5 * h )
    k3 = h * f( pos_vel[i,:] + 0.5 * k2, time[i] + 0.5 * h )
    k4 = h * f( pos_vel[i,:] + k3, time[i+1] )
    # Update the elements of the numerical position and velocity matrix:
    pos_vel[i+1,:] = pos_vel[i,:] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0 # Elements of the numerical position and velocity matrix are updated.
   
for i in range (0, len(time)-1):
    # Update the numerical energy values with variables taken from the i+1th (i.e. updated row for the time t= t[i]+h, this is because we have already included the initial values to our lists for the numerical energies).
    kinetic= 1/2*m_1*pos_vel[i+1,1]**2+1/2*m_2*pos_vel[i+1,3]**2+ 1/2*m_3*pos_vel[i+1,5]**2
    potential = 1/2*k_1*(pos_vel[i+1,0])**2+ 1/2*k_2*(pos_vel[i+1, 0]-pos_vel[i+1,2])**2+ 1/2*k_3*(pos_vel[i+1,2]-pos_vel[i+1,4])**2+ 1/2*k_4*(pos_vel[i+1,4])**2
    total_en= kinetic+ potential
    # Fill the lists for the numerical energies:
    kin_en.append(kinetic)
    pot_en.append(potential)
    tot_en.append(total_en)

# Plotting the results:

# For the numerical energies:

plt.plot(time, kin_en, 'k', label= 'Numerical Kinetic Energy of the System (in J)')
plt.plot(time, pot_en, 'r', label= 'Numerical Potential Energy of the System (in J')
plt.plot(time, tot_en, 'b', label= 'Numerical Total Energy of the System (in J')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')  
plt.title('Numerical Energies of the System vs Time') 
plt.legend()
plt.show()

# For the numerical displacements:

plt.plot(time, pos_vel[:,0], 'k', label= 'Numerical Displacement of the First Mass from the Rest Position')
plt.plot(time, pos_vel[:,2], 'r', label= 'Numerical Displacement of the Second Mass from the Rest Position')
plt.plot(time, pos_vel[:,4], 'b', label= 'Numerical Displacement of the Third Mass from the Rest Position')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')  
plt.title('Numerical Displacements of the Masses vs Time') 
plt.legend()
plt.show()

# For the numerical velocities:

plt.plot(time, pos_vel[:,1], 'k', label= 'Numerical Velocity of the First Mass')
plt.plot(time, pos_vel[:,3], 'r', label= 'Numerical Velocity of the Second Mass')
plt.plot(time, pos_vel[:,5], 'b', label= 'Numerical Velocity of the Third Mass')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')  
plt.title('Numerical Velocities of the Masses vs Time') 
plt.legend()
plt.show()

# Let's calculate the exact result:
    
from scipy.integrate import odeint

# I have turned the system of 3 2nd order ODE's into 6 1st order ODE's
# letting u_1=x_1, u_2= dx_1/dt, v_1=x_2, v_2= dx_2/dt, g_1=x_3, g_2= dg_2/dt.


def rhs(t, u): # rhs stands for the right hand side of the 6 ODE's.
    # u stands for the array u=[u_1,u_2,v_1,v_2,g_1,g_2].
    return [u[1], ((-k_1*u[0])+k_2*(u[2]-u[0]))/m_1, u[3], (k_2*(u[0]-u[2])+k_3*(u[4]-u[2]))/m_2, u[5], (k_3*(u[2]-u[4])-k_4*u[4])/m_3]

result = odeint(rhs, [0.003, 0, 0.004, 0, 0.005, 0], time, tfirst=True )

# To store the exact energy values, one must follow the same procedure as the numerical computation:

kinetic= 1/2*m_1*v_i1**2+1/2*m_2*v_i2**2+ 1/2*m_3*v_i3**2 # Initial kinetic energy of the system in J.
potential = 1/2*k_1*(0.003)**2+ 1/2*k_2*(0.003-0.004)**2+ 1/2*k_3*(0.004-0.005)**2+ 1/2*k_4*(0.005)**2 # Initial potential energy of the system in J.
total_en= kinetic + potential # Initial total energy of the system in J.

kin_en_exact=[kinetic] # For the exact kinetic energy.
pot_en_exact=[potential] # For the exact potential energy.
tot_en_exact=[total_en] # For the exact total energy.    

for i in range (0, len(time)-1):
    # Update the exact energy values with variables taken from the i+1th (i.e. updated row for the time t= t[i]+h, this is because we have already included the initial values to our lists for the exact energies).
    kinetic= 1/2*m_1*result[i+1,1]**2+1/2*m_2*result[i+1,3]**2+ 1/2*m_3*result[i+1,5]**2
    potential = 1/2*k_1*(result[i+1,0])**2+ 1/2*k_2*(result[i+1, 0]-result[i+1,2])**2+ 1/2*k_3*(result[i+1,2]-result[i+1,4])**2+ 1/2*k_4*(result[i+1,4])**2
    # Fill the lists for the exact energies:    
    kin_en_exact.append(kinetic)
    pot_en_exact.append(potential)
    tot_en_exact.append(total_en)

# Plotting the exact results:
    
# Recall that u_1= x_1, u_2= dx_1/dt= vel_1, v_1= x_2, v_2=dx_2/dt= vel_2, g_1= x_3, and g_2= dx_3/dt= vel_3
# where the vel_i stands for the velocities of the masses.
    
# For the exact displacements:

plt.plot(time, result[:,0],'k', label= 'Exact displacement of the First Mass from the Rest Position')
plt.plot(time, result[:,2],'r', label= 'Exact displacement of the Second Mass from the Rest Position')
plt.plot(time, result[:,4],'b', label= 'Exact displacement of the Third Mass from the Rest Position')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')  
plt.title('Exact Displacements of the Masses vs Time') 
plt.legend()
plt.show()

# For the exact velocities:
    
plt.plot(time, result[:,1],'k', label= 'Exact velocity of the First Mass from the Rest Position')
plt.plot(time, result[:,3],'r', label= 'Exact velocity of the Second Mass from the Rest Position')
plt.plot(time, result[:,5],'b', label= 'Exact velocity of the Third Mass from the Rest Position')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')  
plt.title('Exact Velocities of the Masses vs Time') 
plt.legend()
plt.show()

# For the exact energies:

plt.plot(time, kin_en_exact, 'k', label= 'Exact Kinetic Energy of the System (in J)')
plt.plot(time, pot_en_exact, 'r', label= 'Exact Potential Energy of the System (in J')
plt.plot(time, tot_en_exact, 'b', label= 'Exact Total Energy of the System (in J')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')  
plt.title('Exact Energies of the System vs Time') 
plt.legend()
plt.show()


# Name: Ahmet Mustafa Baraz
# ID: 21702127
# Title of the Program: Three Masses Coupled with Four Springs without Damping.



