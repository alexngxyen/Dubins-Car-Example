#  ============================================================================
#  Name        : CBF_with_nominal_controller_CLF_Dubins_Car_Type_I.py
#  Description : Constraint driven control implementation on the Dubins car toy 
#                example problem using a CBF certificate with a nominal control 
#                input (for loop implementation).
#  Author      : Alex Nguyen
#  Date        : September 2022
#  ============================================================================

""" Import Packages """
import timeit
import math 
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy import integrate, linalg

""" Functions """
def dubinsCarDynamics(constant_velocity, heading):
## This function defines the dynamics for the Dubins car problem.

    # Uncontrolled Drift Vector
    f = np.array([constant_velocity*math.cos(heading), constant_velocity*math.sin(heading), 0])

    # Control Actuation Vector
    g = np.array([0, 0, 1])

    return f, g

def defineCbf(robot_states, obstacle_position_states, constant_velocity, safe_distance, decay_constant):
## This function defines the control barrier function for the Dubins car problem.
    
    # Difference X and Y Positions of Robot and Obstacle 
    x_position_diff = (robot_states[0] - obstacle_position_states[0])
    y_position_diff = (robot_states[1] - obstacle_position_states[1])

    # Define the Control Barrier Function
    distance = (x_position_diff**2 + y_position_diff**2) - safe_distance**2
    derivative_distance = 2*constant_velocity*(x_position_diff*math.cos(robot_states[2]) + y_position_diff*math.sin(robot_states[2]))
    control_barrier_function = derivative_distance + decay_constant*distance

    return control_barrier_function

def defineCbfDerivative(robot_states, obstacle_position_states, constant_velocity, decay_constant):
## This function defines the Lie Derivatives of the control barrier function for the Dubins car problem.

    # Dublin Car Dynamics
    f, g = dubinsCarDynamics(constant_velocity, robot_states[2])

    # Difference X and Y Positions of Robot and Obstacle 
    x_position_diff = (robot_states[0] - obstacle_position_states[0])
    y_position_diff = (robot_states[1] - obstacle_position_states[1])

    # CBF Derivative 
    term_one       = 2*(decay_constant*x_position_diff + constant_velocity*math.cos(robot_states[2]))
    term_two       = 2*(decay_constant*y_position_diff + constant_velocity*math.sin(robot_states[2]))
    term_three     = 2*constant_velocity*(-math.sin(robot_states[2])*x_position_diff + math.cos(robot_states[2])*y_position_diff)
    cbf_derivative = np.array([term_one, term_two, term_three])

    # CBF Lie Derivative
    lie_derivative_f_cbf = cbf_derivative @ f
    lie_derivative_g_cbf = cbf_derivative @ g

    return lie_derivative_f_cbf, lie_derivative_g_cbf

def dublinCarDynamics(t, y, constant_velocity, control_input):
## This function defines the ODE of the Dubins car problem for numerical integration.   

    # Dublin Car Dynamics
    f = np.array([constant_velocity*math.cos(y[2]), constant_velocity*math.sin(y[2]), 0])
    g = np.array([0, 0, 1])

    # ODE
    dydt = f + g*control_input

    return dydt

""" Simulation Settings """
# Time Parameters
sampling_time     = 0.01
simulation_time   = 20                                                                    # Simulation Time Allocated (Could Be Less!)
simulation_length = math.ceil(simulation_time / sampling_time)

# State Parameters
number_of_states         = 3
constant_velocity        = 1                                                              # (Linear) Velocity [m/s]
obstacle_radius          = 2                                                              # Obstacle's Radius [m]
initial_states           = np.array([0, 5, 0])                                            # Initial Robot States [x-position, y-position, heading]
obstacle_position_states = np.array([5, 3])                                               # Obstacle Position States [x-position, y-position]
goal_position_states     = np.array([10, 0])                                              # Goal Position States [x-position, y-position]

# Obstacle
angle = np.linspace(0, 2*math.pi)
x_circle = obstacle_position_states[0] + obstacle_radius*np.cos(angle)
y_circle = obstacle_position_states[1] + obstacle_radius*np.sin(angle)

# Gravity Parameter
gravity = 9.81                                                                            # Constant Gravity Field [m/s^2]

# CBF and Nominal Controller Parameters
decay_constant  = 1                                                                       # CBF's Decay Constant 
cbf_rate        = 0.5
controller_gain = 2                

# Control Input Parameters
number_of_inputs  = 1
max_control_input = 3                                                                     # Maximum Control Input 
min_control_input = -3                                                                    # Minimum Control Input

# Preallocate Variables
state_vector_history    = np.zeros((number_of_states, simulation_length))
control_input_history   = np.zeros((number_of_inputs, simulation_length)) 
CBF_history             = np.zeros((simulation_length, ))
CBF_certificate_history = np.zeros((simulation_length, ))
time_history            = np.zeros((simulation_length, ))

""" Simulate Safe Trajectory from Initial Position to Goal Position """
# Begin Timer
start_time = timeit.default_timer() 

# Initialize Loop Vectors
states = initial_states
time   = 0

# Initialize History Vectors
state_vector_history[:, 0] = states
time_history[0]            = time

for k in range(simulation_length):
    # Compute Nominal Controller
    delta_x            = goal_position_states[0] - states[0]
    delta_y            = goal_position_states[1] - states[1]
    gradient_descent   = 2*(math.tan(states[2]) - (delta_y/delta_x)) / math.cos(states[2])**2
    nominal_controller = -controller_gain*gradient_descent

    # Control Barrier Certificate (CBC)
    cbf = defineCbf(states, obstacle_position_states, constant_velocity, obstacle_radius, decay_constant)
    lie_derivative_f_cbf, lie_derivative_g_cbf = defineCbfDerivative(states, obstacle_position_states, constant_velocity, decay_constant)

    # Optimization Constraints {A[u; slack] <= b}
    A_cbf  = np.array([-lie_derivative_g_cbf])
    A_umax = np.array([1])
    A_umin = np.array([-1])
    A      = np.vstack((A_cbf, A_umax, A_umin))

    b_cbf  = np.array([lie_derivative_f_cbf + cbf_rate*cbf**3])
    b_umax = np.array([max_control_input])
    b_umin = np.array([-min_control_input])
    b      = np.vstack((b_cbf, b_umax, b_umin)).reshape(3,)

    # Cost Function ||u - u_{nominal}||^2 
    control_input_reference = np.zeros((number_of_inputs, 1))
    control_input_weight    = np.eye(number_of_inputs)

    # Solve Quadratic Program Using CVXOPT
    x           = cp.Variable(number_of_inputs)
    objective   = cp.Minimize(cp.sum_squares(x - nominal_controller))  
    constraints = [A @ x <= b]
    prob        = cp.Problem(objective, constraints)
    prob.solve()

    # Check Optimization Solution's Status
    if abs(prob.value) == math.inf:
        print('\n\nOptimization Problem Status:', prob.status, '\n')
        break

    # Extract Optimal Control Inputs
    u_optimal = x.value 

    # Numberical Integration
    initial_ODE_states = states
    time_span          = (time, time + sampling_time)
    solution           = integrate.solve_ivp(dublinCarDynamics, time_span, initial_ODE_states, args=(constant_velocity, u_optimal), method='RK45', rtol=1e-10)
    states             = solution.y[:, -1]

    # Save Variables
    state_vector_history[:, k+1] = states
    control_input_history[:, k]  = u_optimal
    time_history[k+1]            = solution.t[-1] 
    CBF_history[k]               = cbf
    CBF_certificate_history[k]   = lie_derivative_f_cbf + lie_derivative_g_cbf*u_optimal + cbf_rate*cbf**3

    # Update Time
    time += sampling_time

    # Break Condition
    if linalg.norm(states[0:2] - goal_position_states) < 0.05:
        break 

# Delete Extra Preallocated Space
state_vector_history    = state_vector_history[:, :k+2]
control_input_history   = control_input_history[:, :k+1]
CBF_history             = CBF_history[:k+1] 
CBF_certificate_history = CBF_certificate_history[:k+1]  
time_history            = time_history[:k+2] 

# End Timer
end_time = timeit.default_timer() 

# Set Logical Variables and Print Results
if abs(prob.value) == math.inf:
    # Logical Variables
    show_simulation_environment = False
    show_input_and_states       = False
    show_CBF                    = False  

else:
    # Print Simulation Time
    print('\nSimulation Time = {}; Run Time = {}'.format(time_history[-1], end_time - start_time))
    print('Robot got within {} m goal position!\n'.format(linalg.norm(states[0:2] - goal_position_states)))

    # Logical Variables
    show_simulation_environment = True
    show_input_and_states       = True
    show_CBF                    = False

""" Plot Results """
if show_simulation_environment:
# Visualize Simulation Environment 
    plt.figure()
    for i_plt in range(0, k+1, 12):
        plt.scatter(state_vector_history[0, 0], state_vector_history[1, 0], s=50, c='g')
        plt.scatter(goal_position_states[0], goal_position_states[1], s=50, c='r')
        plt.plot(state_vector_history[0, :i_plt+1], state_vector_history[1, :i_plt+1], 'k', linewidth=2)
        plt.plot(x_circle, y_circle, 'b', linewidth=2)
        plt.axis('equal')
        plt.xlabel('East (m)') 
        plt.ylabel('North (m)')
        plt.legend(['Start Point', 'Goal Point', 'Robot Trajectory', 'Obstacle'], loc='best')  
        plt.pause(0.01) 
        plt.clf()

if show_input_and_states:
    # Control Input 
    plt.figure()
    plt.plot(time_history[:-1], control_input_history[0, 1:], 'r', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (rad/s)')
    plt.xlim(time_history[0], time_history[-1])

    # Evolution of States
    plt.figure()
    for i in range(number_of_states):
        plt.plot(time_history, state_vector_history[i, :], linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Robot States')
    plt.legend(['x (m)', 'y (m)', r'$\theta$ (rad)'], loc='best')    
    plt.xlim(time_history[0], time_history[-1])

if show_CBF:
    # Evolution of CBF 
    plt.figure()
    plt.plot(time_history[:-1], CBF_history, 'r', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('CBF')
    plt.legend(['h'], loc='best')    
    plt.xlim(time_history[0], time_history[-1])

    # Evolution of CBC 
    plt.figure()
    plt.plot(time_history[:-1], CBF_certificate_history, 'b', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{h} + \alpha (h)$')
    plt.legend(['Control Barrier Certificate'], loc='best')    
    plt.xlim(time_history[0], time_history[-1])
    
# Show Plots
plt.show()