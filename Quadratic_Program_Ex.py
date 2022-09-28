#  ============================================================================
#  Name        : Quadratic_Program_Ex.py
#  Description : Several examples of quadratic programming coded using the 
#                CVXPY package in Python.
#  Author      : Alex Nguyen
#  Date        : September 2022
#  ============================================================================

# Import packages.
import cvxpy as cp
import numpy as np

""" Example 1: Prelim Exam 2012 """
# Number of States
number_of_states = 5

# Initialize QP
A = np.array([1, 0, 1, 0, 1])
b = np.array([6])

# Define CVXPY Problem
x           = cp.Variable(number_of_states)
objective   = cp.Minimize((1/2)*cp.quad_form(x, np.identity(number_of_states))) 
constraints = [b <= A @ x]

# Solve CVXPY Problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("\n[Ex. 1] The optimal value is", prob.value, " with a solution x = ", x.value, "and dual variables =", constraints[0].dual_value,"\n")

""" Example 2: Optimization Methods Course Example (Lecture 17 - QP for Active Set Method) """
# Number of States
number_of_states = 2

# Initialize QP
c = np.array([-3, -2])
A = np.array([[1, 1], [-1, 0], [0, -1]])
b = np.array([[4], [0], [0]]).reshape(3,)

# Define CVXPY Problem
x           = cp.Variable(number_of_states)
objective   = cp.Minimize((1/2)*cp.quad_form(x, np.identity(number_of_states)) + c.T @ x) 
constraints = [A @ x <= b]

# Solve CVXPY Problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("\n[Ex. 2] The optimal value is", prob.value, " with a solution x = ", x.value, "and dual variables =", constraints[0].dual_value,"\n")

""" Example 3: Online Example (https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Quadratic_Programming.pdf) """
# Number of States
number_of_states = 3

# Initialize QP
Q = 2*np.array([[1, 0, 1/2], [0, 2, 0], [1/2, 0, 3]])
c = np.array([1, -2, 4])
Aeq = np.array([2, 3, 4])
beq = np.array([5])
A = np.array([[3, 4, -2], [3, -2, -1], [1, 0, 0], [-1, 0, 0],[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
b = np.array([10, -2, 5, 0, 5, -1, 5, 0])

# Define CVXPY Problem
x           = cp.Variable(number_of_states)
objective   = cp.Minimize((1/2)*cp.quad_form(x, Q) + c.T @ x) 
constraints = [A @ x <= b, Aeq @ x == beq]

# Solve CVXPY Problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("\n[Ex. 3] The optimal value is", prob.value, " with a solution x = ", x.value, "and dual variables =", constraints[0].dual_value,"\n")