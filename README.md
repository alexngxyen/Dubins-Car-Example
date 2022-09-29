# Dubins-Car-Example
Control Barrier Function (CBF) and Control Lyapunov Function (CLF) based control methods implemented on the simple kinematic car model (also known as [Dubins car](https://en.wikipedia.org/wiki/Dubins_path)) to move an agent from a start point to a goal point is presented. Two scenarios are considered: i) an obstacle blocks a path to the goal point so safety and stability constraints are imposed and ii) no obstacle blocks a path to the goal point so only staiblity constraints are imposed. Alternatively, a nominal control input can be introduced into the optimization problem which minimially modifies the nominal control input subject to safety constraints. It is assumed the controller's maximum and minimum input saturates at a certain value.

Note: A script implementing several quadratic programs (QPs) in CVXPY is included as well. 

# Python Package Dependencies:
- timeit
- math 
- numpy 
- matplotlib
- cvxpy 
- scipy 
