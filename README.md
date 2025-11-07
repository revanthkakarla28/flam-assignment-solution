# flam-assignment-solution
Below are the optimised values of Unknown Parameters M,Theta,X obtained from the parametric equation
# Unknown Parameters
Theta (degrees): 30.042849450547024 degrees ,in radians = 0.524346
M: 0.02998871984125151
X: 55.01586330115085

Final Curve expressions are 
x(t) = t*cos(0.524346) - exp(0.029988720*|t|)*sin(0.3*t)*sin(0.524346) + 55.015863301
y(t) = 42 + t*sin(0.524346) + exp(0.029988720*|t|)*sin(0.3*t)*cos(0.524346)

# Method Used
I obtained the unknown parameters (M,Theta,X) by minimizing the L1 error between the given CSV (x,y) points and the model equations using differential evolution (global optimization) in Python. 
I assigned t values from 6 to 60 with equal spacing.
