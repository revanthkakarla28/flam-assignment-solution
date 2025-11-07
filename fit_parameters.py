import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution



df = pd.read_csv(r"C:\Users\revan\Desktop\Projects\Flam\xy_data.csv")
df = df.sort_values(["x"]).reset_index(drop=True)
x_value = df["x"].values
y_value = df["y"].values
N = len(df)

t = np.linspace(6,60,N)


def x_y_prediction(theta,M,X,t):
  theta_rad = np.deg2rad(theta)
  x_hat = ((t*np.cos(theta_rad)) - ((np.exp(M*np.abs(t)))*(np.sin(0.3*t))*(np.sin(theta_rad))) + (X))
  y_hat = (42) +  ((t*np.sin(theta_rad))) + ((np.exp(M*np.abs(t)))*(np.sin(0.3*t))*(np.cos(theta_rad)))
  return x_hat,y_hat


def L1(x_hat,y_hat):
  return np.sum(np.abs(x_value - x_hat) + np.abs(y_value - y_hat))

def objective(params):
    theta_deg, M, X = params
    x_hat, y_hat = x_y_prediction(theta_deg,M,X,t)
    return L1(x_hat,y_hat)

bounds = [(0,50),(-0.05,0.05),(0,100)]

result = differential_evolution(objective,bounds=bounds,maxiter=400,popsize=20,mutation=(0.5,1.0),recombination=0.7,seed=42)

theta_opt, M_opt, X_opt = result.x

print("FINAL PARAMS =", result.x)
print("FINAL LOSS =", result.fun)
print("\nFINAL MODEL EQUATIONS:")
print(f"x(t) = t*cos({theta_opt:.9f}°) - exp({M_opt:.9f}*|t|)*sin(0.3*t)*sin({theta_opt:.9f}°) + {X_opt:.9f}")
print(f"y(t) = 42 + t*sin({theta_opt:.9f}°) + exp({M_opt:.9f}*|t|)*sin(0.3*t)*cos({theta_opt:.9f}°)")


x_hat, y_hat = x_y_prediction(theta_opt, M_opt, X_opt, t)

plt.figure(figsize=(10,10))
plt.plot(x_value, y_value, label="actual (csv)")      
plt.scatter(x_hat, y_hat, s=3, color='red', label="predicted")   
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("XY space: actual vs predicted")
plt.grid(True)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(x_value, x_value, label="actual")     
plt.scatter(x_value, x_hat, s=3, color='red', label="predicted")  
plt.xlabel("actual x")
plt.ylabel("predicted x")
plt.title("actual x  vs  predicted x")
plt.grid(True)
plt.legend()
plt.show()



plt.figure(figsize=(5,5))
plt.plot(y_value, y_value, label="actual")    
plt.scatter(y_value, y_hat, s=3, color='red', label="predicted")  
plt.xlabel("actual y")
plt.ylabel("predicted y")
plt.title("actual y  vs  predicted y")
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimized Parameters:\nTheta (degrees): {theta_opt}\nM: {M_opt}\nX: {X_opt}")
