import numpy as np
from scipy.optimize import minimize

# Define the Gompertz growth model
def gompertz_growth_model(t, A, k, tip):
    return A * np.exp(-np.exp(-k * (t - tip)))

# Target values
targets = [378.18, 626.17, 943.80]
times = [32, 48, 64]

# Objective function to minimize
def objective(params):
    k, tip = params
    return sum((gompertz_growth_model(t, 1000, k, tip) - target) ** 2 for t, target in zip(times, targets))

# Initial guesses for k and tip
initial_guess = [0.01, 20]
result = minimize(objective, initial_guess)

k_opt, tip_opt = result.x
print(f"Optimal k: {k_opt}, Optimal tip: {tip_opt}")
