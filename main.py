import numpy as np
from scipy.optimize import curve_fit

# Define the Gompertz growth function
def gompertz_model(t, A, k, tip):
    return A * np.exp(-np.exp(-k * (t - tip)))

# Example data (replace these with your actual age and weight data)
ages = np.array([28, 56, 84, 112])  # in days (4, 8, 12, and 16 weeks)
weights = np.array([140, 348, 610, 940])  # corresponding weights in grams

# Fit the model to find the best values of A, k, and tip
popt, _ = curve_fit(gompertz_model, ages, weights, p0=[1000, 0.03, 15])
A, k, tip = popt

print("Estimated A:", A)
print("Estimated k:", k)
print("Estimated tip:", tip)

import numpy as np

def gompertz_growth_model(t, A, k, tip):
    """
    Calculate the predicted body weight (BW) at age t using the Gompertz growth model.

    Parameters:
    - t (float): Age in days (or any consistent time unit).
    - A (float): Asymptotic adult body weight.
    - k (float): Growth rate constant.
    - tip (float): Age at the inflection point of the growth curve.

    Returns:
    - float: Predicted body weight at age t.
    """
    BW = A * np.exp(-np.exp(-k * (t - tip)))
    return BW

bw8 = gompertz_growth_model(t=32,A=1000,k=0.06381357088611105,tip=32)
bw12 = gompertz_growth_model(t=48,A=1000,k=0.06381357088611105,tip=32)
bw16 = gompertz_growth_model(t=64,A=1000,k=0.06381357088611105,tip=32)
print(bw8)
print(bw12)
print(bw16)

# bw8 = gompertz_growth_model(t=32,A=1000,k=0.004,tip=25)
# bw12 = gompertz_growth_model(t=48,A=1000,k=0.023,tip=15)
# bw16 = gompertz_growth_model(t=64,A=1000,k=0.05,tip=7)