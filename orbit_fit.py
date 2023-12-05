import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from solar_system_3d import SolarSystem, Sun, Planet

# minimize mean square difference
def function(Mass):
    # create a solar system with the existing planets and a new one with variable mass
    solar_system = SolarSystem(400, projection_2d=True)
    
    sun = Sun(solar_system)

    planets = (
        Planet(solar_system, position=(150, 50, 0), velocity=(0, 5, 5)),
        Planet(solar_system, mass=20, position=(100, 50, 150), velocity=(5, 0, 0)),
        Planet(solar_system, mass=30, position=(200, -50, 150), velocity=(5, 0, 0)),
        Planet(solar_system, mass=15, position=(100, -100, 75), velocity=(5, 0, 0)),
        Planet(solar_system, mass=Mass, position=(0, 0, 0), velocity=(0, 0, 0))  # variable mass
    )

    # run simulation
    t = 0

    while t <= 100:
        solar_system.calculate_all_body_interactions()
        solar_system.update_all()
        t += 0.5

    # Compute the mean square difference
    mse = 0.0
    for i in range(4):  # Compare the trajectories of the first four planets
        planet_trajectory = np.array(planets[i].trajectory)
        observed_trajectory = np.array([planet1_trajectory, planet2_trajectory, planet3_trajectory, planet4_trajectory])[i]
        mse += np.sum((planet_trajectory - observed_trajectory) ** 2)

    return mse

# Initial guess for the mass of the fifth planet
initial_mass_guess = 10.0

# Optimize the mass to minimize the Mean Square Difference
result = minimize(function, initial_mass_guess, method='L-BFGS-B')

# Extract the optimized mass
optimized_mass = result.x[0]

print("Optimized Mass:", optimized_mass)
