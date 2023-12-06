import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from solar_system_3d import SolarSystem, Sun, Planet

solar_system5 = SolarSystem(400, projection_2d=True)

sun5 = Sun(solar_system5)

planets5 = (
    Planet(
        solar_system5,
        position=(150, 50, 0),
        velocity=(0, 5, 5),
    ),
    Planet(
        solar_system5,
        mass=20,
        position=(100, 50, 150),
        velocity=(5, 0, 0)
    ),
    Planet(
        solar_system5,
        mass=30,
        position=(200, -50, 150),
        velocity=(5, 0, 0)
    ),
    Planet(
        solar_system5,
        mass=15,
        position=(100, -100, 75),
        velocity=(5, 0, 0)
    ),
    Planet(
        solar_system5,
        mass=10,
        position=(75, -70, 150),
        velocity=(5, 0, 0)
    )
)

# Set the 'hidden' flag for the second planet
planets5[4].hidden = True

# Run simulation

t = 0

while t <= 100:
    solar_system5.calculate_all_body_interactions()
    solar_system5.update_all()
    solar_system5.draw_all()

    t = t + 0.5

# Access trajectories after the simulation completes
data_planet1_trajectory = np.array([planets5[0].trajectory])


def error_calculation(data_planet1_trajectory, model_planet1_trajectory):
    
    diff_planet1_trajectory = abs(data_planet1_trajectory - model_planet1_trajectory)

    #Calculate square error
    err_planet1_trajectory = np.sqrt(diff_planet1_trajectory[0][:, 0]**2 + diff_planet1_trajectory[0][:, 1]**2 + diff_planet1_trajectory[0][:, 2]**2)

    return (np.mean(err_planet1_trajectory))


# minimize mean square difference
def find_missing_planet_mass(accuracy, initial_mass_guess, data_planet1_trajectory):
    errors = []
    count = 0 #initialize count variable
    Mass = initial_mass_guess
    
    def function(Mass):
        nonlocal count  # Declare count as a nonlocal variable
        nonlocal errors
        # create a solar system with the existing planets and a new one with variable mass
        solar_system4 = SolarSystem(400, projection_2d=True)
        
        sun4 = Sun(solar_system4)
        
        planets4 = (
            Planet(
                solar_system4,
                position=(150, 50, 0),
                velocity=(0, 5, 5),
            ),
            Planet(
                solar_system4,
                mass=20,
                position=(100, 50, 150),
                velocity=(5, 0, 0)
            ),
            Planet(
                solar_system4,
                mass=30,
                position=(200, -50, 150),
                velocity=(5, 0, 0)
            ),
            Planet(
                solar_system4,
                mass=15,
                position=(100, -100, 75),
                velocity=(5, 0, 0)
            ),
            Planet(solar_system4,
                   mass=Mass,  # variable mass
                   position=(75, -70, 150),
                   velocity=(5, 0, 0))  
        )

        # run simulation
        t = 0
        
        while t <= 100:
        
            solar_system4.calculate_all_body_interactions()
            solar_system4.update_all()
            solar_system4.draw_all()
        
            t = t + 0.5
            
        model_planet1_trajectory = np.array([planets4[0].trajectory])

        # Compute the mean square difference
        error = error_calculation(data_planet1_trajectory, model_planet1_trajectory)
        
        errors.append(error)
        count += 1
        
        print(f"Interation {count}: Mass = {Mass}, MSE = {error}")

        return error
        
    # Optimize the mass to minimize the Mean Square Difference
    result = minimize(function, initial_mass_guess, method='L-BFGS-B', tol=accuracy)
    
    # Extract the optimized mass
    optimized_mass = result.x[0]

    if function(Mass) <= accuracy:
        return optimized_mass, errors, count
    
    else:
        print("Optimization failed or resulted in an unreasonable mass.")
        return initial_mass_guess
    
    
# Initial guess for the mass of the fifth planet
initial_mass_guess = 8
accuracy = 1e-6
results = find_missing_planet_mass(accuracy, initial_mass_guess, data_planet1_trajectory)
missing_mass = results[0]
print("Missing planet's mass:", missing_mass)

errors_list, count_list = results[1], results[2]
count_list = range(count_list)

plt.figure()
plt.plot(count_list, errors_list, linestyle='-', color='b')
plt.xlabel('Iterations')
plt.ylabel('Error [mean square difference]')
plt.title('Optimization Process')
plt.show()
