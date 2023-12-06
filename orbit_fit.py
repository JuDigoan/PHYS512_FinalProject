import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from solar_system_3d import SolarSystem, Sun, Planet

def error_calculation(data_trajectory, model_trajectory):
    
    data_planet1_trajectory = np.array([data_trajectory[0]])
    data_planet2_trajectory = np.array([data_trajectory[1]])
    data_planet3_trajectory = np.array([data_trajectory[2]])
    data_planet4_trajectory = np.array([data_trajectory[3]])
    
    model_planet1_trajectory = np.array([model_trajectory[0]])
    model_planet2_trajectory = np.array([model_trajectory[1]])
    model_planet3_trajectory = np.array([model_trajectory[2]])
    model_planet4_trajectory = np.array([model_trajectory[3]])
    
    #Calculate diff
    diff_planet1_trajectory = abs(data_planet1_trajectory - model_planet1_trajectory)
    diff_planet2_trajectory = abs(data_planet2_trajectory - model_planet2_trajectory)
    diff_planet3_trajectory = abs(data_planet3_trajectory - model_planet3_trajectory)
    diff_planet4_trajectory = abs(data_planet4_trajectory - model_planet4_trajectory)
    
    #Calculate square error
    err_planet1_trajectory = np.sqrt(diff_planet1_trajectory[0][:, 0]**2 + diff_planet1_trajectory[0][:, 1]**2 + diff_planet1_trajectory[0][:, 2]**2)
    err_planet2_trajectory = np.sqrt(diff_planet2_trajectory[0][:, 0]**2 + diff_planet2_trajectory[0][:, 1]**2 + diff_planet2_trajectory[0][:, 2]**2)
    err_planet3_trajectory = np.sqrt(diff_planet3_trajectory[0][:, 0]**2 + diff_planet3_trajectory[0][:, 1]**2 + diff_planet3_trajectory[0][:, 2]**2)
    err_planet4_trajectory = np.sqrt(diff_planet4_trajectory[0][:, 0]**2 + diff_planet4_trajectory[0][:, 1]**2 + diff_planet4_trajectory[0][:, 2]**2)
    
    avg_error = np.mean([np.mean(err_planet1_trajectory), np.mean(err_planet2_trajectory), np.mean(err_planet3_trajectory), np.mean(err_planet4_trajectory)])
    return avg_error


# minimize mean square difference
def find_missing_planet_mass(accuracy, initial_mass_guess, data_trajectory):
    errors = []
    count = 0 #initialize count variable
    Mass = initial_mass_guess
    error = accuracy+1
    dM = 0.5
    
    while error>=accuracy:

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
            
            model_planet_trajectories = []
            for n in range(len(planets4)):
                model_planet_trajectories.append(planets4[n].trajectory)

        # Compute the mean square difference
        new_error = error_calculation(data_trajectory, model_planet1_trajectory)
        
        count += 1
        
        print(f"Interation {count}: Mass = {Mass}, MSE = {error}")
        
        if errors == []:
            Mass += dM
        else:
            # Scale dM based on the distance of the error from zero
            dM *= max(0.1, min(1.0, abs(new_error / accuracy)))
            # Update Mass based on the direction of the error change
            if new_error < error:
                Mass += dM
            else:
                Mass -= dM
                
        error = new_error
        errors.append(error)
    
    return Mass, errors, count

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

while t <= time:
    solar_system5.calculate_all_body_interactions()
    solar_system5.update_all()
    solar_system5.draw_all()

    t = t + 0.5

# Access trajectories after the simulation completes
data_planet_trajectories = []
for n in range(len(planets5)-1):
    data_planet_trajectories.append(planets5[n].trajectory)
    
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
