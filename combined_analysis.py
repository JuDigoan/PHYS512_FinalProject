import numpy as np
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
    )
)

# Run simulation
t = 0

while t <= 100:
    solar_system5.calculate_all_body_interactions()
    solar_system5.update_all()
    solar_system5.draw_all()

    solar_system4.calculate_all_body_interactions()
    solar_system4.update_all()
    solar_system4.draw_all()

    t = t + 0.5

# Access trajectories after the simulation completes
data_planet1_trajectory = np.array([planets5[0].trajectory])
data_planet2_trajectory = np.array([planets5[1].trajectory])
data_planet3_trajectory = np.array([planets5[2].trajectory])
data_planet4_trajectory = np.array([planets5[3].trajectory])

model_planet1_trajectory = np.array([planets4[0].trajectory])
model_planet2_trajectory = np.array([planets4[1].trajectory])
model_planet3_trajectory = np.array([planets4[2].trajectory])
model_planet4_trajectory = np.array([planets4[3].trajectory])

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

print(np.mean(err_planet1_trajectory))


