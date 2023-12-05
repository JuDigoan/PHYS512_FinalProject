#More complex_solar_system.py
import numpy as np
import matplotlib.pyplot as plt

from solar_system_3d import SolarSystem, Sun, Planet

solar_system = SolarSystem(400, projection_2d=True)

sun = Sun(solar_system)

planets = (
    Planet(
        solar_system,
        position=(150, 50, 0),
        velocity=(0, 5, 5),
    ),
    Planet(
        solar_system,
        mass=20,
        position=(100, 50, 150),
        velocity=(5, 0, 0)
    ),
    Planet(
        solar_system,
        mass=30,
        position=(200, -50, 150),
        velocity=(5, 0, 0)
    ),
    Planet(
        solar_system,
        mass=15,
        position=(100, -100, 75),
        velocity=(5, 0, 0)
    ),
    Planet(
        solar_system,
        mass=5,
        position=(75, -70, 150),
        velocity=(5, 0, 0)
    )
)

# Set the 'hidden' flag for the second planet
#planets[3].hidden = True


# Run simulation
t = 0

while t<=100:
    solar_system.calculate_all_body_interactions()
    solar_system.update_all()
    solar_system.draw_all()
    t=t+0.5

# Access trajectories after the simulation completes
planet1_trajectory = np.array([planets[1].trajectory])
print(planet1_trajectory)