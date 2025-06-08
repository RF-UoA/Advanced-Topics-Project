import numpy as np
import matplotlib.pyplot as plt

# Constants
R_MOON = 1737  # km, radius of the Moon

def orbital_elements_to_position(a, e, i, raan, arg_periapsis, true_anomaly_deg):
    """Converts orbital elements and true anomaly to position in 3D space (km)."""
    # Convert angles to radians
    i = np.radians(i)
    raan = np.radians(raan)
    arg_periapsis = np.radians(arg_periapsis)
    true_anomaly = np.radians(true_anomaly_deg)
    
    # Distance from the focus to the point
    r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
    
    # Position in perifocal coordinate system
    x_p = r * np.cos(true_anomaly)
    y_p = r * np.sin(true_anomaly)
    z_p = 0
    
    # Rotation matrices
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_argp = np.cos(arg_periapsis)
    sin_argp = np.sin(arg_periapsis)
    
    # Combined rotation matrix from perifocal to inertial frame
    R = np.array([
        [cos_raan*cos_argp - sin_raan*sin_argp*cos_i, -cos_raan*sin_argp - sin_raan*cos_argp*cos_i, sin_raan*sin_i],
        [sin_raan*cos_argp + cos_raan*sin_argp*cos_i, -sin_raan*sin_argp + cos_raan*cos_argp*cos_i, -cos_raan*sin_i],
        [sin_argp*sin_i, cos_argp*sin_i, cos_i]
    ])
    
    r_vec = np.dot(R, np.array([x_p, y_p, z_p]))
    return r_vec

# Sample input
a = 1837  # km
e = 0.01
i = 90  # degrees
raan = 45  # degrees
arg_periapsis = 0  # degrees
true_anomalies = np.linspace(0, 360, 100)  # degrees

# Test sample input
a = 7000  # km
e = 0.1
i = 30  # degrees
raan = 40  # degrees
arg_periapsis = 60  # degrees
true_anomalies = np.linspace(0, 360, 100)  # degrees

# Another sample input
# [6804.126374327793, 0.05456685811147543, 21.176296155842053, 40.92277491914602, 60.11405134964658]
a = 6804.126374327793  # km
e = 0.05456685811147543
i = 21.176296155842053  # degrees
raan = 40.92277491914602  # degrees
arg_periapsis = 60.11405134964658  # degrees
true_anomalies = np.array([28.10520262, 45.9108059, -299.55895686, 73.75325483, 1166.56469151, 260.68328943, 247.55313949, 233.37434459, 216.54411062, 539.9999794])

# Compute orbit points
orbit_points = np.array([orbital_elements_to_position(a, e, i, raan, arg_periapsis, nu) for nu in true_anomalies])

# Calculate limits
max_range = np.array([
    orbit_points[:, 0].max() - orbit_points[:, 0].min(),
    orbit_points[:, 1].max() - orbit_points[:, 1].min(),
    orbit_points[:, 2].max() - orbit_points[:, 2].min()
]).max() / 2.0

mid_x = (orbit_points[:, 0].max() + orbit_points[:, 0].min()) * 0.5
mid_y = (orbit_points[:, 1].max() + orbit_points[:, 1].min()) * 0.5
mid_z = (orbit_points[:, 2].max() + orbit_points[:, 2].min()) * 0.5

# Create figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], label='Orbit')
ax.scatter(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], s=5, color='red', label='True Anomaly Points')

# Draw Moon as a sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = R_MOON * np.outer(np.cos(u), np.sin(v))
y = R_MOON * np.outer(np.sin(u), np.sin(v))
z = R_MOON * np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x, y, z, color='gray', alpha=0.5)

# Set axis limits to enforce equal aspect ratio
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Labels and formatting
ax.set_title("Lunar Orbit Visualization")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.legend()
ax.set_box_aspect([1,1,1])
plt.tight_layout()

plt.show()
