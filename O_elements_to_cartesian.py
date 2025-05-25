import numpy as np

def kepler_to_cartesian(a, e, i, Omega, omega, nu, mu):
    """
    Converts Keplerian orbital elements to Cartesian coordinates.
    
    Parameters:
    a     : Semi-major axis (km or any distance unit)
    e     : Eccentricity (unitless)
    i     : Inclination (degrees)
    Omega : Right Ascension of Ascending Node (degrees)
    omega : Argument of Periapsis (degrees)
    nu    : True Anomaly (degrees)
    mu    : Standard gravitational parameter of central body (km^3/s^2 or compatible units)
    
    Returns:
    (x, y, z)   : Cartesian position coordinates (same unit as 'a')
    (vx, vy, vz): Cartesian velocity components (unit depends on 'a' and 'mu')
    """
    # Convert degrees to radians
    i = np.radians(i)
    Omega = np.radians(Omega)
    omega = np.radians(omega)
    nu = np.radians(nu)
    
    # Compute the orbital radius
    r = (a * (1 - e**2)) / (1 + e * np.cos(nu))
    
    # Compute position in perifocal coordinates (PQW frame)
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    z_p = 0
    
    # Compute velocity in perifocal coordinates
    p = a * (1 - e**2)  # Semi-latus rectum
    h = np.sqrt(mu * p) # Specific angular momentum
    vx_p = (-mu / h) * np.sin(nu)
    vy_p = (mu / h) * (e + np.cos(nu))
    vz_p = 0
    
    # Rotation matrices
    R3_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    # Full transformation matrix
    Q_pqw_to_eci = R3_Omega @ R1_i @ R3_omega
    
    # Transform position
    r_eci = Q_pqw_to_eci @ np.array([x_p, y_p, z_p])
    
    # Transform velocity
    v_eci = Q_pqw_to_eci @ np.array([vx_p, vy_p, vz_p])
    
    return r_eci, v_eci

if __name__ == "__main__":
    # Example Usage
    mu_earth = 398600  # km^3/s^2 (gravitational parameter of Earth)
    a = 7000  # km
    e = 0.1
    i = 30  # degrees
    Omega = 40  # degrees
    omega = 60  # degrees
    nu = 45  # degrees

    position, velocity = kepler_to_cartesian(a, e, i, Omega, omega, nu, mu_earth)
    print("Position (km):", position)
    print("Velocity (km/s):", velocity)
