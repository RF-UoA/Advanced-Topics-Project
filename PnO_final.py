import numpy as np
import scipy.optimize
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ONLY WORKS WITH SCIPY VERSION 1.10.1 GIVES A CERTIFICATE ERROR AND REQUIRES BYPASS
from orbdtools import KeprvTrans

# Local package imports
from read_input import get_correspondences
from O_elements_to_cartesian import kepler_to_cartesian

def project_points(K, R, t, points_3D):
    """
    Projects 3D points onto the image plane using the intrinsic (K), rotation (R), and translation (t).
    
    Args:
        K (np.ndarray): Intrinsic matrix of shape (3, 3).
        R (np.ndarray): Rotation matrix of shape (3, 3).
        t (np.ndarray): Translation vector of shape (3, 1).
        points_3D (np.ndarray): 3D points of shape (N, 3).
    
    Returns:
        np.ndarray: Projected 2D points of shape (N, 2).
    """
    # Ensure points_3D is a numpy array
    points_3D = np.array(points_3D)  # Ensure it's a numpy array
    # Divide points_3D by 1000 to convert from km to m
    points_3D = points_3D / 1000.0  # Convert from km to m

    # Transpose points_3D to shape (3, N)
    points_3D = points_3D.T  # Shape (3, N)

    # Apply the pinhole camera model: K * (R * points_3D + t)
    projected_points = K @ (R @ points_3D + t)

    # Normalize by depth (z-coordinate)
    projected_points /= projected_points[2]  # Normalize by depth (z)

    # Return the 2D points (N, 2)
    return projected_points[:2].T

def create_extrinsic_matrix(plane_normal, radius):
    """
    Constructs the camera extrinsic matrix from a plane normal (view direction)
    and a distance from the origin (radius).
    
    Args:
        plane_normal (np.ndarray): Unit vector pointing from the Moon to the spacecraft.
        radius (float): Distance from the Moon center to the spacecraft (in meters).
    
    Returns:
        np.ndarray: Extrinsic matrix (3x4) to transform world coordinates to camera coordinates.
    """
    # Ensure unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Camera looks along -z in its own coordinate system
    z_axis = -plane_normal
    
    # Choose an up vector (not parallel to z_axis)
    if abs(np.dot(z_axis, [0, 1, 0])) < 0.99:
        up = np.array([0, 1, 0])
    else:
        up = np.array([1, 0, 0])

    # Camera x and y axes (right and up)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Build rotation matrix (world to camera)
    R = np.vstack((x_axis, y_axis, z_axis)).T  # Shape (3, 3)

    # Camera position in world coordinates
    C = plane_normal * radius  # From Moon center to camera position

    # Translation vector: t = -R * C
    t = -R @ C

    # Build extrinsic matrix
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic

def reprojection_error(params, R_list, points_3D_list, points_2D_list):
    """
    Computes the sum of squared distances between projected and detected 2D points
    across multiple frames, using individual rotation matrices for each frame.
    """
    # Extract shared orbital elements and per-frame true anomalies
    orbital_elements = params[:5]  # Shared across all frames
    true_anomalies = params[8:]    # One true anomaly per frame
    intrinsic_vals = params[5:8]   # Intrinsic parameters (K matrix)

    # Extract intrinsic parameters
    fov, img_width, img_height = intrinsic_vals
    fx = img_width / (2 * np.tan(fov / 2))
    fy = img_height / (2 * np.tan(fov / 2))
    cx = img_width / 2
    cy = img_height / 2
    K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])  # Intrinsic matrix

    total_error = 0
    for i, (points_3D, points_2D, R) in enumerate(zip(points_3D_list, points_2D_list, R_list)):
        # Compute translation vector (x, y, z) for the current frame
        mu_moon = 4902.800066
        tvec = kepler_to_cartesian(*orbital_elements, true_anomalies[i], mu_moon)[0]

        # Use external library to convert orbital elements to position vector
        # coe_oe = np.array([orbital_elements[0], orbital_elements[1], orbital_elements[2], orbital_elements[3], orbital_elements[4], true_anomalies[i]])
        
        # rv = KeprvTrans.coe2rv(coe_oe, mu_moon)
        # tvec = rv[:3]

        # Normalize tvec and compute its magnitude (ASSUME tvec IS CORRECT)
        plane_normal = tvec / np.linalg.norm(tvec)  # Unit vector
        radius = np.linalg.norm(tvec)              # Magnitude

        # Create the extrinsic matrix
        extrinsic = create_extrinsic_matrix(plane_normal, radius)

        # Extract the translation vector from the extrinsic matrix
        t = extrinsic[:, 3].reshape(3, 1)

        # Project the 3D points using the intrinsic matrix (K), rotation matrix (R), and translation vector (t)
        projected_points = project_points(K, R, t, points_3D)  # Using rotation matrix and translation vector

        # Accumulate reprojection error
        total_error += np.sum((projected_points - points_2D) ** 2)

    return total_error

def solve_pnp(points_3D_list, points_2D_list, R_list):
    """
    Solves the PnP problem for multiple frames using nonlinear optimization.
    Optimizes for the orbital elements (shared across frames) and true anomalies (per frame),
    using individual rotation matrices for each frame.
    """
    num_frames = len(points_3D_list)

    # Initial guess for orbital elements and true anomalies
    initial_orbital_elements = np.array([1837, 0.01, 90, 45, 0])  # Example initial guess
    initial_true_anomalies = np.linspace(0, 230, num_frames)       # Spread true anomalies evenly
    initial_intrinsic_vals = np.array([60, 1024, 1024])  # Example intrinsic values (K matrix)
    initial_guess = np.hstack((initial_orbital_elements, initial_intrinsic_vals, initial_true_anomalies))

    # Add K to initial guess

    # Define bounds for the optimization
    bounds = [
        (1837, 1837),  # Semi-major axis bounds
        (0.01, 0.01),        # Eccentricity bounds
        (90, 90),      # Inclination bounds (degrees)
        (45, 45),      # RAAN bounds (degrees)
        (0, 0)       # Argument of Perigee bounds (degrees)
    ]
    # Add bounds for true anomalies (one per frame)
    bounds.extend([(0, 240)] * num_frames)

    # Optimize
    result = scipy.optimize.minimize(
        reprojection_error,
        initial_guess,
        args=(R_list, points_3D_list, points_2D_list),
        method='L-BFGS-B'
        # options={'disp': True, 'maxiter': 1500}
    )

    # Extract optimized parameters
    optimized_orbital_elements = result.x[:5]
    optimized_true_anomalies = result.x[8:]
    optimized_intrinsic_vals = result.x[5:8]

    return optimized_orbital_elements, optimized_true_anomalies, optimized_intrinsic_vals

if __name__ == "__main__":
    
    # Get the 3D points, 2D points, and extrinsics
    points_3D, points_2D, _, extrinsics = get_correspondences()
    
    # Obtain the rotation matrices from the extrinsics
    attitudes = [attitude[:, :3] for attitude in extrinsics]  # Keep only the rotation part

    # Run the PnP solver
    o_elements, anomalies, intrinsics = solve_pnp(points_3D, points_2D, attitudes)

    print("Optimised Semimajor Axis:", o_elements[0])
    print("Optimized Eccentricity:", o_elements[1])
    print("Optimized Inclination:", o_elements[2])
    print("Optimized RAAN:", o_elements[3])
    print("Optimized Argument of Perigee:", o_elements[4])
    print("Optimized True Anomalies:\n", anomalies)

    print("Optimized Intrinsic Parameters (K):\n", intrinsics)
