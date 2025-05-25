import numpy as np
import scipy.optimize
from O_elements_to_cartesian import kepler_to_cartesian
from read_input import get_correspondences
import ast
import re

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ONLY WORKS WITH SCIPY VERSION 1.10.1 GIVES A CERTIFICATE ERROR AND REQUIRES BYPASS
from orbdtools import ArcObs, OrbeleTrans, FrameTrans, KeprvTrans

mu_earth = 398600
# add mu_moon
mu_moon = 4902.800066

def project_points(K, R, t, points_3D):
    """Projects 3D points onto the image plane using the intrinsic (K), rotation (R), and translation (t)."""

    # Ensure points_3D is a numpy array
    points_3D = np.array(points_3D)  # Ensure it's a numpy array
    # Divide points_3D by 1000 to convert from km to m
    points_3D = points_3D / 1000.0  # Convert from km to m

    points_3D = np.array(points_3D).T  # Shape (3, N)
    projected_points = K @ (R @ points_3D + t)

    projected_points /= projected_points[2]  # Normalize by depth (z)
    return projected_points[:2].T  # Return (N,2) 2D coordinates

def project_points2(K, extrinsic, points_3D):
    """
    Projects 3D points onto the image plane using the intrinsic matrix (K) and extrinsic matrix.
    
    Args:
        K (np.ndarray): Intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Extrinsic matrix of shape (3, 4) containing rotation and translation.
        points_3D (np.ndarray): 3D points of shape (N, 3).
    
    Returns:
        np.ndarray: Projected 2D points of shape (N, 2).
    """
    # Ensure points_3D is a numpy array
    points_3D = np.array(points_3D)  # Ensure it's a numpy array
    # Divide points_3D by 1000 to convert from km to m
    points_3D = points_3D / 1000.0  # Convert from km to m

    print("Points to be projected:", points_3D)

    # Convert 3D points to homogeneous coordinates
    points_3D_homogeneous = np.hstack([points_3D, np.ones((points_3D.shape[0], 1))])  # Shape (N, 4)

    # Project 3D points onto the image plane
    projected_points = K @ extrinsic @ points_3D_homogeneous.T  # Shape (3, N)

    # Normalize by depth (z-coordinate)
    projected_points /= projected_points[2]  # Normalize by z-coordinate

    print("Projected points:", projected_points[:2].T)  # Shape (N, 3)
    # Extract 2D points
    return projected_points[:2].T  # Shape (N, 2)

def project_points3(K, R, t, points_3D):
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

def project_points4(K, R, tvec, points_3D):
    """
    Projects 3D points onto the image plane using the intrinsic (K), rotation (R), and camera position in world frame (tvec).
    
    Args:
        K (np.ndarray): Intrinsic matrix of shape (3, 3).
        R (np.ndarray): Rotation matrix of shape (3, 3).
        tvec (np.ndarray): Camera position in the world frame of shape (3, 1).
        points_3D (np.ndarray): 3D points of shape (N, 3).
    
    Returns:
        np.ndarray: Projected 2D points of shape (N, 2).
    """
    # Ensure points_3D is a numpy array
    points_3D = np.array(points_3D)  # Ensure it's a numpy array
    # Divide points_3D by 1000 to convert from km to m
    points_3D = points_3D / 1000.0  # Convert from km to m

    # Compute the translation vector in the camera's coordinate system
    t = -R @ tvec  # Transform tvec (world frame) to t (camera frame)

    # Transpose points_3D to shape (3, N)
    points_3D = points_3D.T  # Shape (3, N)

    # Apply the pinhole camera model: K * (R * points_3D + t)
    projected_points = K @ (R @ points_3D + t)

    # Normalize by depth (z-coordinate)
    projected_points /= projected_points[2]  # Normalize by depth (z)

    # Return the 2D points (N, 2)
    return projected_points[:2].T

def rotation_matrix_from_vector(rvec):
    """Computes a rotation matrix from a rotation vector using Rodrigues' formula."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def graph_2d_points(reprojected_points):
    """Visualizes the reprojected 2D points on a 2D plot."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(reprojected_points[:, 0], reprojected_points[:, 1], c='r', marker='o')
    plt.title('Reprojected 2D Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.axis('equal')
    plt.show()

def graph_2_sets_of_2d_points(points_2D, reprojected_points):
    """Visualizes the original and reprojected 2D points on a 2D plot."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(points_2D[:, 0], points_2D[:, 1], c='b', marker='o', label='Original Points')
    plt.scatter(reprojected_points[:, 0], reprojected_points[:, 1], c='r', marker='x', label='Reprojected Points')
    plt.title('Original vs Reprojected 2D Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def create_extrinsic_matrix(plane_normal, radius):
    # Ensure the plane normal is a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Camera's z-axis is the opposite of the plane normal
    z_axis = -plane_normal

    # Determine an up vector. If the z-axis is not parallel to [0, 1, 0], use [0, 1, 0] as the up vector.
    # Otherwise, use [1, 0, 0].
    if np.abs(np.dot(z_axis, [0, 1, 0])) != 1:
        up_vector = [0, 1, 0]
    else:
        up_vector = [1, 0, 0]

    # Camera's x-axis
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Camera's y-axis
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix
    R = np.array([x_axis, y_axis, z_axis]).T
        
    # Translation vector (camera's position in world coordinates)
    t = plane_normal * radius

    # Extrinsic matrix
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ t  # Convert world position to camera-centric position
    # extrinsic[3, 3] = 1

    return extrinsic

def create_extrinsic_matrix2(plane_normal, radius):
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


######################################################

def reprojection_error(params, K, R_list, points_3D_list, points_2D_list):
    """
    Computes the sum of squared distances between projected and detected 2D points
    across multiple frames, using individual rotation matrices for each frame.
    """
    # Extract shared orbital elements and per-frame true anomalies
    orbital_elements = params[:5]  # Shared across all frames
    true_anomalies = params[5:]    # One true anomaly per frame

    total_error = 0
    for i, (points_3D, points_2D, R) in enumerate(zip(points_3D_list, points_2D_list, R_list)):
        # Compute translation vector (x, y, z) for the current frame
        tvec = kepler_to_cartesian(*orbital_elements, true_anomalies[i], mu_moon)[0]

        # Use external library to convert orbital elements to position vector
        coe_oe = np.array([orbital_elements[0], orbital_elements[1], orbital_elements[2], orbital_elements[3], orbital_elements[4], true_anomalies[i]])
        rv = KeprvTrans.coe2rv(coe_oe, mu_moon)
        tvec = rv[:3]
        # tvec = tvec.reshape(3, 1) # if tvec is used directly in projected_points

        # Normalize tvec and compute its magnitude (ASSUME tvec IS CORRECT)
        plane_normal = tvec / np.linalg.norm(tvec)  # Unit vector
        radius = np.linalg.norm(tvec)              # Magnitude

        # Create the extrinsic matrix
        extrinsic = create_extrinsic_matrix2(plane_normal, radius)

        # Extract the translation vector from the extrinsic matrix
        t = extrinsic[:, 3].reshape(3, 1)

        # print(t.shape)
        # exit()
        # Project 3D points to 2D using the rotation matrix for the current frame
        # projected_points = project_points(K, R, t, points_3D)
        projected_points = project_points3(K, R, t, points_3D)  # Using rotation matrix and translation vector

        # project points 2 uses the extrinsic matrix instead of R and tvec. 
        # projected_points = project_points2(K, R, points_3D)  # Using extrinsic matrix

        # Accumulate reprojection error
        total_error += np.sum((projected_points - points_2D) ** 2)
        # print(total_error)
        # graph_2_sets_of_2d_points(points_2D, projected_points)  # Visualize original vs reprojected points
        # exit()

    # print("3d points:", points_3D)
    # print("2d points:", points_2D)
    # print("Reprojected points:", projected_points)
    # print("Orbital elements:", orbital_elements)
    # print("Total reprojection error:", total_error)
    # graph_2d_points(projected_points)  # Visualize the reprojected points
    # graph_2_sets_of_2d_points(points_2D, projected_points)  # Visualize original vs reprojected points
    # exit()
    return total_error

def solve_pnp(points_3D_list, points_2D_list, K, R_list):
    """
    Solves the PnP problem for multiple frames using nonlinear optimization.
    Optimizes for the orbital elements (shared across frames) and true anomalies (per frame),
    using individual rotation matrices for each frame.
    """
    num_frames = len(points_3D_list)

    # Initial guess for orbital elements and true anomalies
    initial_orbital_elements = np.array([1700, 0.02, 92, 46, 0])  # Example initial guess
    initial_true_anomalies = np.linspace(0, 230, num_frames)       # Spread true anomalies evenly
    initial_guess = np.hstack((initial_orbital_elements, initial_true_anomalies))

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
        args=(K, R_list, points_3D_list, points_2D_list),
        method='L-BFGS-B'
        # options={'disp': False, 'maxiter': 1500}
    )

    # Extract optimized parameters
    optimized_orbital_elements = result.x[:5]
    optimized_true_anomalies = result.x[5:]

    return optimized_orbital_elements, optimized_true_anomalies

######################################################

# def reprojection_error(params, K, points_3D, points_2D):
#     """Computes the sum of squared distances between projected and detected 2D points."""
#     rvec, tvec = params[:3], kepler_to_cartesian(*params[3:], mu_moon)[0]  # Extract rotation and translation
#     R = rotation_matrix_from_vector(rvec)  # Compute rotation matrix
#     tvec = tvec.reshape(3, 1)
#     projected_points = project_points(K, R, tvec, points_3D)
#     return np.sum((projected_points - points_2D) ** 2)

# def solve_pnp(points_3D, points_2D, K):
#     """Solves the PnP problem using nonlinear optimization."""
#     # Initial guess (assume no rotation, and translation at mean depth)
#     initial_rvec = np.zeros(3)
#     initial_tvec = np.array([7000, 0.1, 30, 40, 60, 45])  # Example initial guess
#     initial_guess = np.hstack((initial_rvec, initial_tvec))
    
#     # Optimize
#     result = scipy.optimize.minimize(
#         reprojection_error,
#         initial_guess,
#         args=(K, points_3D, points_2D),
#         method='Powell'
#     )
    
#     # Extract optimized parameters
#     optimized_rvec = result.x[:3]
#     optimized_tvec = result.x[3:].reshape(6, 1)
#     optimized_R = rotation_matrix_from_vector(optimized_rvec)
    
#    return optimized_R, optimized_tvec

def get_intrinsic(calibration_file):
    f = open(calibration_file, 'r')
    lines = f.readlines()
    calibration = lines[1].split(',')
    fov = int(calibration[0])
    # fx = int(calibration[1])
    # fy = int(calibration[2])
    image_width = int(calibration[3])
    image_height = int(calibration[4])

    fov = fov*np.pi/180
    fx = image_width/(2*np.tan(fov/2)) # Conversion from fov to focal length
    fy = image_height/(2*np.tan(fov/2)) # Conversion from fov to focal length
    cx = image_width/2
    cy = image_height/2

    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

import numpy as np

def strings_to_matrices(string_list):
    """
    Converts a list of matrix-like strings into a list of NumPy arrays with correct shape.
    """
    matrices = []
    for s in string_list:
        # Flatten numbers
        clean = s.replace('[', '').replace(']', '').replace('\n', ' ')
        numbers = np.fromstring(clean, sep=' ')

        # Count the number of rows = number of lines with a leading bracket
        # This avoids counting the outermost bracket
        num_rows = s.count('\n') + 1  # since rows are newline-separated

        # Reshape
        matrix = numbers.reshape((num_rows, -1))
        matrices.append(matrix)
    return matrices


# Example usage
if __name__ == "__main__":
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Example intrinsic matrix
    points_3D = np.array([[0, 0, 10], [1, 1, 10], [-1, -1, 10], [1, -1, 10], [-1, 1, 10]])
    points_2D = np.array([[320, 240], [370, 260], [270, 220], [370, 220], [270, 260]])

    points_3D, points_2D, _, extrinsics = get_correspondences()  # Uncomment to use actual data
    K = get_intrinsic('data/calibration.txt')
    # points_3D = np.vstack([obj.reshape(-1, 3) for obj in points_3D])  # Shape (N, 3)
    # points_2D = np.vstack([img.reshape(-1, 2) for img in points_2D])  # Shape (N, 2)
    # attitudes = strings_to_matrices(attitudes)  # Convert string matrices to NumPy arrays
    
    # Obtain the rotation matrices from the extrinsics
    attitudes = [attitude[:, :3] for attitude in extrinsics]  # Keep only the rotation part

    # Run the PnP solver
    o_elements, anomalies = solve_pnp(points_3D, points_2D, K, attitudes)

    print("Optimised Semimajor Axis:", o_elements[0])
    print("Optimized Eccentricity:", o_elements[1])
    print("Optimized Inclination:", o_elements[2])
    print("Optimized RAAN:", o_elements[3])
    print("Optimized Argument of Perigee:", o_elements[4])
    print("Optimized True Anomalies:\n", anomalies)

    # print("Optimized Rotation Matrix:", R)
    # print("Optimized Translation Vector:", t)
