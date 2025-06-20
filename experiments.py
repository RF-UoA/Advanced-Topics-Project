import numpy as np
import matplotlib.pyplot as plt
from moon_orbit import generate_data
from solver import optimise_orbit
from read_input import read_data_from_df

# Add random seed
np.random.seed(1)

def add_noise_to_points_2D(points_2D, std_dev):
    """
    Adds normally distributed noise to each 2D point in all frames.

    Args:
        points_2D (list of list of np.ndarray): Outer list is frames, inner list is 2D points (shape (2,) or (2,1)).
        std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        list: New points_2D structure with noise added.
    """
    noisy_points_2D = []
    for frame in points_2D:
        noisy_frame = []
        for pt in frame:
            pt = np.array(pt).flatten()  # Ensure shape (2,)
            noise = np.random.normal(0, std_dev, size=pt.shape)
            noisy_pt = pt + noise
            noisy_frame.append(noisy_pt)
        noisy_points_2D.append(noisy_frame)
    return noisy_points_2D

# Use source code to generate a DataFrame with synthetic data (ground truth)
orbit = np.array([1837, 0.01, 90.0, 45.0, 0.0, 0.0])  # Example orbital elements
timestack = np.arange(0, 5000, 500) # Max TA ~ 230 degrees, so 5000 seconds is a good range
df = generate_data(orbit, timestack)

# Optimise the orbit based on the generated data
K_GT = [60, 1024, 1024]  # Example intrinsic values (K Ground-Truth matrix)

# Read the data from the DataFrame generated by generate_data
points_3D, points_2D, extrinsics = read_data_from_df(df)

# EXPERIMENT 1: Optimise the orbit with a guess for the intrinsic matrix K
K_guess = [50, 1024, 1024]
guess = [1800, 0.05, 94, 43, 0, 0, 230, *K_guess]  
errors = []

optimised_orbit = optimise_orbit(guess, points_3D, points_2D, extrinsics, K=None, errors=errors)
print("Optimised Orbit:", optimised_orbit)

# Plot the total reprojection error over iterations
plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Total Reprojection Error')
plt.yscale('log')  # Use log scale for better visibility
plt.show()

# Plot %difference between optimised and ground truth K matrix
percent_diff = (np.array(list(optimised_orbit.values())[:5]) - orbit[:5])
percent_diff = np.append(percent_diff, (np.array(optimised_orbit['True Anomalies'][0]) - orbit[5]))
labels = ['a', 'e', 'i', 'RAAN', 'ω', 'Initial TA']
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, percent_diff, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
plt.ylabel('Difference')
plt.axhline(0, color='gray', linestyle='--')
plt.show()

# Plot difference between optimised and ground truth K matrix
diff = np.array(list(optimised_orbit['Intrinsic Parameters'])) - np.array(K_GT)
guess_diff = np.array(K_guess) - np.array(K_GT)
labels_guess = ['FOV', 'Width', 'Height']
labels_k = ['FOV', 'Width', 'Height']
plt.figure(figsize=(8, 5))
bars_guess = plt.bar(labels_guess, guess_diff, color='orange', label='Guess Accuracy')
bars_k = plt.bar(labels_k, diff, color='skyblue', label='Optimised K Accuracy')
plt.ylabel('Difference')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.show()

# EXPERIMENT 2: Add noise to the 2D points and optimise the orbit
std_dev = range(0, 10, 2)  # Standard deviation of noise to add
final_reprojection_errors = []

for std in std_dev:
    noisy_points_2D = add_noise_to_points_2D(points_2D, std)
    errors = []  # Reset errors for each noise level
    optimised_orbit_noisy = optimise_orbit(guess, points_3D, noisy_points_2D, extrinsics, K=None, errors=errors)
    
    print(f"Optimised Orbit with noise std {std}:", optimised_orbit_noisy)
    
    # Plot the total reprojection error over iterations for noisy data
    final_reprojection_errors.append(errors[-1])  # Store the last error value
    plt.plot(errors, label=f'Std Dev: {std}')
plt.xlabel('Iteration')
plt.ylabel('Total Reprojection Error')
plt.yscale('log')  # Use log scale for better visibility
plt.legend()
plt.show()

# Plot the final reprojection errors for different noise levels
plt.figure(figsize=(8, 5))
plt.bar(std_dev, final_reprojection_errors, color='skyblue')
plt.xlabel('Standard Deviation of Noise')
plt.ylabel('Final Reprojection Error')
plt.xticks(std_dev)
plt.show()

# Plot %difference between optimised and ground truth K matrix
percent_diff = (np.array(list(optimised_orbit_noisy.values())[:5]) - orbit[:5])
percent_diff = np.append(percent_diff, (np.array(optimised_orbit_noisy['True Anomalies'][0]) - orbit[5]))
labels = ['a', 'e', 'i', 'RAAN', 'ω', 'Initial TA']
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, percent_diff, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
plt.ylabel('Difference')
plt.axhline(0, color='gray', linestyle='--')
plt.show()