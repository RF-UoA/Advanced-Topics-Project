# Advanced-Topics-Project - Orbital Camera Calibration

## Overview

This project explores a unique algorithm to achieve camera calibration and orbit determination for a spacecraft in a lunar orbit. 

## Instructions

Experimentation figures generated for academic report can be produced using the ``experiments.py`` file. An orbit and timestack can be specified here, with an example shown for variables ``orbit`` and ``timestack``. Modified reference code from ``moon_orbit.py`` is used to propogate the desired orbit. 

When ``moon_orbit.py`` is run alone, a specified orbit can be propogated and used to produce crater centre correspondences using the Robbins Catalogue. Running this script individually will also generate crater projection images according to the specified inputs. 

The developed algorithm itself is contained in ``solver.py``. The ``optimise_orbit`` function should be called which can handle cases both with and without camera intrinsic parameters. This function takes the following inputs:

- ``Guess: List``: The initial guess of orbital parameters with list items of the following values:
    - Semi-major axis
    - Eccentricity
    - Inclination
    - Right ascension of the ascending node
    - Argument of the periapsis
    - True anomaly of frame 1
    - True anomaly of final frame (num frames extracted from point correspondence)
    - FOV (Optional -> if optimising intrinsics)
    - Image width (Optional -> if optimising intrinsics)
    - Image height (Optional -> if optimising intrinsics)
- ``Points_3D: List``: List of 3D points corresponding to visible crater centres for each frame (see ``read_input.read_data_from_df``)
- ``Points_2d: List``: List of 2D crater centre detections for each frame (see ``read_input.read_data_from_df``)
- ``Extrinsics: List``: Extrinsic matrix for each camera frame, used to extract rotation (see ``read_input.read_data_from_df``)
- ``K: (Optional) List``: Ground-truth FOV, img_width, img_height if not optimising intrinsics. 
- ``Errors: (Callback) List``: Empty list for storing reprojection error during optimisation for later graphing. 