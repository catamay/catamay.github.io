---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: posts
title: Differential Drive LQR and Filtering
permalink: /projects/differential-drive/
description: Le vent se lève! Il faut tenter de vivre
classes: wide
author_profile: true
---

## Background

1. A differential drive robot is a type of mobile robot that uses two independently driven wheels to achieve movement. Control input is given by the tangential velocity and the rotational velocity of the robot. For linear-quadratic regulator (LQR) control, the system dynamics can be linearized around a nominal trajectory, and the LQR controller can be designed to minimize a quadratic cost function that penalizes deviations from the desired trajectory and control effort.
2. Filtering techniques, such as Kalman filters or particle filters, are often used in robotics to estimate the state of the robot (e.g., position, velocity) based on noisy sensor data. These filters help in improving the accuracy of the robot's state estimation, which is crucial for effective control and navigation.

## Results

1. Implemented LQR control for a differential drive robot, demonstrating the ability to follow a desired trajectory while minimizing control effort.
2. Applied Unscented Kalman Filter (UKF) for state estimation, showing improved accuracy in estimating the robot's position and velocity compared to a standard Kalman filter, especially in the presence of non-linearities in the system dynamics.
3. Used quintic hermite splines for trajectory generation, allowing for smooth and continuous trajectories that can be effectively tracked by the LQR controller.


### Trajectory Results

After a spline trajectory is generated, the simulated robot (in python IR-Sim) was able to follow the trajectory with the LQR controller, demonstrating the effectiveness of the control strategy in maintaining the desired path while accounting for system dynamics and noise.
![Spline trajectory over 5 waypoints arriving to the final destination.]({{ site.baseurl }}/assets/image/lqr%20path.png)

Additionally, the following time-series data was collected, showing relative tracking error in conjunction with filtering errors up to one standard deviation.
![Time-series data showing relative tracking error and filtering errors.]({{ site.baseurl }}/assets/image/lqr%20result.png)
