---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: posts
title: Lyapunov Trajectory Learning
permalink: /projects/trajectory/
description: Le vent se lève! Il faut tenter de vivre
classes: wide
author_profile: true
---

This collection of examples serves to showcase a deep learning method for learning an energy function associated with a stable dynamical system, following [2309.08849](https://arxiv.org/abs/2309.08849) on the ArXiv. The codebase and notebooks can be found on my page [here](https://github.com/catamay/Lyapunov-Function-Learning)

## Background

1. An autonomous dynamical system is characterized by $\dot{x} = f(x)$ where $f(x_0)=0$. Without loss of generality, $x_0=0$ with a translation $\tilde{x}= x-x_0$. 
2. A stable system is one where $\lim_{t\to\infty} x(t) = x_0$.

## Use Cases

Learning a dynamical system allows for more seamless trajectory learning and control for repeated tasks, such as robot writing or manufacturing, where the ending position is consistent.
1. Repeated Task -> Learn Energy -> Offline position/velocity from arbitrary starting point.
2. Couple lyapunov function $V$ with controller $u$ assuming control-affine systems $\dot{x} = f(x) + g(x)u$.

## Results

There were two examples presented in this series of experiments:
1. Using a known dynamical system ($\dot{x} = Ax + x \odot \sin(Ax)$, where $A$ is a known stable linear system, and $\sin$ of a vector is understood to mean $\sin(x) = [\sin(x_1), \sin(x_2)]^T$).
2. Using the PyLASA dataset to learn trajectories at different starting points based on demos.

Both systems used the following parameters:
1. Batch size of 64
2. AdamW Optimizer with learning rate `1e-5` and weight decay `0.99`.

All results can be reproduced in the provided notebooks.


### Known Dynamical System


#### Data Simulation
1. Redone every 128 epochs to decrease overfitting
2. Performed by forward Euler on the provided dynamical system with `dt=0.01` over 1000 steps. 
3. Simulated rough environment with small amounts of slippage on position sensors.

#### Training Loss Over Time

![Losses over 1000 epochs. The first 100 epochs have a steep decline followed by little visible change for the remaining training. Final training loss was 0.01 (summed against 16 batches).](https://github.com/catamay/Lyapunov-Function-Learning/blob/master/images/Known%20DS%20Loss.png)

#### Trajectory Results

After training, the energy landscape, vector field, and random trajectories are plotted.
![Left image depicts the Lyapunov energy function landscape with the respective vector field showing 10 computed trajectories plotted against learned trajectories. Right image depicts the learned transformed trajectories from the function y associated with the energy.](https://github.com/catamay/Lyapunov-Function-Learning/blob/master/images/images/known%20DS.png)


### LASA Dataset

#### Data Collection

1. Training:Evalulation ratio is 6:1.
2. Training trajectories divided into 990 sub-trajectories of length 10. I.e. $[x_i, v_i, t_i], [x_{i+1}, v_{i+1}, t_{i+1}],\dots, [x_{i+9}, v_{i+9}, t_{i+9}]$.
3. Data simulation performed with forward Euler for 10 steps each batch.

#### Training Loss Over Time

![Losses over 150 epochs. The first 10 epochs after a significant jump have a steep decline followed by gradual change for the remaining training. Final training loss was 1.1 (summed against 16 batches).](https://github.com/catamay/Lyapunov-Function-Learning/blob/master/images/images/LASA%20Loss.png)

#### Trajectory Results

After training, the energy landscape, vector field, and random trajectories are plotted.
![Left image depicts the Lyapunov energy function landscape with the respective vector field showing 10 computed trajectories plotted against learned trajectories. Right image depicts the learned transformed trajectories from the function y associated with the energy.](https://github.com/catamay/Lyapunov-Function-Learning/blob/master/images/images/LASA%20DS.png)


## Citation

Zhang et al., *Learning a Stable Dynamic System with a Lyapunov Energy Function for Demonstratives Using Neural Networks*, 2024,  	arXiv:2309.08849