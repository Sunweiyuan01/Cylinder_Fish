# Cylinder_Fish

Open-source CFD-coupled reinforcement learning framework for fish-like robotic locomotion in cylinder-wake environments.

This repository provides the implementation used for studying fish-like robotic swimming under obstacle-generated unsteady flows. The project integrates CFD-based simulation, reinforcement learning, hydrodynamic interaction, and flow-aware control, with a focus on locomotion around a cylinder wake.

The codebase currently includes the baseline control framework, simulation environment, training pipeline, parallel execution modules, LSTM-related network components, illustrative framework figures, and demo results.

---
<p align="center">
  <img src="cover.png" width="500">
</p>
## Overview

The goal of this project is to train and evaluate fish-like locomotion controllers in CFD environments with obstacle-induced wakes. In particular, the repository focuses on how robotic swimmers interact with cylinder-generated vortical flow structures and how control policies can be learned through CFD-coupled reinforcement learning.

The main workflow of the project includes:

1. **CFD case preparation** for fish locomotion and wake interaction;
2. **Environment construction** through a Fluent-coupled Python interface;
3. **Policy training** using reinforcement learning;
4. **Parallel rollout and acceleration** for efficient simulation-based learning;
5. **Sequence-aware modeling** using LSTM-related network components when required;
6. **Demonstration and visualization** of learned swimming behavior.

---

## Framework illustration

The overall framework and control pipeline are illustrated in the following figures:

### Figure 1
![Framework 1](1.png)

### Figure 2
![Framework 2](2.png)

These figures provide a schematic explanation of the CFD-coupled control framework, including the interaction between the fluid solver, the reinforcement learning environment, the control policy, and the resulting locomotion behavior.

---

## Demo

The `Demo/` folder contains visualization or demonstration results of the learned locomotion behaviors.

Typical contents may include:

- swimming trajectories,
- wake interaction results,
- policy behavior demonstrations,
- videos or image sequences for qualitative comparison.

This folder is intended to help readers quickly understand the practical effect of the control framework.
---

## Repository structure

```text
Cylinder_Fish/
├── 1.png                          # Framework illustration
├── 2.png                          # Framework illustration
├── Demo/                          # Demo results and visualizations
├── Baseline_control/              # Baseline control framework
│   ├── fishmove/                  # CFD simulation cases and related resources
│   ├── EnvFluent.py               # Fluent-coupled simulation environment
│   ├── training.py                # Main training script
│   ├── main_parallel_launcher.py  # Parallel rollout / multi-process execution
│   ├── ...                        # LSTM-related modules and other utilities
└── README.md
